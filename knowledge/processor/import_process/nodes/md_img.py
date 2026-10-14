'''
markdown 图片处理节点
'''
import re
from collections import deque
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Tuple, List, Dict, Deque
from threading import Lock

import base64
import time

import json

from gradio.themes import origin
from ray._private.thirdparty.pathspec import pattern

from knowledge.processor.import_process import config
from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import StateFieldError, FileProcessingError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients

from dotenv import load_dotenv
load_dotenv()

# ── 数据模型 ──

@dataclass
class ImageContext:
    """图片在 MD 中的上下文信息。"""
    heading: str      # 最近的章节标题
    pre_text: str     # 图片上方的正文内容
    post_text: str    # 图片下方的正文内容


@dataclass
class ImageInfo:
    """一张图片的完整信息。"""
    name: str                # 图片文件名，如 "abc123.jpg"
    path: str                # 图片完整路径
    context: ImageContext    # 在 MD 中的上下文

#-------- 1. 文件读写 & 备份--------
class MdFileHandler:
    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name
    def read_md(self, state: ImportGraphState) -> Tuple[str, Path, Path]:
        md_path = state.get("md_path", "")
        md_path = Path(md_path)# 1.校验文件存在（self.md_path 是 Path，可直接 .exists()）
        if not md_path:
            raise StateFieldError(self.node_name, "md_path", str, "md_path not found")
        # 2.读取 md 文件
        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            raise FileProcessingError(self.node_name, md_path, "File not found")
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        # 3.计算图片目录（pathlib 风格）
        images_dir = md_path_obj.parent / 'images'
        return md_content, md_path_obj, images_dir

    def backup_md(self, md_path_obj, new_md_content):
        backup_md_path = md_path_obj.with_name(md_path_obj.stem + '_new.md')
        with open(backup_md_path, 'w', encoding='utf-8') as f:
            f.write(new_md_content)


#--------2.图片扫描上下文--------
class ImageScanner:
    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def scan_img_dir(self, md_content: str, images_dir: Path, image_extensions: set, img_content_length: int=200) -> List[ImageInfo]:
        '''扫描md文档，找到每一个图片的上下文信息，返回图片列表
        :param md_content: md文档内容
        :param images_dir: 图片目录
        :param image_extensions: 图片扩展名集合
        :param img_content_length: 图片上下文长度
        :return: 图片列表List[ImageInfo]
        '''
        image_info_list: List[ImageInfo] = []
        for img_path in images_dir.iterdir():
            if not img_path.is_file():
                continue
            if not img_path.suffix.lower() in image_extensions:
                continue
            #查找图片上下文，如果表格图片，在md_content中可能找不到上下文，则返回None
            ctx = self._find_context(md_content, img_path.name, img_content_length)
            if ctx is None:
                continue
            image_info_list.append(ImageInfo(img_path.name, str(img_path), ctx))
        return image_info_list

    def _find_context(
            self, md_content: str, img_name: str, max_chars: int = 200
    ) -> ImageContext | None:
        """返回图片在 MD 中第一次出现位置的上下文，找不到返回 None。"""
        pattern = re.compile(
            r"!\[.*?\]\(.*?" + re.escape(img_name) + r".*?\)"
        )
        md_lines = md_content.split("\n")

        for line_idx, line in enumerate(md_lines):
            if not pattern.search(line):
                continue

            # 向上：找最近标题，取标题到图片之间的内容作为上文
            prev_title, prev_boundary = self._find_heading_above(
                md_lines, line_idx
            )
            pre_content = md_lines[prev_boundary + 1: line_idx]
            img_pre = self._extract_limited_context(
                pre_content, max_chars, direction="front"
            )

            # 向下：找下一个标题，取图片到标题之间的内容作为下文
            next_boundary = self._find_heading_below(md_lines, line_idx)
            post_content = md_lines[line_idx + 1: next_boundary]
            img_post = self._extract_limited_context(
                post_content, max_chars, direction="end"
            )

            return ImageContext(
                heading=prev_title,
                pre_text=img_pre,
                post_text=img_post,
            )

        return None

    @staticmethod
    def _find_heading_above(
            md_lines: List[str], from_idx: int
    ) -> Tuple[str, int]:
        """从 from_idx 向上查找最近的标题。"""
        for i in range(from_idx - 1, -1, -1):
            if re.match(r"^#{1,6}\s+", md_lines[i]):
                return md_lines[i], i
        return "", -1

    @staticmethod
    def _find_heading_below(md_lines: List[str], from_idx: int) -> int:
        """从 from_idx 向下查找下一个标题。"""
        for i in range(from_idx + 1, len(md_lines)):
            if re.match(r"^#{1,6}\s+", md_lines[i]):
                return i
        return len(md_lines)

    @staticmethod
    def _extract_limited_context(
            lines: List[str], max_chars: int, direction: str
    ) -> str:
        """按段落分割，按 direction 方向贪心装填，保持段落完整性。"""
        current_paragraph: List[str] = []
        paragraphs: List[str] = []

        for line in lines:
            # line.strip(): 去除字符串首尾的空白字符(空格、制表符、换行符等)
            is_blank_line = not line.strip()
            is_other_image = re.match(
                r"^!\[.*?\]\(.*?\)$", line.strip()
            )

            if is_blank_line or is_other_image:
                if current_paragraph:
                    paragraphs.append("\n".join(current_paragraph))
                    current_paragraph = []
                continue

            current_paragraph.append(line)

        if current_paragraph:
            paragraphs.append("\n".join(current_paragraph))

        if direction == "front":
            paragraphs.reverse()  # 就近原则

        total = 0
        selected: List[str] = []
        for para in paragraphs:
            if (total + len(para) > max_chars) and selected:  # 至少有个段落
                break
            selected.append(para)
            total += len(para)

        if direction == "front":
            selected.reverse()  # 与原文顺序一致，利于VLM

        return "\n\n".join(selected)  # 折行并空一行


#--------3.VLM生成摘要--------
class VLMSummarizer:
    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name
        self.stamps_deque:Deque[float] = deque()
        self.lock:Lock = Lock()  # 创建一个锁对象

    def summarizer_all(self, doc_name: str, image_list: List[ImageInfo], vl_model:str, requests_per_minute:int=5)-> Dict[str, str]:
        """通过视觉语言模型为每张图片生成中文标题/摘要。
        :param doc_name: 文档名称
        :param image_list: 图片列表
        :param vl_model: 视觉语言模型
        :param requests_per_minute: 限流
        :return: 图片摘要字典   {'image_name.jpg': '图片摘要'}
        """
        summaries: Dict[str, str] = {}
        # stamps_deque:Deque[float] = deque() #不能每次请求都创建一个新的队列，多个请求共用一个队列
        try:
            openai_client = AIClients.get_openai()
        except Exception as e:
            #降级处理
            self.logger.warning(f"视觉语言模型调用异常，将使用降级处理：{e}")
            for image_info in image_list:
                summaries[image_info.name] = '默认图片摘要'
            return summaries
        for image_info in image_list:
            # 滑动窗口限流处理
            self._enforce_rate_limit(requests_per_minute)
            # 调用视觉语言模型生成图片摘要
            summary = self._summaries_one(image_info, openai_client, vl_model,doc_name)
            summaries[image_info.name] = summary

        return summaries

    def _summaries_one(self, image_info: ImageInfo, openai_client, vl_model, doc_name: str):
        """为单张图片生成中文标题/摘要。
        :param image_info: 图片信息
        :param openai_client: openai_client
        :param vl_model: 视觉语言模型
        :param doc_name: 文档名称
        :return: 图片摘要
        """
        final_context = (image_info.context.heading + "\n" +
                         image_info.context.pre_text + "\n" +
                         image_info.context.post_text)
        # 图片转 Base64
        with open(image_info.path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")
        reasoning_content = ""  # 定义完整思考过程
        answer_content = ""  # 定义完整回复
        is_answering = False  # 判断是否结束思考过程并开始回复
        enable_thinking = False
        # 创建聊天完成请求
        completion = openai_client.chat.completions.create(
            model=vl_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                f"任务：为Markdown文档中的图片生成一个简短的中文标题。\n"
                                f"背景信息：\n"
                                f"  1. 所属文档标题：\"{doc_name}\"\n"
                                f"  2. 图片上下文：{final_context}\n"
                                f"请结合图片内容和上述上下文信息，"
                                f"用中文简要总结这张图片的内容，"
                                f"生成一个精准的中文标题（不要包含图片二字）。"
                            ),
                         },
                    ],
                },
            ],
            stream=True,
            # enable_thinking 参数开启思考过程，thinking_budget 参数设置最大推理过程 Token 数
            extra_body={
                'enable_thinking': False,
                "thinking_budget": 81920},

            # 解除以下注释会在最后一个chunk返回Token使用量
            # stream_options={
            #     "include_usage": True
            # }
        )

        if enable_thinking:
            print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

        for chunk in completion:
            # 如果chunk.choices为空，则打印usage
            if not chunk.choices:
                print(chunk.usage)
            else:
                delta = chunk.choices[0].delta
                # 打印思考过程
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                    reasoning_content += delta.reasoning_content
                else:
                    # 开始回复
                    if delta.content != "" and is_answering is False:
                        is_answering = True
                    # 打印回复过程
                    answer_content += delta.content

        # print("=" * 20 + "完整思考过程" + "=" * 20 + "\n")
        # print(reasoning_content)
        # print("=" * 20 + "完整回复" + "=" * 20 + "\n")
        # print(answer_content)
        return answer_content

    def _enforce_rate_limit(self, requests_per_minute, window:int = 60):
        '''滑动窗口限流处理
        :param stamps_deque: 存取每次请求的时间戳队列
        :param requests_per_minute: 每分钟请求次数限制
        '''
        with self.lock:
            stamps_deque = self.stamps_deque
            while len(stamps_deque) > 0 and (time.time() - stamps_deque[0]) > window:
                stamps_deque.popleft() #删除首元素
            if len(stamps_deque) >= requests_per_minute:
                sleep_duration = (stamps_deque[0] + window - time.time())
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
                while len(stamps_deque) > 0 and (time.time() - stamps_deque[0]) > window:
                    stamps_deque.popleft()  # 删除首元素

            stamps_deque.append(time.time())

#--------4.图片存储minio & 替换md--------
class ImageUploader:
    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def upload_and_replace(self, doc_name, md_content, image_list, summaries, minio_bucket, minio_endpoint):
        '''
        上传图片到minio，替换md中的图片
        :param doc_name:   文档名称,用于minio存储的文件夹名称,不带扩展名
        :param md_content: md内容
        :param image_list: 图片列表
        :param summaries: 图片摘要的字典
        :param minio_bucket: minio存储的桶名称
        :param minio_endpoint: minio访问地址
        :return:
        '''
        # 1.图片上传到minio
        remote_urls:Dict[str, str] = self._upload_all(doc_name,image_list, minio_bucket, minio_endpoint)
        # 2.图片替换 摘要+路径
        new_md_content = self._replace_in_md(remote_urls, md_content, image_list, summaries)
        return new_md_content

    def _upload_all(self, doc_name, image_list, minio_bucket, minio_endpoint) -> Dict[str, str]:
        '''
        上传图片到minio
        :param doc_name:    文档名称,用于minio存储的文件夹名称,不带扩展名
        :param image_list:  图片列表
        :param minio_bucket:    minio存储的桶名称
        :param minio_endpoint:  minio访问地址
        :return:    {'p1.jpg': 'http://192.168.6.150:9000/mybucket/images/p1.jpg'}
        '''
        remote_urls:Dict[str, str] = {}
        try:
            minio_client = StorageClients.get_minio_client()
        except Exception as e:
            # 降级处理
            for image_info in image_list:
                remote_urls[image_info.name] = image_info.path
            return remote_urls
        # 循环上传图片到minio
        for image_info in image_list:
            # 上传文件
            try:
                minio_client.fput_object(
                    bucket_name=minio_bucket,
                    object_name=f"images/{doc_name}/{image_info.name}",  # Object名称含路径
                    file_path=image_info.path,  # 上传的本地文件路径
                    content_type="image/jpeg"  # MIME类型
                )
                # 上传后访问地址
                url = f"{minio_endpoint}/{minio_bucket}/images/{doc_name}/{image_info.name}"
                remote_urls[image_info.name] = url
            except Exception as e:
                #二次降级处理
                self.logger.error(f"上传图片 {image_info.name} 失败: {e}")
                remote_urls[image_info.name] = image_info.path
        return remote_urls

    def _replace_in_md(self,remote_urls, md_content, image_list, summaries) -> str:
        '''
        替换md中图片的路径和摘要 ！[图片摘要](图片访问地址)
        :param remote_urls:   图片名称对应访问地址的字典
        :param md_content:    md内容
        :param image_list:    图片列表
        :param summaries:     图片名称和摘要的字典
        :return: 替换后的md内容
        '''
        pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")
        def replace_match(match):
            origin_url = match.group(2) # 原始图片路径
            image_name_in_md = Path(origin_url).name
            for img_name,summary in summaries.items():
                if img_name == image_name_in_md:
                    return f"![{summary}]({remote_urls[img_name]})" # 替换后的摘要+访问地址
            return match.group(0) # 未找到摘要，不替换，返回原始内容
        return pattern.sub(replace_match, md_content)

class MarkDownImageNode(BaseNode):
    '''
    上传md或pdf转md后，需要对md中的图片进行处理
    1.图片上传到minio
    2.图片描述 VLM(视觉语言模型) 千问
    3.图片上传进度 限流
    4.图片替换 摘要+路径
    5.备份
    '''
    name = "md_img_node"
    def __init__(self):
        super().__init__()
        self.md_file_handler = MdFileHandler(self.logger,self.name)
        self.image_scanner = ImageScanner(self.logger,self.name)
        self.vlm_summarizer = VLMSummarizer(self.logger,self.name)
        self.image_uploader = ImageUploader(self.logger,self.name)

    def process(self, state:ImportGraphState) -> ImportGraphState:
        # 1.文件处理：获取整个md文档， 获取图片路径
        md_content,md_path_obj, images_dir = self.md_file_handler.read_md(state)
        if not images_dir.exists():
            self.logger.warning(
                f"文件 {md_path_obj.name} 暂无图片要处理"
            )
            state['md_content'] = md_content
            return state
        # 2.获取图片上下文
        image_list: list[ImageInfo] = self.image_scanner.scan_img_dir(
            md_content,
            images_dir,
            image_extensions=self.config.image_extensions,
            img_content_length=self.config.img_content_length
        )
        # 3. VLM生成图片摘要
        summaries:Dict[str, str] = self.vlm_summarizer.summarizer_all(md_path_obj.stem, image_list, self.config.vl_model,self.config.requests_per_minute)
        print(summaries)
        # 4.图片上传到minio,替换md中的图片路径,插入图片摘要
        new_md_content:str = self.image_uploader.upload_and_replace(doc_name= md_path_obj.stem,
                                               md_content=md_content,
                                               image_list=image_list,
                                               summaries=summaries,
                                               minio_bucket=self.config.minio_bucket,
                                               minio_endpoint=self.config.get_minio_base_url())
        # 5.备份替换后的md文件
        self.md_file_handler.backup_md(md_path_obj, new_md_content)
        state['md_content'] = new_md_content
        return state
if __name__ == "__main__":
    setup_logging()

    node = MarkDownImageNode()
    state = {
        "md_path": r"E:\zw\project\file\temp_dir\万用表RS-12的使用_online\full.md"
    }
    process_state = node(state)
    print(json.dumps(process_state, indent=4, ensure_ascii=False))