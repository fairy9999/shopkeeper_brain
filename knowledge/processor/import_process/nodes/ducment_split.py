import os
import re
import json
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from knowledge.processor.import_process import config
from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.markdown_util import MarkdownTableLinearizer


class DocumentSplitNode(BaseNode):
    '''
        文档拆分节点
        
        职责:
            将导入流程中已解析的文档内容按照指定策略（如长度、段落、分隔符等）
            拆分为若干文本块（chunks），并为每个文本块附加来源、序号等元数据，
            以便后续向量化与检索使用。
        输入 (ImportGraphState):
            解析后的文档内容及相关的导入任务信息
        输出 (ImportGraphState):
            拆分后的文本块列表，写回状态中传递给下游节点

        面试题：为什么要文档拆分？答案：
            文档拆分的目的是为了让文本块更容易被检索和理解，
            同时也方便后续的向量化和检索。
            文档拆分的策略可以根据不同的应用场景和需求进行调整，
            比如长度、段落、分隔符等。

    '''
    name = "document_split_node"
    def process(self, state:ImportGraphState) -> ImportGraphState:
        '''
        文档拆分处理
            1. 获取输入参数并校验 md_content,file_title,max_content_length,min_content_length
            2. 按照标题进行拆分：按\n获取文档行的集合。遍历、判断是否存在代码围栏，正则表达式识别标题；构建section
                sections=[{title:str,parent_title:str,file_title:str, body:str}]
            3. 长切短合：
                a.长文档切分:如果文档长度大于max_content_length，则进行二次切分
                b.短文档合:如果文档长度小于min_content_length，则与前一个文档合并（贪婪算法）
            4. 组装最终的chunks
                chunks=[{title:str,parent_title:str,file_title:str, content:title+body}]
            5.日志统计
            6.备份切片（可选，为了自己测试）
        '''
        #1. 获取输入参数并校验 md_content,file_title,max_content_length,min_content_length
        md_content, file_title, max_content_length, min_content_length = self._validate_get_input(state)
        #2. 按照标题进行拆分
        sections:List[Dict[str, Any]] = self._split_by_title(md_content, file_title)
        #3. 长切短合
        final_sections = self._split_and_merge(sections, max_content_length, min_content_length)
        #4. 组装最终的chunks
        chunks = self._assemble_chunk(final_sections)
        state["chunks"] = chunks
        #5.日志统计
        self._log_summary(md_content, chunks, max_content_length)
        #6.备份切片（可选，为了自己测试）
        self._backup_chunks(state, chunks)

        #7. 返回状态
        state["chunks"] = chunks

        return state
    #============================================================
    # 1.验证并获取输入参数
    #============================================================
    def _validate_get_input(self, state):
        md_content = state.get("md_content")
        file_title = state.get("file_title")
        max_content_length = self.config.max_content_length
        min_content_length = self.config.min_content_length
        if not md_content:
            raise ValidationError("md_content is required",self.name)
        if not file_title:
            raise ValidationError("file_title is required",self.name)
        if max_content_length <= 0:
            raise ValidationError("max_content_length must be greater than 0",self.name)
        if min_content_length <= 0:
            raise ValidationError("min_content_length must be greater than 0",self.name)
        if min_content_length > max_content_length:
            raise ValidationError("min_content_length must be less than max_content_length",self.name)
        return md_content, file_title, max_content_length, min_content_length

    # ============================================================
    # 2.按标题拆分文档（初切）
    # ============================================================
    def _split_by_title(self, md_content, file_title):
       '''sections=[{title:str,parent_title:str,file_title:str, body:str}]'''
       sections:List[Dict[str, Any]] = []
       heading_re = re.compile(r"^\s*(#{1,6})\s+(.+)")
       body_lines = []
       title = ""
       parent_title = ""
       # 存放标题层级，["", "# 标题", "## 标题", "### 标题", "#### 标题", "##### 标题", "###### 标题"]
       hierarchy = [""]*7
       current_level = 0
       in_fence = False # 是否在代码围栏中

       # 获取文档行的集合
       lines = md_content.split("\n")

       def _flush():
           body = "\n".join(body_lines)
           if current_level or body:
               parent_title = ""
               for i in range(current_level - 1, 0, -1):
                   if hierarchy[i]:
                       parent_title = hierarchy[i]
                       break

               if not parent_title:
                   parent_title = title if title else file_title

               section = {"title": title if title else file_title,
                          "parent_title": parent_title,
                          "file_title": file_title,
                          "body": body
                          }
               sections.append(section)

       for line in lines:
           # 判断是否存在代码块围栏
           if line.strip().startswith("```") or line.strip().startswith("~~~"):
               in_fence = not in_fence

           match = heading_re.match(line) if not in_fence else None

           if match:# 匹配到标题行,封装section
               _flush()
               title = line
               parent_title = ""
               level = len(match.group(1)) #获取#数量，
               current_level = level
               hierarchy[level] = line
               # 清空下级标题(因为是上个章节遗留的标题)
               for i in range(level + 1, 7):
                   hierarchy[i] = ""

               body_lines = [] # 清空上一个section的正文内容
           else:# 非标题行，属于正文内容，装箱
               body_lines.append(line)
       _flush()

       return sections

    #============================================================
    # 3.长切短合（二次切分和合并）
    # ============================================================
    def _split_and_merge(self, sections:List[Dict[str, Any]],
                         max_content_length:int=1000, min_content_length:int=200) -> List[Dict[str, Any]]:
        """
               二次切分和合并

               Args:
                   sections: 根据一级标题切分后的所有 section（章节）块
                   max_content_length: 每一个 section 的 content 内容最大长度
                   min_content_length: 触发合并的最小长度
               """
        self.log_step("step3", "切分及合并...")

        # 1. 切分
        current_sections = []
        for section in sections:
            current_sections.extend(self.split_long_section(section, max_content_length))

        # 2. 合并
        final_sections = self.merge_short_section(current_sections, min_content_length)

        # 3. 返回
        return final_sections

    # ============================================================
    # 3.1长切
    # ============================================================
    def split_long_section(self, section: Dict[str, Any], max_content_length: int):
        """
        对超长章节进行二次切分
        """
        self.log_step("step3", "进行长内容的切分")

        # 1. 获取 section 对象属性
        title = section.get('title')
        body = section.get('body')
        file_title = section.get('file_title')
        parent_title = section.get('parent_title')

        # 2. 判断表格
        if "<table>" in body:
            self.logger.info("检测到了表格数据...")
            body = MarkdownTableLinearizer.process(body)
            section['body'] = body

        # 3. 对标题做校验
        TITLE_MAX_LENGTH = 50
        if len(title) > TITLE_MAX_LENGTH:
            self.logger.warning(f"检测文件{file_title}对应的{title}长度过长...")
            title = title[:50]

        # 4. 拼接 title 前缀
        title_prefix = f"{title}\n\n"

        # 5. 计算总长度
        total_length = len(title_prefix) + len(body)

        # 6. 判断是否需要切分
        if total_length <= max_content_length:
            return [section]

        # 7. 计算 body 可用的长度
        body_length = max_content_length - len(title_prefix)

        if body_length <= 0:
            return [section]

        # 8. 使用 RecursiveCharacterTextSplitter 切分
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=body_length,
            chunk_overlap=0,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""],
            keep_separator=False
        )
        texts = text_splitter.split_text(body)

        # 8.3 判断切分结果
        if len(texts) <= 1:
            return [section]

        sub_section = []
        for index, text in enumerate(texts):
            sub_section.append({
                "title": title + "-" + f"{index + 1}",
                "body": text,
                "file_title": file_title,
                "parent_title": parent_title,
                "part": f"{index + 1}"  # 标记序号
            })

        return sub_section

    # ============================================================
    # 3.2短合
    # ============================================================
    def merge_short_section(self, current_sections: List[Dict[str, Any]],
                            min_content_length: int):
        """
        贪心累加算法合并过短的章节

        局限性：
        1. 撑破最小的阈值：大一点不用管
        2. 孤儿小块：也不用管（大量都是小块）
        """
        # 1. 定义变量
        current_section = current_sections[0]
        final_sections = []  # 最终的箱子

        # 2. 遍历以及合并
        for next_section in current_sections[1:]:
            # 同源检查
            same_parent = (current_section['parent_title'] == next_section['parent_title'])

            if same_parent and len(current_section.get('body')) < min_content_length:
                # body的合并
                current_section['body'] = (
                        current_section.get('body').rstrip() + "\n\n" + next_section.get('body').lstrip()
                )
                # 更新 current_title
                current_section['title'] = current_section['parent_title']
                current_section['part'] = 0
            else:
                # 将原来 current_section 进行封箱
                final_sections.append(current_section)
                # 更新 next_section
                current_section = next_section

        # 最后一个（封装起来）
        final_sections.append(current_section)

        # 4. 对所有 section 的 part 做处理
        part_counter = {}
        result = []
        for final_section in final_sections:
            if "part" in final_section:
                parent_title = final_section.get('parent_title')
                part_counter[parent_title] = part_counter.get(parent_title, 0) + 1
                new_part = part_counter[parent_title]
                final_section['part'] = new_part
                final_section['title'] = final_section['title'] + f"- {new_part}"

            result.append(final_section)

        return result

    # ------------------------------------------------------------------ #
    #               Step 4: 组装最终 chunk                                 #
    # ------------------------------------------------------------------ #
    def _assemble_chunk(self, final_sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """最终组合 chunk"""
        self.log_step("step4", "组装最终的切片信息...")
        chunks = []

        for chunk in final_sections:
            # 1. 获取 chunk 的信息
            title = chunk.get('title')
            file_title = chunk.get('file_title')
            parent_title = chunk.get('parent_title')
            body = chunk.get('body')
            content = f"{title}\n\n{body}"

            # 2. 构建最终 chunk 对象
            assemble_chunk = {
                "title": title,
                "file_title": file_title,
                "parent_title": parent_title,
                "content": content,
            }

            # 3. 判断 part 是否存在
            if "part" in chunk:
                assemble_chunk['part'] = chunk.get('part')

            chunks.append(assemble_chunk)

        return chunks

    # ------------------------------------------------------------------ #
    #                       日志 & 备份                                    #
    # ------------------------------------------------------------------ #

    def _log_summary(self, raw_content: str, chunks: List[dict], max_length: int):
        """输出切分统计信息"""
        self.log_step("step5", "输出统计")

        lines_count = raw_content.count("\n") + 1
        self.logger.info(f"原文档行数: {lines_count}")
        self.logger.info(f"最终切分章节数: {len(chunks)}")
        self.logger.info(f"最大切片长度: {max_length}")

        if chunks:
            self.logger.info("章节预览:")
            for i, sec in enumerate(chunks[:5]):
                title = sec.get("title", "")[:30]
                self.logger.info(f"  {i + 1}. {title}...")
            if len(chunks) > 5:
                self.logger.info(f"  ... 还有 {len(chunks) - 5} 个章节")

    def _backup_chunks(self, state: ImportGraphState, sections: List[dict]):
        """将切分结果备份到 JSON 文件"""
        self.log_step("step6", "备份切片")

        local_dir = state.get("file_dir", "")
        if not local_dir:
            self.logger.debug("未设置 file_dir，跳过备份")
            return

        try:
            os.makedirs(local_dir, exist_ok=True)
            output_path = os.path.join(local_dir, "chunks.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sections, f, ensure_ascii=False, indent=2)
            self.logger.info(f"已备份到: {output_path}")
        except Exception as e:
            self.logger.warning(f"备份失败: {e}")

if __name__ == '__main__':
    setup_logging()

    node = DocumentSplitNode()
    file_path = r"E:\zw\project\file\temp_dir\万用表RS-12的使用_online\万用表RS-12的使用_new.md"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    state = {
        "file_title": "万用表的使用",
        "md_content": content,
        "file_dir": r"E:\zw\project\file\temp_dir\万用表RS-12的使用_online"
    }

    node.process(state)







