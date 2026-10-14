"""
PDF转Markdown节点：调用 MinerU 在线 API 完成转换

调用流程（精准解析 API，参考 https://mineru.net/apiManage/docs）：
    1. POST /api/v4/file-urls/batch       申请上传链接
    2. PUT  {file_url}                    上传 PDF 文件
    3. GET  /api/v4/extract-results/batch/{batch_id}  轮询直到 state=done
    4. 下载 full_zip_url 指向的 ZIP 包
    5. 解压到 file_dir/{title}_online/，定位 full.md 写入 state["md_path"]

为什么改用在线 API：
- 本地需要下 3GB 模型 + 装 CUDA 版 PyTorch，对 6GB 显存的 RTX 2060 不友好
- 在线 VLM 模型解析质量优于本地 pipeline
- 每日 1000 页免费额度，产品说明书场景足够

安全提示：
- Token 走 .env 配置（已被 .gitignore 忽略），禁止硬编码
- 如不慎泄露，到 https://mineru.net/apiManage/token 撤销
"""
import os
import re
import time
import zipfile
from pathlib import Path
from typing import Dict, Optional

import requests

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.exceptions import (
    StateFieldError,
    PdfConversionError,
    ConfigurationError,
)
from knowledge.processor.import_process.state import ImportGraphState

# 轮询超时（秒）：按万用表那种 5-20 页 PDF 经验，120s 足够
POLL_TIMEOUT_SECONDS = 600
# 轮询间隔（秒）
POLL_INTERVAL_SECONDS = 3
# 单次 HTTP 请求超时
HTTP_TIMEOUT_SECONDS = 60
# Token 错误码（不重试）
TOKEN_ERROR_CODES = {"A0202", "A0211"}


class PdfToMdNode(BaseNode):
    """PDF 转 Markdown 节点（仅 PDF 分支会执行）"""

    name = "pdf_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1. 校验状态字段
        import_file_path = state.get("import_file_path", "")
        file_dir = state.get("file_dir", "")
        file_title = state.get("file_title", "")
        if not import_file_path or not file_dir:
            raise StateFieldError(
                node_name=self.name,
                field_name="import_file_path / file_dir",
                expected_type=str,
            )

        # 2. 校验 Token 已配置（不打印明文）
        token = self.config.mineru_api_token
        if not token:
            raise ConfigurationError(
                "未配置 MINERU_API_TOKEN（请在 .env 中设置）",
                node_name=self.name,
            )

        # 3. 申请上传链接
        file_name = os.path.basename(import_file_path)
        batch_id, file_urls = self._apply_upload_urls(token, file_name)

        # 4. PUT 上传文件
        self._put_file(file_urls[0], import_file_path)

        # 5. 轮询任务状态直到完成
        zip_url = self._poll_batch_result(token, batch_id)

        # 6. 下载并解压
        extract_dir = os.path.join(file_dir, f"{file_title}_online")
        md_path = self._download_and_extract(zip_url, extract_dir, file_title)
        if not md_path:
            raise PdfConversionError(
                f"解压完成但未找到 full.md，目录: {extract_dir}",
                node_name=self.name,
            )

        state["md_path"] = md_path
        self.log_step("MinerU", f"转换成功: {md_path}")
        return state

    # ------------------------------------------------------------------
    # 1) 申请上传链接
    # ------------------------------------------------------------------
    def _apply_upload_urls(self, token: str, file_name: str):
        """POST /api/v4/file-urls/batch，返回 (batch_id, file_urls)"""
        url = f"{self.config.mineru_api_base}/api/v4/file-urls/batch"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        payload = {
            "enable_formula": True,
            "enable_table": True,
            "language": "ch",
            "model_version": "vlm",  # 在线默认 vlm，质量最好
            "files": [
                {
                    "name": file_name,
                    # data_id 仅允许 字母/数字/_/-/. ，中文文件名需净化
                    "data_id": self._safe_data_id(file_name),
                    # is_ocr 属 file 级参数（文档：files[].is_ocr），非顶层
                    "is_ocr": False,
                }
            ],
        }
        data = self._post(url, headers, payload)
        if data.get("code") != 0:
            self._raise_api_error(data, "申请上传链接")
        batch_id = data["data"]["batch_id"]
        file_urls = data["data"]["file_urls"]
        self.log_step("申请上传链接", f"batch_id={batch_id}")
        return batch_id, file_urls

    # ------------------------------------------------------------------
    # 2) PUT 上传文件
    # ------------------------------------------------------------------
    def _put_file(self, file_url: str, file_path: str) -> None:
        """PUT 上传到 MinerU 返回的临时 URL（24h 有效）"""
        file_size = os.path.getsize(file_path)
        with open(file_path, "rb") as f:
            # 注意：MinerU 要求 PUT 时**不要**带 Content-Type
            resp = requests.put(
                file_url, data=f, timeout=HTTP_TIMEOUT_SECONDS * 2
            )
        # 调试用：打印上传状态码 + ETag + 返回内容
        body = resp.text[:500] if resp.text else "<empty>"
        self.log_step(
            "MinerU HTTP PUT",
            f"status={resp.status_code} size={file_size}B "
            f"etag={resp.headers.get('ETag', 'N/A')} body={body}",
        )
        resp.raise_for_status()
        self.log_step("上传PDF", f"{os.path.basename(file_path)} ok")

    # ------------------------------------------------------------------
    # 3) 轮询批量任务
    # ------------------------------------------------------------------
    def _poll_batch_result(self, token: str, batch_id: str) -> str:
        """GET /api/v4/extract-results/batch/{batch_id}，返回 ZIP 下载 URL"""
        url = (
            f"{self.config.mineru_api_base}"
            f"/api/v4/extract-results/batch/{batch_id}"
        )
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        start = time.time()
        first_poll = True
        while time.time() - start < POLL_TIMEOUT_SECONDS:
            data = self._get(url, headers)
            if data.get("code") != 0:
                self._raise_api_error(data, "查询任务状态")

            task = self._extract_task_payload(data)
            state_name = self._extract_state(task) or "unknown"
            elapsed = int(time.time() - start)

            # 第一次轮询把原始响应 key 打出来，方便排查结构差异
            if first_poll:
                self.log_step(
                    "MinerU 轮询原始结构",
                    f"keys={list(task.keys())}, state={state_name}"
                )
                first_poll = False

            if state_name == "done":
                zip_url = task.get("full_zip_url", "")
                if not zip_url:
                    raise PdfConversionError(
                        "任务完成但未返回 full_zip_url",
                        node_name=self.name,
                    )
                self.log_step("解析完成", f"耗时 {elapsed}s")
                return zip_url

            if state_name == "failed":
                raise PdfConversionError(
                    f"MinerU 解析失败: {task.get('err_msg', 'unknown')}",
                    node_name=self.name,
                )

            # running / pending / converting：打印进度后继续轮询
            progress = task.get("extract_progress") or {}
            cur = progress.get("extracted_pages")
            tot = progress.get("total_pages")
            if cur is not None and tot is not None:
                progress_str = f"{cur}/{tot} 页"
            else:
                progress_str = "进度待返回"
            self.log_step(
                "MinerU 状态",
                f"[{elapsed}s] {state_name} | {progress_str}",
            )
            time.sleep(POLL_INTERVAL_SECONDS)

        raise PdfConversionError(
            f"MinerU 解析超时（>{POLL_TIMEOUT_SECONDS}s）",
            node_name=self.name,
        )

    @staticmethod
    def _extract_task_payload(data: dict) -> dict:
        """兼容不同的 data 结构：data.extract_result / data.tasks[0]"""
        task = data.get("data") or {}
        if isinstance(task, list):
            if not task:
                return {}
            task = task[0]
        # 实际响应：data.data.extract_result 才是真正的任务结果
        if isinstance(task, dict) and "extract_result" in task:
            task = task["extract_result"]
        if isinstance(task, list):
            if not task:
                return {}
            task = task[0]
        return task

    @staticmethod
    def _extract_state(task: dict) -> str:
        """兼容 state/status/task_state 等字段名"""
        for key in ("state", "status", "task_state", "taskStatus"):
            value = task.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @staticmethod
    def _safe_data_id(file_name: str) -> str:
        """data_id 仅允许 字母/数字/_/-/.（文档要求），中文等非法字符替换为 _"""
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", file_name)
        return safe[:128] or "file"

    # ------------------------------------------------------------------
    # 4) 下载并解压 ZIP
    # ------------------------------------------------------------------
    def _download_and_extract(self, zip_url: str, extract_dir: str,
                              file_title: str) -> Optional[str]:
        """下载 ZIP 包并解压，按原文件名重命名内部文件，定位 md 路径返回"""
        os.makedirs(extract_dir, exist_ok=True)
        zip_path = os.path.join(extract_dir, f"{file_title}.zip")

        with requests.get(zip_url, stream=True,
                          timeout=HTTP_TIMEOUT_SECONDS) as resp:
            resp.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        self.log_step("下载ZIP", zip_path)

        # 解压
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        os.remove(zip_path)  # 节省空间
        self.log_step("解压", extract_dir)

        # 按原文件名重命名 MinerU 输出，并同步更新 md 里的图片引用
        md_path = self._rename_extracted_files(extract_dir, file_title)
        return md_path

    def _rename_extracted_files(self, extract_dir: str,
                                file_title: str) -> Optional[str]:
        """把 MinerU 生成的 UUID/默认文件名统一改为 {file_title}_* 形式。"""
        base_dir = Path(extract_dir)
        if not base_dir.exists():
            return None

        title = file_title.strip() or "document"

        # 1) 重命名 images/ 下的图片，并记录 old_name -> new_name
        images_dir = base_dir / "images"
        img_rename_map: Dict[str, str] = {}
        if images_dir.exists() and images_dir.is_dir():
            img_files = sorted(p for p in images_dir.iterdir() if p.is_file())
            for idx, old_path in enumerate(img_files, start=1):
                new_name = f"{title}_img_{idx:03d}{old_path.suffix.lower()}"
                new_path = old_path.with_name(new_name)
                old_path.rename(new_path)
                img_rename_map[old_path.name] = new_name

        # 2) 替换 md 文件里的图片引用
        md_path: Optional[Path] = None
        for md_file in base_dir.iterdir():
            if md_file.is_file() and md_file.suffix.lower() == ".md":
                content = md_file.read_text(encoding="utf-8")
                for old_name, new_name in img_rename_map.items():
                    content = content.replace(
                        f"images/{old_name}", f"images/{new_name}"
                    )
                md_file.write_text(content, encoding="utf-8")
                if md_file.name == "full.md":
                    md_path = md_file

        # 3) 重命名顶层文件
        for p in list(base_dir.iterdir()):
            if not p.is_file():
                continue
            old_name = p.name

            if old_name == "full.md":
                new_name = f"{title}.md"
            elif old_name == "full_new.md":
                new_name = f"{title}_new.md"
            elif old_name == "layout.json":
                new_name = f"{title}_layout.json"
            else:
                # 去掉 36 位 UUID 前缀（含下划线）
                rest = re.sub(
                    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}_",
                    "",
                    old_name,
                )
                # 去掉可能的 full_ 前缀
                if rest.startswith("full_"):
                    rest = rest[5:]
                if rest:
                    new_name = f"{title}_{rest}"
                else:
                    new_name = f"{title}_{old_name}"

            new_path = p.with_name(new_name)
            p.rename(new_path)
            if old_name == "full.md":
                md_path = new_path

        return str(md_path) if md_path else None

    # ------------------------------------------------------------------
    # HTTP 封装
    # ------------------------------------------------------------------
    def _post(self, url: str, headers: dict, payload: dict) -> dict:
        try:
            resp = requests.post(url, headers=headers, json=payload,
                                 timeout=HTTP_TIMEOUT_SECONDS)
        except requests.RequestException as e:
            raise PdfConversionError(
                f"请求 MinerU 失败: {e}", node_name=self.name, cause=e,
            )
        self._log_http("POST", url, payload, resp)
        return self._parse_response(resp)

    def _get(self, url: str, headers: dict) -> dict:
        try:
            resp = requests.get(url, headers=headers,
                                timeout=HTTP_TIMEOUT_SECONDS)
        except requests.RequestException as e:
            raise PdfConversionError(
                f"请求 MinerU 失败: {e}", node_name=self.name, cause=e,
            )
        self._log_http("GET", url, None, resp)
        return self._parse_response(resp)

    def _log_http(self, method: str, url: str, payload, resp) -> None:
        """调试用：打印每次 MinerU 接口调用的状态码与原始返回内容"""
        body = resp.text[:2000] if resp.text else "<empty>"
        req = ""
        if payload is not None:
            import json as _json
            req = f" payload={_json.dumps(payload, ensure_ascii=False)[:500]}"
        self.log_step(
            f"MinerU HTTP {method}",
            f"status={resp.status_code} url={url}{req} body={body}",
        )

    @staticmethod
    def _parse_response(resp: requests.Response) -> dict:
        try:
            return resp.json()
        except ValueError as e:
            raise PdfConversionError(
                f"MinerU 返回非 JSON: status={resp.status_code}, "
                f"body={resp.text[:200]}",
                node_name="pdf_to_md_node",
                cause=e,
            )

    def _raise_api_error(self, data: dict, action: str) -> None:
        code = str(data.get("code", ""))
        msg = data.get("msg", "")
        if code in TOKEN_ERROR_CODES:
            raise ConfigurationError(
                f"{action} 失败（Token 错误: {code} {msg}），"
                f"请检查 .env 中的 MINERU_API_TOKEN",
                node_name=self.name,
            )
        raise PdfConversionError(
            f"{action} 失败: code={code}, msg={msg}",
            node_name=self.name,
        )
