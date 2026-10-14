#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MinerU 轻量解析 API（Agent）测试脚本

【这个脚本有什么用】
    用 MinerU 的「Agent 轻量解析 API」独立测试 PDF 转 Markdown。
    它和我们生产代码里的「精准解析 API」是**完全不同的两套后端服务**，
    所以可以用它来判断：
        - MinerU 是整体宕机了，还是只有「精准解析」那套队列卡住

【两套 API 的区别】
    ┌──────────────────┬─────────────────────────┬──────────────────────────┐
    │                  │ 精准解析 API（生产用的） │ Agent 轻量 API（本脚本）  │
    ├──────────────────┼─────────────────────────┼──────────────────────────┤
    │ 需要 Token        │ 需要                     │ 不需要（按 IP 限频）      │
    │ 单文件大小        │ ≤ 200 MB                 │ ≤ 10 MB                  │
    │ 单文件页数        │ ≤ 200 页                 │ ≤ 20 页                  │
    │ 批量              │ 支持                     │ 不支持，一次一个文件      │
    │ 输出              │ ZIP（含 md/json/图片）    │ 只有一个 Markdown 的 URL  │
    └──────────────────┴─────────────────────────┴──────────────────────────┘

【调用流程】（三步，异步提交 + 轮询）
    1. POST /api/v1/agent/parse/file   → 拿到 task_id 和 file_url
    2. PUT  {file_url}                  → 把 PDF 上传到 OSS
    3. GET  /api/v1/agent/parse/{task_id} → 轮询直到 state=done，拿到 markdown_url
    4. 下载 markdown_url 到本地

【状态流转】
    waiting-file → uploading → pending → running → done / failed

【限制】
    - 文件 ≤ 10MB、≤ 20 页（超出返回 err_code=-30001 / -30003）
    - 每 IP 每分钟提交次数有限，超出返回 HTTP 429

运行方式（在项目根目录）：
    python knowledge/test/import_process/02-mineru/test_mineru.py
"""
import json
import os
import time
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ==================== 配置（按需修改） ====================
API_BASE = os.getenv("MINERU_API_BASE", "https://mineru.net")

# 待测试的 PDF 路径
PDF_PATH = r"E:\zw\project\file\doc\万用表RS-12的使用.pdf"

# 结果输出目录（留空则用 PDF 同级的 {文件名}_agent 目录）
OUTPUT_DIR = ""

# 解析语言（ch=中英，详见官方文档 language 参数）
LANGUAGE = "ch"
ENABLE_TABLE = True       # 表格识别
ENABLE_FORMULA = True     # 公式识别
IS_OCR = False            # 是否强制 OCR
# 页码范围（Agent API 只支持 "1-10" 或 "5" 这种，不支持逗号）。留空表示全部页
PAGE_RANGE = ""

# 轮询参数
POLL_INTERVAL_SECONDS = 3
POLL_TIMEOUT_SECONDS = 300
HTTP_TIMEOUT_SECONDS = 60


def _print_resp(tag: str, resp: requests.Response) -> None:
    """统一打印 HTTP 状态码 + 原始返回内容，方便排查"""
    body = resp.text[:2000] if resp.text else "<empty>"
    print(f"  [{tag}] status={resp.status_code} body={body}")


# ==================== 步骤 1：申请上传链接 ====================
def step1_get_upload_url(file_name: str) -> tuple[str, str]:
    """POST /api/v1/agent/parse/file，返回 (task_id, file_url)"""
    url = f"{API_BASE}/api/v1/agent/parse/file"
    payload = {
        "file_name": file_name,
        "language": LANGUAGE,
        "enable_table": ENABLE_TABLE,
        "enable_formula": ENABLE_FORMULA,
        "is_ocr": IS_OCR,
    }
    if PAGE_RANGE:
        payload["page_range"] = PAGE_RANGE

    print(f"[步骤1] 申请上传链接 POST {url}")
    print(f"        payload={json.dumps(payload, ensure_ascii=False)}")

    resp = requests.post(url, json=payload, timeout=HTTP_TIMEOUT_SECONDS)
    _print_resp("步骤1", resp)
    resp.raise_for_status()

    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"申请上传链接失败: code={data.get('code')}, msg={data.get('msg')}")

    d = data["data"]
    task_id = d["task_id"]
    file_url = d["file_url"]
    print(f"        -> task_id={task_id}")
    return task_id, file_url


# ==================== 步骤 2：PUT 上传 PDF ====================
def step2_upload(file_url: str, pdf_path: str) -> None:
    """PUT 上传 PDF 到返回的 OSS 签名 URL（不要设置 Content-Type）"""
    file_size = os.path.getsize(pdf_path)
    print(f"[步骤2] 上传 PDF PUT {file_url[:80]}... (size={file_size}B)")

    with open(pdf_path, "rb") as f:
        resp = requests.put(file_url, data=f, timeout=HTTP_TIMEOUT_SECONDS * 2)

    _print_resp("步骤2", resp)
    resp.raise_for_status()
    print("        -> 上传成功")


# ==================== 步骤 3：轮询任务状态 ====================
def step3_poll(task_id: str) -> str:
    """GET /api/v1/agent/parse/{task_id}，返回 markdown_url"""
    url = f"{API_BASE}/api/v1/agent/parse/{task_id}"
    print(f"[步骤3] 轮询结果 GET {url}")

    start = time.time()
    while time.time() - start < POLL_TIMEOUT_SECONDS:
        resp = requests.get(url, timeout=HTTP_TIMEOUT_SECONDS)
        _print_resp("步骤3", resp)
        resp.raise_for_status()

        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"查询失败: code={data.get('code')}, msg={data.get('msg')}")

        d = data.get("data") or {}
        state = d.get("state", "unknown")
        elapsed = int(time.time() - start)

        if state == "done":
            markdown_url = d.get("markdown_url", "")
            if not markdown_url:
                raise RuntimeError("任务完成但没有返回 markdown_url")
            print(f"        -> 解析完成，耗时 {elapsed}s")
            return markdown_url

        if state == "failed":
            raise RuntimeError(
                f"解析失败: err_code={d.get('err_code')}, err_msg={d.get('err_msg')}"
            )

        print(f"        [状态] [{elapsed}s] {state}")
        time.sleep(POLL_INTERVAL_SECONDS)

    raise TimeoutError(f"轮询超时（>{POLL_TIMEOUT_SECONDS}s）")


# ==================== 步骤 4：下载 Markdown ====================
def step4_download(markdown_url: str, out_path: Path) -> None:
    """下载解析好的 Markdown 到本地"""
    print(f"[步骤4] 下载 Markdown -> {out_path}")
    resp = requests.get(markdown_url, timeout=HTTP_TIMEOUT_SECONDS)
    _print_resp("步骤4", resp)
    resp.raise_for_status()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(resp.text, encoding="utf-8")
    print(f"        -> 已保存（{len(resp.text)} 字符）")


def main() -> bool:
    print("=" * 60)
    print("MinerU 轻量解析 API（Agent）测试")
    print("=" * 60)

    pdf_path = Path(PDF_PATH)
    if not pdf_path.exists():
        print(f"✗ PDF 不存在: {pdf_path}")
        return False

    # 文件大小预检（轻量 API 上限 10MB）
    size_mb = pdf_path.stat().st_size / 1024 / 1024
    print(f"PDF: {pdf_path.name}  ({size_mb:.2f} MB)")
    if size_mb > 10:
        print("✗ 超出轻量 API 的 10MB 限制，请换小文件或改用精准解析 API")
        return False

    out_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else pdf_path.parent / f"{pdf_path.stem}_agent"
    out_path = out_dir / "full.md"

    try:
        task_id, file_url = step1_get_upload_url(pdf_path.name)
        step2_upload(file_url, str(pdf_path))
        markdown_url = step3_poll(task_id)
        step4_download(markdown_url, out_path)
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ 测试失败: {e}")
        print("=" * 60)
        return False

    print("\n" + "=" * 60)
    print(f"✓ 测试成功！Markdown 已保存到:\n  {out_path}")
    print("=" * 60)
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
