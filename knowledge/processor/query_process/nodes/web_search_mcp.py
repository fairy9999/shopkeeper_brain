"""MCP 网络搜索节点

通过 MCP 协议（Streamable HTTP 模式）调用网络搜索服务获取外部信息。
使用 async with 上下文管理器自动管理 MCP 连接生命周期。
"""
import json
from json import JSONDecodeError
from typing import Dict, Any

import asyncio
from agents.mcp import MCPServerStreamableHttp

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState


class WebSearchMcpNode(BaseNode):
    name = "web_search_mcp"
    """MCP 网络搜索节点。

    负责从网络查询当前的问题【整个知识库没有找到该问题，兜底的网络结果】
    通过 MCP 协议调用灵积平台的通用搜索工具（bailian_web_search）。
    """
    def process(self, state: QueryGraphState) -> Dict:
        """执行网络搜索。

                Args:
                    state: 查询图状态，需包含 rewritten_query 和 item_names。

                Returns:
                    更新后的状态，包含 web_search_docs 搜索结果。
                """
        # 1. 参数校验
        validated_query, validate_item_names = self._validate_state(state)

        # 2. 执行 web_mcp_search（asyncio.run 桥接同步→异步）
        mcp_result = asyncio.run(self._web_mcp(validated_query))
        if not mcp_result:
            return {}

        # 3. 更新 state web_search_docs
        state['web_search_docs'] = mcp_result

        return state

    def _validate_state(self, state):
        validated_query = state.get("rewritten_query")
        validate_item_names = state.get("item_names")
        if not validated_query or not isinstance(validated_query, str):
            raise StateFieldError("rewritten_query 字段不存在或不是字符串")
        if not validate_item_names or not isinstance(validate_item_names, list):
            raise StateFieldError("item_names 字段不存在或不是列表")
        return validated_query, validate_item_names

    async def _web_mcp(self, validated_query):
        """异步执行 MCP 网络搜索。

                使用 async with 上下文管理器自动管理连接生命周期：
                - 进入 with 块时自动建立连接（connect）
                - 退出 with 块时自动关闭连接（cleanup），即使抛异常也会执行

                Args:
                    validated_query: 校验后的查询文本。

                Returns:
                    搜索结果列表，每项包含 title、url、snippet。
                """
        # 1. 定义 MCP 客户端（Streamable HTTP 模式，自动管理连接）
        async with MCPServerStreamableHttp(
                name="web_search",
                params={
                    "url": self.config.mcp_dashscope_base_url, #MCP服务端点
                    "headers": {"Authorization": "Bearer " + self.config.mcp_dashscope_api_key},
                    "timeout": 300, #MCP超时时间（秒）
                    "terminate_on_close": True, #MCP关闭时是否终止
                },
                max_retry_attempts=2, #MCP最大重试次数
                cache_tools_list=True,
        ) as mcp_client:
            # 2. 调用工具
            execute_tool_result = await mcp_client.call_tool(
                tool_name="bailian_web_search",
                arguments={"query": validated_query, "count": 3}
            )
            print(f"MCP Search Result: {execute_tool_result}")

            # 3. 解析工具执行完的结果
            # 3.1 获取最外层的对象
            if not execute_tool_result:
                return []
            # 3.2 获取对象的 content 属性
            if not execute_tool_result.content[0]:
                return []
            # 3.3 获取 TextContent 对象的 text
            text_content_text: str = execute_tool_result.content[0].text
            if not text_content_text:
                return []
            # 3.4 反序列化
            try:
                text_content_text: Dict[str, Any] = json.loads(text_content_text)

                # a) 获取 pages
                pages = text_content_text.get('pages', "")
                if not pages:
                    return []
                search_result = []
                # b) 遍历得到每一个结果
                for page in pages:
                    snippet = page.get('snippet', "").strip()
                    title = page.get('title', "").strip()
                    url = page.get('url', "").strip()
                    search_result.append({"snippet": snippet, "title": title, "url": url})
                # c) 最终返回
                return search_result
            except JSONDecodeError as e:
                self.logger.error(
                    f"反序列MCP结果失败信息 {str(e.msg)} 原文{e.doc} 位置{e.pos}"
                )
                return []

if __name__ == '__main__':
    import json

    state = {
        "rewritten_query": "今天的小米汽车的股价是多少",
    }

    web_mcp_search = WebSearchMcpNode()
    result = web_mcp_search.process(state)

    for r in result.get('web_search_docs', []):
        print(json.dumps(r, ensure_ascii=False, indent=2))

