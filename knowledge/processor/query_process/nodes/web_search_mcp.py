from typing import Dict

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState


class WebSearchMcpNode(BaseNode):
    name = "web_search_mcp"
    def process(self, state: QueryGraphState) -> Dict:
        return {}