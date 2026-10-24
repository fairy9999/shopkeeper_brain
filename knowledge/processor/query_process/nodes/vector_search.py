from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from typing import Dict


class VectorSearchNode(BaseNode):
    name = "search_embedding"
    def process(self, state:QueryGraphState) -> Dict:
        return {}