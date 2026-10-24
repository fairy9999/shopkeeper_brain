from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState


class RrfNode(BaseNode):
    name = "rrf"
    def process(self, state:QueryGraphState) -> QueryGraphState:
        return state