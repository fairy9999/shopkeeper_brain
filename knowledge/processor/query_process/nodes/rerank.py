from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState


class RerankNode(BaseNode):
    name = "rerank"
    def process(self, state:QueryGraphState) -> QueryGraphState:
        return state