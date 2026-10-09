from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.state import ImportGraphState


class EntryNode(BaseNode):
    def process(self, state: ImportGraphState) -> ImportGraphState:
        pass