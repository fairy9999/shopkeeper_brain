
"""答案输出节点

组装提示词、调用 LLM 生成答案，支持流式/非流式输出，写入历史记录。
"""
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState


class AnswerOutputNode(BaseNode):
    name = "answer_output"
    """答案输出节点

    流程: 检查已有答案 → 构建提示词 → LLM 生成 → 写入历史 → 发送结束事件
    """
    def process(self, state:QueryGraphState) -> QueryGraphState:
        """执行答案生成
        Args:
            state: 需包含 task_id、session_id、reranked_docs 等
        Returns:
            更新后的 state，包含 answer 和 prompt
        """
        return state