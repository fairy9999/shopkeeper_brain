
"""答案输出节点

组装提示词、调用 LLM 生成答案，支持流式/非流式输出，写入历史记录。
"""
import re
from typing import List, Dict, Tuple

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompt.query_prompt import ANSWER_PROMPT
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.mongo_history_util import save_chat_message
from knowledge.utils.sse_util import push_sse_event, SSEEvent
from knowledge.utils.task_util import set_task_result


class AnswerOutputNode(BaseNode):
    name = "answer_output"
    """答案输出节点

    流程: 检查是否已有答案 → 有答案（选择/抱歉） → 写入历史 → 发送结束事件（流式/非流式）
         检查是否已有答案 →无答案（三路检索） → 构建提示词 → LLM 生成 → 写入历史 → 发送结束事件（流式/非流式）
    """
    def process(self, state:QueryGraphState) -> QueryGraphState:
        """执行答案生成
        Args:
            state: 需包含 task_id、session_id、reranked_docs、
            item_names、rewritten_query、is_stream、history 等
        Returns:
            更新后的 state，包含 answer 和 prompt
        """
        task_id = state.get("task_id")
        is_stream = state.get("is_stream")
        # 1. 检查是否已有答案
        answer = state.get("answer")
        if answer:
            # 商品名称确认节点，生成了答案:中置信选择或抱歉
            # state["answer"]=f"我不确定您指的是哪款产品。您是在询问以下产品吗:{'、'.join(options
            # state["answer"]="抱歉，我无法识别您询问的具体产品名称，请提供更准确的产品名称或型号。
            self._push_existing_answer(state)
        # 2. 构建提示词 → 调用 LLM 生成答案
        else:
            prompt = self._build_prompt(state)
            state["prompt"] = prompt
            self._generate_answer(state, prompt)
        # 3. 写入历史记录
        self._write_history(state)

        # 4. 流式模式发送结束事件
        if is_stream:
            push_sse_event(task_id=state["task_id"], event=SSEEvent.FINAL, data={"answer": state["answer"]})
        return state
    # ================================================================== #
    #                    已有答案推送                                       #
    # ================================================================== #
    def _push_existing_answer(self, state):
        if not state.get("is_stream"):
            set_task_result(task_id=state["task_id"], key="answer", value=state["answer"])
        # else: #这里省略代码。由记录历史之后，统一发送结束事件
        #     push_sse_event(task_id=state["task_id"], event=SSEEvent.FINAL, data={"answer": state["answer"]})

    # ================================================================== #
    #                    提示词构建                                         #
    # ================================================================== #
    def _build_prompt(self, state):
        char_budget = self.config.max_context_chars # 最大字符预算12000

        # 1. 获取问题和商品名
        question = state.get("rewritten_query") or state.get("original_query", "")
        item_names = state["item_names"]

        # 2. 格式化上下文文档(拼串后上下文，可继续拼接剩余字符长度。)
        context_str, char_budget = self._format_reranked_docs(
            state.get("reranked_docs") or [], char_budget
        )

        # 3. 格式化历史对话(拼串后历史对话，可继续拼接剩余字符长度。)
        history_str, char_budget = self._format_chat_history(
            state.get("history") or [], char_budget
        )

        # 5. 组装提示词
        return ANSWER_PROMPT.format(
            context=context_str or "无参考内容",
            history=history_str if history_str else "暂无历史对话",
            item_names=", ".join(item_names),
            question=question,
        )

    def _format_reranked_docs(self, reranked_docs: List[Dict], char_budget: int) -> Tuple[str, int]:
        """格式化重排序文档，带字符预算控制
        拼串格式：
            [1][source="local"][chunk_id=str(467775438085177876)][title="万用表RS-12的使用"][score=0.9999] \n
             xxxxxxxxxxxxxхх内容xxxxxxxxxxxxxxxxхxх \n\n
            [2][source="web"][url="http:/xxxx"][title="万用表RS-12的使用"][score=0.9999] \n
            xxxxxxxxxxxxxхх内容xxxxxxxxxxxxxxxxхxхх \n\n
        """

        formatted_lines = []
        used_chars = 0

        for idx, doc in enumerate(reranked_docs, 1):
            content = doc.get("content", "").strip()
            if not content:
                continue

            meta_tags = [f"[{idx}]"]
            for field, template in [
                ("source", "[source={}]"),
                ("chunk_id", "[chunk_id={}]"),
                ("url", "[url={}]"),
                ("title", "[title={}]"),
            ]:
                field_value = str(doc.get(field, "")).strip()
                if field_value:
                    meta_tags.append(template.format(field_value))

            relevance_score = doc.get("score")
            if relevance_score is not None:
                meta_tags.append(f"[score={float(relevance_score):.4f}]")

            doc_entry = " ".join(meta_tags) + "\n" + content

            if used_chars + len(doc_entry) > char_budget:
                break

            formatted_lines.append(doc_entry)
            used_chars += len(doc_entry) + 2

        return "\n\n".join(formatted_lines), char_budget - used_chars

    def _format_chat_history(self, chat_history: List[Dict], char_budget: int) -> Tuple[str, int]:
        """格式化历史对话"""
        formatted_lines = []
        used_chars = 0

        role_label_map = {"user": "用户", "assistant": "助手"}

        for message in chat_history:
            role = message.get("role", "")
            text = message.get("text", "")
            if not text or role not in role_label_map:
                continue

            formatted_line = f"{role_label_map[role]}: {text}"

            if used_chars + len(formatted_line) > char_budget:
                break

            formatted_lines.append(formatted_line)
            used_chars += len(formatted_line) + 1

        return "\n".join(formatted_lines), char_budget - used_chars

    # ================================================================== #
    #                    LLM 生成                                          #
    # ================================================================== #
    def _generate_answer(self, state, prompt):
        """调用 LLM 生成答案（流式/非流式）"""
        self.log_step("generate", "生成答案")
        llm_client = AIClients.get_llm_openai(False)
        if llm_client is None:
            raise ValueError("LLM 客户端初始化失败")
        if state.get("is_stream"):
            state["answer"] = self.call_llm_stream(llm_client, state, prompt)
        else:
            state["answer"] = self.call_llm_invoke(llm_client, state, prompt)
            set_task_result(task_id=state["task_id"], key="answer", value=state["answer"])
    def call_llm_stream(self, llm_client, state, prompt) -> str:
        """流式生成，逐 chunk 推送 delta 事件"""
        accumulated_answer = ""  # 积累答案
        try:
            for chunk in llm_client.stream(prompt):
                delta_text = getattr(chunk, "content", "") or ""
                if delta_text:
                    accumulated_answer += delta_text
                    push_sse_event(task_id=state["task_id"], event=SSEEvent.DELTA, data={"delta": delta_text})
        except Exception as e:
            self.logger.error(f"流式生成出错: {e}")
        return accumulated_answer

    def call_llm_invoke(self, llm_client, state, prompt) -> str:
        """非流式生成"""
        self.log_step("generate", "生成答案")

        try:
            response = llm_client.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            self.logger.error(f"生成回答出错: {e}")
            return "抱歉，生成回答时出现错误。"

    # ================================================================== #
    #                    历史记录                                           #
    # ================================================================== #
    def _write_history(self, state: QueryGraphState):

         """
        使用历史记录:
            商品名确认节点。拉取最近10条
            生成提示词模板，参数需要历史会话
        生成答案时需要记录历史：
             将用户问题和助手回答写入 MongoDB 历史记录
         """
         session_id = state["session_id"] # 会话ID,后端生成的
         rewritten_query = state.get("rewritten_query", "") or state.get("original_query", "")
         item_names = state.get("item_names") or []
         try:
             # 保存用户问题
             save_chat_message(
                 session_id = session_id,
                 role = "user",
                 text = rewritten_query,
                 rewritten_query = rewritten_query,
                 item_names = item_names,
             )
             # 保存助手回答
             save_chat_message(
                 session_id = session_id,
                 role = "assistant",
                 text = state["answer"],
                 rewritten_query = rewritten_query,
                 item_names = item_names,
             )
         except Exception as e:
             self.logger.error(f"保存历史记录出错: {e}")

if __name__ == "__main__":
    from dotenv import load_dotenv
    import json

    load_dotenv()

    from knowledge.processor.query_process.base import setup_logging
    setup_logging()

    print("=" * 60)
    print("开始测试: 答案生成节点 (AnswerOutputNode)")
    print("=" * 60)

    # 构造模拟状态
    mock_state = {
        "task_id": "test_task_001",
        "session_id": "test_session_001",
        "is_stream": False,
        "original_query": "万用表怎么测电压？",
        "rewritten_query": "RS-12数字万用表如何测量电压？",
        "item_names": ["RS-12数字万用表"],
        "reranked_docs": [
            {
                "content": "数字万用表测量电压步骤：1. 将旋钮转到V档位；2. 黑表笔插COM孔，红表笔插V孔；3. 将表笔并联到被测点两端。",
                "source": "local",
                "chunk_id": "chunk_001",
                "title": "万用表使用手册",
                "score": 0.9234
            },
            {
                "content": "测量直流电压时需注意正负极性，红表笔接正极，黑表笔接负极。",
                "source": "web",
                "url": "https://example.com/guide",
                "title": "电压测量指南",
                "score": 0.8756
            }
        ],
        "history": [
            {"role": "user", "text": "万用表是什么？"},
            {"role": "assistant", "text": "万用表是一种多功能电子测量仪器..."}
        ],
    }

    print("【输入状态】:")
    print(f"  query: {mock_state['rewritten_query']}")
    print(f"  item_names: {mock_state['item_names']}")
    print(f"  reranked_docs: {len(mock_state['reranked_docs'])} 篇")
    print("-" * 60)

    # 执行答案生成
    node = AnswerOutputNode()
    result = node.process(mock_state)

    # 打印结果
    print("\n【生成结果】:")
    print("-" * 60)
    print(result.get("answer", "无答案"))
    print("-" * 60)

    print("\n测试完成")


