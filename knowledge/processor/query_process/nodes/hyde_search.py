"""HyDE 检索节点

    使用 Hypothetical Document Embedding 技术：
    先让 LLM 生成假设性文档，再将其与原查询拼接后向量化检索，提升召回质量。
"""
from typing import Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage
from mineru.backend.postprocess import llm_client

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompt.query_prompt import HYDE_USER_PROMPT_TEMPLATE, HYDE_SYSTEM_PROMPT_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import item_names_filter, create_hybrid_search_requests, execute_hybrid_search_query


class HyDeSearchNode(BaseNode):
    name = "search_embedding_hyde"
    """HyDE 检索节点

       流程: 参数校验 → LLM 生成假设文档 → 拼接原查询 → 向量化 → 混合检索
       """
    def process(self, state: QueryGraphState) -> QueryGraphState | Dict[str, Any] :
        """执行 HyDE 检索
           Args:
               state: 需包含 rewritten_query 和 item_names
           Returns:
               {"hyde_embedding_chunks": [...]} 搜索结果列表
        """
        # 1. 参数校验
        validated_query, validate_item_names = self._validate_state(state)
        # 2. 生成假设性文档
        hyde_doc = self._generate_hyde_doc(validated_query,validate_item_names)
        # 3. 假设性文档嵌入(注入问题+假设性文档)
        embedding_document = validated_query + "\n" + hyde_doc
        self.logger.info(f"嵌入文档: {embedding_document}")
        # 4. 获取嵌入模型 & milvus 客户端
        try:
            embedding_model = AIClients.get_bge_m3_client()
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"获取 嵌入模型 & milvus 客户端失败: {str(e)}")
            return {}
        try:
            # 5. 对问题嵌入
            embed_query = generate_bge_m3_hybrid_vectors(model=embedding_model, embedding_documents=[embedding_document])
            # 6. 获取 item_name 的过滤表达式
            filter_expr, filter_expr_param = item_names_filter(validate_item_names)
            # 7. 创建混合搜索请求
            search_requests = create_hybrid_search_requests(
                dense_vector=embed_query["dense"][0],
                sparse_vector=embed_query["sparse"][0],
                expr=filter_expr,
                expr_params=filter_expr_param
            )
            # 8. 执行混合搜索请求
            hybrid_search_reps = execute_hybrid_search_query(
                milvus_client=milvus_client,
                search_requests=search_requests,
                collection_name=self.config.chunks_collection,
                ranker_weights=(0.5, 0.5),
                norm_score=True,
                output_fields=["chunk_id",  "item_name","content","title"]
            )
            # 9. 只更新 hyde_embedding_chunks
            return {
                "hyde_embedding_chunks": hybrid_search_reps[0]
            }
        except Exception as e:
            self.logger.error(f"HyDE 检索失败: {str(e)}")
            return {}

    def _validate_state(self, state):
        validated_query = state.get("rewritten_query")
        validate_item_names = state.get("item_names")
        if not validated_query or not isinstance(validated_query, str):
            raise StateFieldError("rewritten_query 字段不存在或不是字符串")
        if not validate_item_names or not isinstance(validate_item_names, list):
            raise StateFieldError("item_names 字段不存在或不是列表")
        return validated_query, validate_item_names

    def _generate_hyde_doc(self, validated_query, validate_item_names):
        try:
            llm_client = AIClients.get_llm_openai(response_format=False)
        except ConnectionError as e:
            self.logger.error(f"llm客户端调用失败: {str(e)}")
            return ""
        system_prompt=HYDE_SYSTEM_PROMPT_TEMPLATE.format(item_names=validate_item_names)
        user_prompt = HYDE_USER_PROMPT_TEMPLATE.format(rewritten_query=validated_query,item_names=validate_item_names)
        llm_response = llm_client.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        llm_content = llm_response.content.strip()
        if not llm_content:
            return ""

        return llm_content

# ================================================================== #
#                        测试入口                                   #
# ================================================================== #

if __name__ == "__main__":
    from knowledge.processor.query_process.base import setup_logging

    setup_logging()

    print("=" * 60)
    print("开始测试: HyDE 检索节点 (HydeSearchNode)")
    print("=" * 60)

    mock_state = {
        "rewritten_query": "RS-12 数字万用表如何测量直流电压？",
        "item_names": ["RS PRO RS-12 数字万用表"],
    }

    print("【输入状态】:")
    print(f"  查询: {mock_state['rewritten_query']}")
    print(f"  商品: {mock_state['item_names']}")
    print("-" * 60)

    node = HyDeSearchNode()
    result = node.process(mock_state)

    chunks = result.get("hyde_embedding_chunks", [])
    print(f"\n【HyDE 检索结果】: {len(chunks)} 条")
    for i, chunk in enumerate(chunks, 1):
        entity = chunk.get("entity", {})
        print(f"  [{i}] chunk_id={entity.get('chunk_id')} "
              f"item_name={entity.get('item_name')} "
              f"distance={chunk.get('distance', 'N/A')}")
        content = entity.get("content", "")
        print(f"      内容: {content[:80]}...")

    print("-" * 60)
    print("测试完成")