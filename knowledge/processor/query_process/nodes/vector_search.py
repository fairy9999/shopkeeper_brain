from agents.tracing import create

import json
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState
from typing import Dict, Any, Union

from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, item_names_filter, execute_hybrid_search_query


class VectorSearchNode(BaseNode):
    name = "search_embedding"
    # QueryGraphState | Dict[str, Any]与 QueryGraphState or Dict[str, Any] 与 Union[QueryGraphState, Dict[str, Any]] 是等价的
    def process(self, state:QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:
        """三路检索之一： 向量检索节点

        对用户查询进行向量化，在 Milvus 中执行混合搜索（稠密 + 稀疏），返回相关切片。
        """
        # 1. 参数校验
        validated_query, validate_item_names = self._validate_state(state)
        # 2. 获取嵌入模型
        try:
            embedding_model = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"获取嵌入模型失败: {str(e)}")
            return {}

        # 3. 获取milvus客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"获取milvus客户端失败: {str(e)}")
            return {}
        try:
            # 4. 对问题嵌入
            embed_query = generate_bge_m3_hybrid_vectors(model=embedding_model, embedding_documents=[validated_query])
            # 5. 构建过滤表达式以及表达式参数
            filter_expr, filter_expr_param = item_names_filter(validate_item_names)
            # 6. 创建混合请求
            search_requests = create_hybrid_search_requests(
                dense_vector=embed_query["dense"][0],
                sparse_vector=embed_query["sparse"][0],
                expr=filter_expr, #调用工具类，生成过滤表达式
                expr_params=filter_expr_param #调用工具类，生成过滤表达式参数
                # expr="item_name in validate_item_names", #直接写方式
                # expr_params={"validate_item_names": validate_item_names},#直接写方式

            )
            # 7. 执行混合搜索请求
            hybrid_search_reps = execute_hybrid_search_query(
                milvus_client=milvus_client,
                search_requests=search_requests,
                collection_name=self.config.chunks_collection,
                ranker_weights=(0.5, 0.5),
                norm_score=True,
                output_fields=["chunk_id",  "item_name","content","title"]
            )

            # 8. 数据回填
            return {
                "embedding_chunks": hybrid_search_reps[0]
            }
        except Exception as e:
            self.logger.error(f"向量检索失败: {str(e)}")
            return {}

    def _validate_state(self, state):
        rewritten_query = state.get("rewritten_query")
        item_names = state.get("item_names")
        if not rewritten_query or not isinstance(rewritten_query, str):
            self.logger.error(f"rewritten_query is not a string: {rewritten_query}")
            raise StateFieldError(node_name=self.name, field_name='rewritten_query', expected_type=str)
        if not item_names or not isinstance(item_names, list):
            self.logger.error(f"item_names is not a list: {item_names}")
            raise StateFieldError(node_name=self.name, field_name='item_names', expected_type=list)
        return rewritten_query, item_names

if __name__ == '__main__':
    state = {
        "rewritten_query": "万用表如何测量电阻",
        "item_names": ["RS PRO RS-12 数字万用表"]
    }

    vector_search = VectorSearchNode()
    result = vector_search.process(state)

    for r in result.get('embedding_chunks'):
        print(json.dumps(r, ensure_ascii=False, indent=2))