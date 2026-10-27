"""RRF 融合排序节点

使用 Reciprocal Rank Fusion 算法融合多路检索结果。
"""
from typing import Any, Dict, Tuple, List

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState


class RrfNode(BaseNode):
    name = "rrf"
    """RRF 融合排序节点

        流程: 收集多路检索结果 → 带权重 RRF 融合 → 按得分降序返回
        """

    def __init__(self):
        super().__init__()
        self._top_k = self.config.rrf_max_results
        self._rrf_k = self.config.rrf_k
    def process(self, state:QueryGraphState) -> QueryGraphState:
        """执行 RRF 融合
        Args:
            state: 需包含 embedding_chunks 和 hyde_embedding_chunks
            embedding_chunks向量检索结果
            [{
                "chunk_id": 467775438085177895,
                "distance": 0.7380014657974243,
                "entity": {
                    "chunk_id": 467775438085177895,
                    "item_name": "RS-12 数字万用表",
                    "content": "#电阻测量\n\n警告:为防触电,测量前应断开电源，把所有电容放电，取出电池和拔掉电线。\n1、将功能转盘置于最高电阻Q位置。",
                    "title": "# 电阻测量"
                }
            }]
            hyde_embedding_chunks HyDE(假设性文档)检索结果
            [{
                "chunk_id": 467775438085177891,
                "distance": 0.8033741116523743,
                "entity": {
                    "chunk_id": 467775438085177891,
                    "item_name": "RS-12 数字万用表",
                    "content": "##直流电压测量\n\n注意:正打开或关闭电源时不要进行此项测量，瞬间的强大电压将损坏仪表。\n1，将功能转盘置于V DC的",
                    "title": "## 直流电压测量"
                }
            }]

        Returns:
            更新后的 state，包含 rrf_chunks RRF融合后的切片
        """
        # 1. 各路搜索的结果
        # 1.1 获取向量检索路的结果
        vector_search_chunks = state.get('embedding_chunks') or []
        # 1.2 获取 hyde 向量检索路的结果
        hyde_search_chunks = state.get('hyde_embedding_chunks') or []

        # 2.格式规整化 只要每路中的entity部分
        vector_chunks_entities:list[Dict[str,Any]] = self._normalize_input(vector_search_chunks)
        hyde_chunks_entities:list[Dict[str,Any]] = self._normalize_input(hyde_search_chunks)

        # 3. 设置各路的权重
        rrf_inputs = [(vector_chunks_entities,1.0),(hyde_chunks_entities,1.0)]

        # 4. 利用 RRF 的计算公式去获取到所有路查询到的所有 chunk 对应的 score
        rrf_merge_results: List[Tuple[Dict[str, Any], float]] = self._rrf_merge(
            rrf_inputs, self._rrf_k, self._top_k
        )

        # 5. 获取 rrf_chunks（只取文档，不要分数）
        rrf_chunks = [chunk for chunk, _ in rrf_merge_results]

        # 6. 记录分数范围（便于调试）
        if rrf_merge_results:
            scores = [s for _, s in rrf_merge_results]
            self.logger.info(f"分数范围: [{min(scores):.6f}, {max(scores):.6f}]")

        # 7. 更新 state
        state['rrf_chunks'] = rrf_chunks

        # 8. 返回 state
        return state

    def _normalize_input(self, input_chunks):
        """
        各路列表数据规整化。只保留entity部分
        注意：保持原有顺序，RRF 依赖该顺序作为 rank；上游需保证按相关度降序。
        :param input_chunks:
        :return
        """
        if not input_chunks:
            return []
        normalized_output = []
        for input_chunk in input_chunks:
            if not input_chunk:
                continue
            entity = input_chunk.get('entity')
            if not entity:
                continue
            normalized_output.append(entity)
        return normalized_output

    def _rrf_merge(self, rrf_inputs:List[Tuple[List[Dict[str, Any]], float]], _rrf_k:int = 60, _top_k:int = 10):
        """
                利用 RRF 公式计算每一个文档的总得分

                Args:
                    rrf_inputs: 各路的搜索结果 + 权重
                    _rrf_k: 平滑参数，通常取 60
                    _top_k: 合并完之后返回的文档数 10

                Returns:
                    合并以及排序后的文档列表
                """
        chunk_scores = {}  # 存放所有 chunk 的 RRF 计算后的分数值
        chunk_data = {}  # 存放所有 chunk 的文档数据
        for entityList , weight in rrf_inputs:
            for index,entity in enumerate(entityList,1):#index从1开始
                chunk_id = entity.get('chunk_id')
                if not chunk_id:
                    continue

                # RRF 公式: score += weight / (k + rank)
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + weight / (_rrf_k + index)
                chunk_data.setdefault(chunk_id,entity) #只在chunk_data中保存第一个出现的chunk，多路都有的chunk只保存一个

        # 所有路累加完毕后，按得分降序排序并截取前 top_k 条
        sorted_results = sorted(
            [(chunk_data[chunk_id], score) for chunk_id, score in chunk_scores.items()],
            key=lambda x: x[1],  # 根据score进行排序
            reverse=True
        )
        return sorted_results[:_top_k] if _top_k else sorted_results

# ================================================================== #
#                        测试入口                                   #
# ================================================================== #

if __name__ == "__main__":
    print("=" * 60)
    print("开始测试: RRF 融合节点")
    print("=" * 60)

    # 模拟两路检索结果
    # chunk_1 命中 2 路（预期最高分）
    # chunk_2 命中 2 路
    # chunk_3, chunk_4 各命中 1 路
    mock_state = {
        "embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "向量搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_2", "content": "向量搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_3", "content": "向量搜索结果#3"}},
        ],
        "hyde_embedding_chunks": [
            {"entity": {"chunk_id": "chunk_2", "content": "HyDE搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_1", "content": "HyDE搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_4", "content": "HyDE搜索结果#3"}},
        ],
    }

    print("【输入状态】:")
    print(f"  embedding_chunks: {len(mock_state['embedding_chunks'])} 条")
    print(f"  hyde_embedding_chunks: {len(mock_state['hyde_embedding_chunks'])} 条")
    print("-" * 60)

    rrf_node = RrfNode()
    result = rrf_node.process(mock_state)

    print("\n【融合结果】:")
    for i, chunk in enumerate(result["rrf_chunks"], 1):
        print(f"[{i}] {chunk.get('chunk_id')} - {chunk.get('content')}")

    print("-" * 60)
    print("测试完成")


