"""Rerank 重排序节点

使用 Reranker 模型对 RRF 融合结果和网络搜索结果进行重排序，
并通过优化版断崖检测实现动态 TopK 截断。
"""
from typing import List, Dict, Any

from knowledge.processor.query_process.base import BaseNode, setup_logging
from knowledge.processor.query_process.exceptions import RerankError
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.client.ai_clients import AIClients


class RerankNode(BaseNode):
    name = "rerank"
    """Rerank 重排序节点

    流程: 合并多源文档 → Reranker 计算相关性 → 断崖检测动态截断
    """
    def process(self, state:QueryGraphState) -> QueryGraphState:
        # 1. 获取 query
        user_query = state.get('rewritten_query', '') or state.get('original_query', '')

        # 2. 合并多源文档(本地 RRF 文档 + 网络搜索文档)
        merged_multi_docs: List[Dict[str, Any]] = self._merge_multi_source_docs(state)

        # 3. Rerank 精排
        reranked_docs: List[Dict[str, Any]] = self._rerank_merged_docs(user_query, merged_multi_docs)

        # 4. 动态 Top_K 截取(断崖检测 + 绝对分数底线过滤)
        cutoff_docs = self._cliff_catoff(reranked_docs)

        # 5. 存储结果
        state['reranked_docs'] = cutoff_docs
        return state

    def _merge_multi_source_docs(self, state):
        """合并本地 RRF 文档和 Web 搜索文档"""
        final_docs = []
        rrf_chunks = state.get("rrf_chunks")
        web_search_docs = state.get("web_search_docs")
        for rrf_doc in (rrf_chunks or []):
            if not isinstance(rrf_doc, dict):
                continue
            content = rrf_doc.get('content', '').strip()
            if not content:
                continue
            title = rrf_doc.get('title', '').strip()
            chunk_id = rrf_doc.get('chunk_id')
            format_rrf_doc = self._format_rrf_docs(
                content=content, title=title, chunk_id=chunk_id, source="local"
            )
            final_docs.append(format_rrf_doc)

        for web_doc in (web_search_docs or []):
            if not isinstance(web_doc, dict):
                continue
            content = web_doc.get('content', '') or web_doc.get('snippet', '').strip()
            if not content:
                continue
            title = web_doc.get('title', '').strip()
            url = web_doc.get('url', '').strip()
            format_web_doc = self._format_rrf_docs(
                content=content, title=title, url=url, source="web"
            )
            final_docs.append(format_web_doc)

        self.logger.info(f"收集到准备进行 Rerank 精排的文档 {len(final_docs)}")
        return final_docs

    def _format_rrf_docs(self, content: str, title: str="", chunk_id: str=None, url: str="", source: str=""):
        return {
            "content": content,
            "title": title,
            "chunk_id": chunk_id,
            "url": url,
            "source": source,
        }

    def _rerank_merged_docs(self, user_query, merged_multi_docs:List[Dict[str, Any]])->List[Dict[str, Any]]:
        """
        使用 Reranker 模型对文档进行精排
            1.构建reranker模型的的输入
            例子：pairs = [
                    ["什么是万用表？", "万用表是一种测量电压、电流、电阻的仪器"],
                    ["什么是万用表？", "今天天气很好"]
                ]
            2.reranker计算得分
            scores = reranker.compute_score(pairs)
        """
        #构建reranker模型的的输入
        pairs = [[user_query, doc['content']] for doc in merged_multi_docs]
        #reranker计算得分
        try:
            rerank_client = AIClients.get_bge_m3_rerank_client()
            # rerank_scores:List[float] = rerank_client.compute_score(pairs)#没有进行归一化：[5.0625, -4.7421875, 3.96484375, -9.4765625]
            rerank_scores: List[float] = rerank_client.compute_score(pairs, normalize=True) #normalize=True归一化 [0.9937100978631075, 0.008644177936723585, 0.9813821977381683, 7.662101864956481e-05]
            print("Rerank scores:", rerank_scores)

            # 将得分添加到doc中
            score_docs  = [{**doc, "score": score} for doc, score in zip(merged_multi_docs, rerank_scores)]
            # 按得分排序(降序)
            sorted_score_docs = sorted(score_docs, key=lambda x: x["score"], reverse=True)
            print(f"排序后结果返回:{sorted_score_docs}")
            return sorted_score_docs

        except Exception as e:
                self.logger.error(f"Rerank 计算得分出错: {e}")
                return [{**doc, "score": None} for doc in merged_multi_docs]

    def _cliff_catoff(self, reranked_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """断崖检测动态截断
        动态断崖检测截断。
        目的，全部文档返回，可能有部分文档分值较低，属于音;需要去除音文档。
        将相邻文档分数差值比较大的位置找到，叫做截断处。可以从断处进行截断。只保留断处之前的文档。后面文档属于噪音就不要了。
        第一断崖 VS 最大断崖 =>取决于业务场景。

        参数:
            reranked_docs: 按分数降序排列的文档列表
            rerank_min_top_k: 最少返回文档数
            rerank_max_top_k: 最多返回文档数

        返回值:
            截断后的文档列表
        """
        upper_bound = min(self.config.rerank_max_top_k, len(reranked_docs)) #10
        lower_bound = min(self.config.rerank_min_top_k, upper_bound) #3
        gap_abs = self.config.rerank_gap_abs #0.15 差值阈值

        if upper_bound <= lower_bound: #无需断崖检测，直接返回
            return reranked_docs[:upper_bound]

        cut_off = upper_bound #默认截断位置为最大值
        max_gap = 0.0 #记录最大断崖位置差值
        for i in range(0, upper_bound - 1):
            current_score = reranked_docs[i].get("score")
            next_score = reranked_docs[i + 1].get("score")

            if current_score is None or next_score is None:
                continue

            # 分数差值
            gap = current_score - next_score
            if gap >= gap_abs and gap > max_gap:
                max_gap = gap
                cut_off = i + 1
                break
        # 兜底：不管断崖在哪，至少保留lower_bound个
        cut_off = max(cut_off, lower_bound)
        return reranked_docs[:cut_off]



# ================================================================== #
#                        测试入口                                   #
# ================================================================== #

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    setup_logging()

    print("=" * 60)
    print("开始测试: 重排序节点 (RerankNode)")
    print("=" * 60)

    mock_state = {
        "rewritten_query": "怎么测这块主板的短路问题？",
        "rrf_chunks": [
            {"chunk_id": "local_1", "title": "主板维修手册",
             "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。"},
            {"chunk_id": "local_2", "title": "闲聊",
             "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"},
        ],
        "web_search_docs": [
            {"url": "https://example.com/repair", "title": "短路查修指南",
             "snippet": "主板通电前先打各主供电电感的对地阻值，阻值偏低就是短路。"},
            {"url": "https://example.com/news", "title": "科技新闻",
             "snippet": "苹果发布新款手机，A系列芯片性能提升20%。"},
        ],
    }

    print("【输入状态】:")
    print(f"  查询: {mock_state['rewritten_query']}")
    print(f"  本地文档: {len(mock_state['rrf_chunks'])} 篇")
    print(f"  网络文档: {len(mock_state['web_search_docs'])} 篇")
    print("-" * 60)

    node = RerankNode()
    result = node.process(mock_state)

    print("\n【重排序结果】:")
    for i, doc in enumerate(result["reranked_docs"], 1):
        score = doc.get('score')
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(f"[{i}] score={score_str} | {doc['source']:5} | {doc['content'][:50]}...")

    print("-" * 60)
    print("测试完成")
