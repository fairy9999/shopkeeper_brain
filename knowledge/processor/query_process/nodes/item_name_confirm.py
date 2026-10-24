"""商品名称确认节点

从用户查询中提取商品名称，通过向量相似度匹配与数据库中已有商品对齐确认。
"""
import re
import json

from json import JSONDecodeError
from typing import List, Dict, Any, Tuple

from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompt.query_prompt import ITEM_NAME_EXTRACT_TEMPLATE, ITEM_NAME_EXTRACT_SYSTEM_PROMPT
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query
from knowledge.utils.mongo_history_util import get_recent_messages

class ItemNameExtractor:
    """
    商品名提取器
    基于用户的原始问题和历史对话提取用户真正想问的商品名
    """
    def __init__(self, logger, node_name):
        self.logger = logger
        self.node_name = node_name

    def extract_item_name(self, history_text: str, original_query: str) -> List[str]:
        result = {"item_names": [], "rewritten_query": original_query}
        llm_client = AIClients.get_llm_openai(response_format=True)
        if llm_client is None:
            return result
        human_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(
            history_text=history_text if history_text else "暂无上下文",
            query=original_query
        )
        system_prompt = ITEM_NAME_EXTRACT_SYSTEM_PROMPT
        # invoke 的第二个位置参数是 config，消息必须作为列表整体传入第一个参数
        llm_response = llm_client.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        llm_content = llm_response.content.strip()
        if not llm_content:
            return result
        # 清洗并解析LLM的输出
        try:
            parsed_result = self._clean_parse(llm_content)
            result["rewritten_query"] = parsed_result.get("rewritten_query") or original_query
            result["item_names"] = parsed_result.get("item_names")
        except Exception as e:
            self.logger.error(f"清洗以及解析LLM的输出失败: {str(e)}")
        return result

    def _clean_parse(self, llm_response: str) -> Dict[str, Any]:
        """清洗并解析 LLM 响应"""
        cleaned = re.sub(r"^```(?:json)?\s*", "", llm_response.strip())
        content = re.sub(r"\s*```$", "", cleaned)

        try:
            parsed_llm_result: Dict[str, Any] = json.loads(content)
            raw_item_names = parsed_llm_result.get('item_names')
            if not isinstance(raw_item_names, list):
                clean_item_names = []
            else:
                clean_item_names = [raw_item for raw_item in raw_item_names if raw_item.strip()]

            raw_rewritten_query = parsed_llm_result.get('rewritten_query')
            clean_rewritten_query = "" if not isinstance(raw_rewritten_query, str) else raw_rewritten_query.strip()

            return {"item_names": clean_item_names, "rewritten_query": clean_rewritten_query}
        except JSONDecodeError as e:
            raise ValueError(f"JSON反序列LLM的输出失败：{str(e)}")

class ItemNameAligner:
    """
    商品名对齐器
    主要职责：
    1. 查询向量数据库
    2. 评分对齐
    3. 分数差异过滤
    """
    def __init__(self, logger, node_name,config):
        self.logger = logger
        self.node_name = node_name
        self.config = config

    def match_align_filter(self, item_names: List[str]) -> Tuple[List[str], List[str]]:
        """执行匹配、对齐、过滤三步流程

        extracted_name就是LLM根据用户问题，识别的商品名称:
        search_results: List[Dict[str, Any]]
           [
              {
                "extracted_name": extracted_name,
                "matches":[
                    {"item_name":"商品A1","score":0.9},
                    {"item_name":"商品A2","score":0.6}
                ]
               },
               {
                "extracted_name": extracted_name,
                "matches":[
                    {"item_name":"商品B1","score":0.9},
                    {"item_name":"商品B2","score":0.6}
                ]
               }
           ]
        """
        # 1. 查询向量数据库
        search_result: List[Dict[str, Any]] = self._match_vector(item_names)

        # 2. 评分对齐
        # confirmed 高置信商品 >=0.7
        # options   中等置信商品，需要用户选择 >=0.6 <0.7
        confirmed, options = self._item_name_score_align(search_result)

        # 3. 分数差异过滤（仅当 confirmed 有多个时）,最高商品分数与其他商品分数差值大于0.15,去掉分太低商品
        if len(confirmed) > 1:
            confirmed = self._item_name_score_filter(confirmed, search_result)

        return confirmed, options

    def _match_vector(self, item_names):
        """向量匹配: 根据LLM识别商品名称，从Milvus中查询相似商品名称(hybrid混合向最检索)"""
        search_results = []
        # 1.获取milvus客户端对象
        milvus_client = StorageClients.get_milvus_client()
        if milvus_client is None:
            return search_results

        # 2.获取向量转换模型BGE-M3
        embedding_model = AIClients.get_bge_m3_client()
        if embedding_model is None:
            logger.error(f"获取嵌入模型失败")
            return search_results

        # 3.将商品名称转换为向量,用工具类，批量转换
        # hybrid_embedding_result = {
        #     "dense": [[],[]],
        #     "sparse": [{},{}]
        # }
        hybrid_embedding_result = generate_bge_m3_hybrid_vectors(embedding_model, item_names)

        # 4.循环并装填，查询Milvus库中商品名称匹配，混合检索
        for index, extract_item_name in enumerate(item_names):
            # 4.1 获取dense向量
            dense_vector = hybrid_embedding_result["dense"][index]
            # 4.2 获取sparse向量
            sparse_vector = hybrid_embedding_result["sparse"][index]
            # 4.3  创建hybrid检索请求
            hybrid_search_request = create_hybrid_search_requests(dense_vector, sparse_vector)
            # 4.4 执行检索
            hybrid_search_result = execute_hybrid_search_query(
                milvus_client,
                collection_name=self.config.item_name_collection, #kb_item_names_v1
                search_requests=hybrid_search_request,
                ranker_weights=(0.5, 0.5),
                norm_score=True,
                output_fields=["item_name"] # 只查询item_name字段
            )
            # 4.5 装填结果
            item_name_search_result = {
                "extracted_name": extract_item_name,
                "matches": [
                    {"item_name": h["entity"]["item_name"], "score": h["distance"]}
                    for h in (hybrid_search_result[0] if hybrid_search_result else [])
                ]
            }
            search_results.append(item_name_search_result)
        return search_results

    def _item_name_score_align(self, search_result)->Tuple[List[str], List[str]]:
        """评分对齐"""
        confirmed = []
        options = []
        for item_name_search_result in search_result:
            extracted_name = item_name_search_result["extracted_name"]
            matches = sorted(item_name_search_result.get('matches'), key=lambda x: x['score'], reverse=True)
            if not matches:
                continue
            # 这段代码是商品名称评分对齐逻辑，根据向量搜索的分数决定商品名的确认或候选：
            # 高分区（≥0.7）：
            #  若存在与提取名称完全匹配的 → 直接确认
            #  若仅1条高分结果 → 直接确认
            #  若多条高分 → 取前3条作为候选选项
            # 中分区（0.6~0.7）：无高分时，取中分结果前3条作为候选选项
            # 返回：confirmed（确认列表）和 options（候选列表，最多3个）

            # 取最高分数的商品名称
            high = [m for m in matches if m['score'] >= 0.7]
            if high: #高分区
                #next表示找到第一个满足条件的就退出，None表示如果没有满足条件的元素，则返回None
                extract = next((h for h in high if str(h['item_name']) == extracted_name), None)
                if extract: # 若存在与llm提取名称完全匹配的 → 直接确认
                    picked = extract["item_name"]
                    if picked not in confirmed:
                        confirmed.append(picked)
                elif len(high) == 1: #若仅1条高分结果 → 直接确认
                    picked = high[0]["item_name"]
                    if picked not in confirmed:
                        confirmed.append(picked)
                else: #若多条高分 → 取前3条作为候选选项
                    for h in high[:3]:
                        picked = h.get('item_name')
                        if picked not in options and picked not in confirmed:
                            options.append(picked)
            else: # 中分区
                mid = [m for m in matches
                       if
                       m['score'] >= 0.6 and m.get('item_name') not in options and m.get('item_name') not in confirmed]
                if mid: #无高分时，取中分结果前3条作为候选选项
                    for m in mid[:3]:
                        picked = m.get('item_name')
                        options.append(picked)
        #返回：confirmed（确认列表）和 options（候选列表，最多3个）
        return confirmed, options[:3]

    def _item_name_score_filter(self, confirmed, search_result):
        """分数差异过滤"""
        item_name_score = {} # 记录所有高分组中（confirmed中）商品名称和分数的字典
        for item_name_search_result in search_result:
            matches = item_name_search_result.get('matches')
            for m in matches:
                score = m.get('score')
                item_name = m.get('item_name')
                if item_name in confirmed:
                    item_name_score[item_name] = max(item_name_score.get(item_name) or 0, score)

        sorted_item_name_score = sorted(item_name_score.items(), key=lambda x: x[1], reverse=True)
        max_item_name_score = sorted_item_name_score[0][1] # 最高分数
        #保留高分组中 与最高分差值小于0.15的商品名称
        return [name for name, score in item_name_score.items() if max_item_name_score - score <= 0.15]

class ItemNameConfirmNode(BaseNode):
    """商品名称确认节点
    流程: 获取历史 → LLM提取商品名 → 向量匹配 → 评分对齐 → 更新状态 → 历史回填
    数据结构:
    {
      _id: ObjectId('6a48cc0f82601667381db7a5'),
      session_id: 'sess-f41icdzoh87mr5zeel1',
      role: 'user',
      text: '万用表怎么测量电阻',
      rewritten_query: '万用表怎么测量电阻',
      item_names: [],
      ts: Double('1783155727.498414')
    }
    {
      _id: ObjectId('6a48cc0f82601667381db7a6'),
      session_id: 'sess-f41icdzoh87mr5zeel1',
      role: 'assistant',
      text: '我不确定您指的是哪款产品。, 您是在询问以下产品吗: RS PRO RS-12 数字万用表、HAK 180 扫描仪？',
      rewritten_query: '万用表怎么测量电阻',
      item_names: [],
      ts: Double('1783155727.52313')
    }
    """
    name = "item_name_confirm"

    def __init__(self):
        super().__init__()
        self._item_name_extractor = ItemNameExtractor(self.logger,self.name)
        self._item_name_aligner = ItemNameAligner(self.logger,self.name,self.config)
    def process(self, state:QueryGraphState) -> QueryGraphState:
        session_id = state.get("session_id")
        origin_query = state.get("original_query")
        # 1.获取历史
        history_messages: List[Dict[str, Any]] = get_recent_messages(session_id, limit=10)
        history_text = "" #准备提示词模版参数
        for message in history_messages:
            history_text += f"{message['role']}: {message['text']}\n"
        # 2.LLM提取商品名
        extract_item_names_result: Dict[str, Any] = self._item_name_extractor.extract_item_name(history_text, origin_query)
        print(extract_item_names_result)
        #{'item_names': ['RS PRO RS-12数字万用表', 'HAK180'], 'rewritten_query': 'RS PRO RS-12数字万用表如何测量电压，以及HAK180的介质规格有哪些？'}
        item_names = extract_item_names_result.get("item_names")
        rewritten_query = extract_item_names_result.get("rewritten_query")

        # 3.向量匹配
        # 4.评分对齐
        # 5.分数差异过滤（仅当confirmed长度大于1时）
        # confirmed 存放已经确认商品名称。存储商品名称和用户问题中的商品名称几乎一致。得分>=0.7
        # options 存放不确定用户问的是哪个商品? 选项商品。 让用户来选择其中商品，再进行下一步处理
        # state['answer'] 存在->  选项?抱歉?
        # state['answer'] 不存在-> 商品名称已确认
        if item_names:
            confirmed, options = self._item_name_aligner.match_align_filter(item_names)
        else:
            confirmed, options = [], []

        # 6.决策更新状态
        self._decide(state, item_names, confirmed, options, rewritten_query)

        # 7.历史回填仅当confirmed有值）
        if confirmed:
            for msg in history_messages:
                msg["item_names"] = confirmed
            # ids_to_update = [
            #     str(msg["_id"]) for msg in history_messages if not msg.get("item_names")
            # ]
            # if ids_to_update:
            #     try:
            #         update_message_item_names(ids_to_update, confirmed)
            #     except Exception as e:
            #         self.logger.warning(f"回填历史 item_names 失败: {e}")
            # 将历史对话写入 state，供下游使用
            state["history"] = history_messages

        return state

    def _decide(self, state: QueryGraphState, item_names: List[str],
                confirmed: List[str], options: List[str], rewritten_query: str):
        """根据对齐结果更新 state"""
        if confirmed:
            state['rewritten_query'] = rewritten_query
            state['item_names'] = confirmed
        elif options:
            state['answer'] = (
                f"我不确定您指的是哪款产品。"
                f"您是在询问以下产品吗：{'、'.join(options)}？"
            )
        else:
            state['answer'] = "抱歉，我无法识别您询问的具体产品名称，请提供更准确的产品名称或型号。"


if __name__ == '__main__':
    item_name_confirmed_node = ItemNameConfirmNode()
    init_state = {
        "session_id": 'sess-f41icdzoh87mr5zeel1',
        # "original_query": "RS-12数字万用表和H3C LA2608 室内无线网关的操作区别是什么?"
        # "original_query": "RS-12数字万用表和RS-13数字万用表的区别?"
        # "original_query": "RS-12数字万用表如何测量电压以及HAK180的介质规格有哪些?"
        # "original_query": "RS-12数字万用表如何测量电压"  # 单个商品询问
        # "original_query": "苹果手机怎么用?"
        "original_query": "今天天气怎么样?"
    }
    processed_state = item_name_confirmed_node(init_state)

    print(json.dumps(processed_state, indent=4, ensure_ascii=False))