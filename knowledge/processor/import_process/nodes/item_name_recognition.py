import json
import os
from typing import Tuple, List

from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus import MilvusClient, DataType
from rich.json import JSON

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import StateFieldError, ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.prompt.import_prompt import ITEM_NAME_USER_PROMPT_TEMPLATE, ITEM_NAME_SYSTEM_PROMPT
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


class ItemNameRecognitionNode(BaseNode):
    '''
    商品名称识别节点
    利用3个chunk的内容上下文 小于2500字符。通过提示词模板让LLM识别出商品名称及型号。将商品名称信息Bge-m3向量化存MILVUS，供后续匹配使用。
    '''
    name = "item_name_rec_node"
    def process(self, state:ImportGraphState) -> ImportGraphState:
        # 1. 参数校验
        file_title, chunks, item_name_chunks_k, item_name_chunk_size = self._validate_state(state)

        # 2. 构建商品名识别上下文(取3个chunk的内容上下文，小于2500字符)
        item_name_recognition_context = self._prepare_item_name_recognition_context(
            chunks, item_name_chunks_k, item_name_chunk_size
        )

        # 3. LLM商品名识别，获取客户端AIClient
        item_name = self._recognition_name(file_title, item_name_recognition_context)

        # 4. 向量化提取到商品名(使用Bge-m3模型)
        dense_vector: List[float]
        sparse_vector: dict[float]
        dense_vector, sparse_vector = self._embedding_item_name(item_name)

        # 5. 存储到milvus中，获取客户端StorageClients,创建集合，数据保存
        self._insert_milvus(file_title, item_name, dense_vector, sparse_vector,
                            self.config.item_name_collection)

        # 6. 回填item_name信息，将item_name网填每一个chunk对象中。也同时回填state对象中，
        self._fill_item_name(item_name, state, chunks)

        # 7. 备份chunks,给后续步骤测试使用，
        # self._backup_chunks(state, chunks)

        return state

    def _validate_state(self, state: ImportGraphState) ->Tuple[str,List,int,int]:
        file_title = state.get('file_title')
        chunks = state.get('chunks')

        if not file_title:
            raise StateFieldError(node_name=self.name, field_name="file_title", expected_type=str)
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(node_name=self.name, field_name="chunks", expected_type=list)

        item_name_chunks_k = self.config.item_name_chunk_k
        if not item_name_chunks_k or item_name_chunks_k <= 0:
            raise ValidationError(message="item_name_chunk_k为空或者无效", node_name=self.name)

        item_name_chunk_size = self.config.item_name_chunk_size
        if not item_name_chunk_size or item_name_chunk_size <= 0:
            raise ValidationError(message="item_name_chunk_size为空或者无效", node_name=self.name)

        return file_title, chunks, item_name_chunks_k, item_name_chunk_size

    def _prepare_item_name_recognition_context(self, chunks, item_name_chunks_k, item_name_chunk_size)->str:
        total = 0
        final_context = []
        for index, chunk in enumerate(chunks[:item_name_chunks_k]):
            if not chunk or not isinstance(chunk, dict):
                continue
            chunk_content = chunk.get('content')
            context = f"【切片】-{index}-{chunk_content}"

            if total + len(context) > item_name_chunk_size:
                break

            total += len(context)
            final_context.append(context)

        return "\n".join(final_context)

    def _recognition_name(self, file_title, item_name_recognition_context)->str:
        try:
            # 获取LLM客户端
            llm_client = AIClients.get_llm_openai(response_format=False)
            # 构建用户提示
            user_prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(
                file_title=file_title, context=item_name_recognition_context
            )
            # 调用LLM
            llm_response = llm_client.invoke([
                SystemMessage(content=ITEM_NAME_SYSTEM_PROMPT),# 系统提示
                HumanMessage(content=user_prompt)# 用户提示
            ])

            response_dict = json.loads(llm_response.content.strip())
            llm_result = response_dict.get("item_name")

            if not llm_result or llm_result == "UNKNOWN":
                self.logger.info(f"LLM未识别出商品名，降级使用标题: {file_title}")
                return file_title

            self.logger.info(f"LLM提取到商品名: {llm_result}")
            return llm_result
        except Exception as e:
            self.logger.error(f"LLM调用失败，降级使用标题: {file_title}，异常: {e}")
            return file_title

    def _embedding_item_name(self, item_name)->Tuple[List[float], List[float]]:
        try:
            # 获取向量化客户端
            bge_m3_client = AIClients.get_bge_m3_client()
            # 2. 生成嵌入
            embeddings = bge_m3_client.encode_documents([item_name])

            # 3. 提取向量
            dense_vector = embeddings["dense"][0].tolist()  # List[float], 长度 1024
            sparse_matrix = embeddings["sparse"]  # CSR 稀疏矩阵

            # 4. 从 CSR 矩阵提取稀疏向量
            start_idx = sparse_matrix.indptr[0]
            end_idx = sparse_matrix.indptr[1]
            token_ids = sparse_matrix.indices[start_idx:end_idx].tolist()
            weights = sparse_matrix.data[start_idx:end_idx].tolist()
            sparse_vector = dict(zip(token_ids, weights))  # Dict[int, float]


            return dense_vector, sparse_vector
        except ConnectionError as e:
            self.logger.error(f"BGE-M3 客户端获取失败: {e}")
            return None, None
        except Exception as e:
            self.logger.error(f"BGE-M3 向量化失败: {e}")
            return None, None

    def _insert_milvus(self, file_title, item_name, dense_vector, sparse_vector, item_name_collection):
        if not dense_vector or not sparse_vector:
            self.logger.error(f"文档{file_title} 对应的商品名{item_name} 向量生成不完整")
            return
        try:
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            self.logger.error(f"Milvus 客户端创建失败: {e}")
            return
        try:
            if not milvus_client.has_collection(item_name_collection):
                self.logger.info(f"集合 {item_name_collection} 不存在，正在创建集合...")
                self._create_item_name_collection(item_name_collection, milvus_client)
                self.logger.info(f"集合 {item_name_collection} 创建成功")
            # 插入数据
            data = {
                "file_title": file_title,
                "item_name": item_name,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector
            }
            milvus_client.insert(
                collection_name=item_name_collection,
                data=[data],
            )
        except Exception as e:
            self.logger.error(f"Milvus插入数据失败: {e}")

    def _create_item_name_collection(self, collection_name, milvus_client):
        schema = milvus_client.create_schema()
        schema.add_field(field_name="pk", datatype=DataType.VARCHAR,
                         is_primary=True, auto_id=True, max_length=100) # 主键字段
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535) #标量
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535) # 标量
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024) # 稠密向量
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR) # 稀疏向量

        index_param = milvus_client.prepare_index_params()
        index_param.add_index(field_name="dense_vector", index_name="dense_vector_index",
                              index_type="AUTOINDEX", metric_type="COSINE") # 模糊近似查询 余弦相似度 只考虑方向、夹角，不考虑长度
        index_param.add_index(field_name="sparse_vector", index_name="sparse_vector_index",
                              index_type="SPARSE_INVERTED_INDEX", metric_type="IP") # IP内积 方向长度都考虑(查询更精准) 如果归一化后，与COSINE相似

        milvus_client.create_collection(collection_name=collection_name,
                                        schema=schema, index_params=index_param)
        self.logger.info(f"集合 {collection_name} 创建成功并构建了索引")

    def _fill_item_name(self, item_name, state, chunks):
        for chunk in chunks:
            chunk["item_name"] = item_name
        state["item_name"] = item_name

    def _backup_chunks(self, state, chunks):
        """
        将切分结果备份到json文件
        :param state:
        :param chunks:
        :return:
        """
        local_dir = state.get("file_dir", "")
        print(f"备份切分结果到: {local_dir}")
        if not local_dir:
            return

        os.makedirs(local_dir, exist_ok=True)  # exist_ok=True：如果目录已存在，不会抛出异常，直接跳过
        output_path = os.path.join(local_dir, "chunks_item_name.json")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=4)
        except Exception as e:
            self.logger.warning(f"备份失败：{e}")

# 测试
if __name__ == '__main__':
    setup_logging()

    # 1. 读取chunk.json
    chunk_json_path = r"E:\zw\project\file\temp_dir\万用表RS-12的使用_online\chunks.json"
    with open(chunk_json_path, "r", encoding="utf-8") as f:
        chunk_content = json.load(f)

    # 2. 构建state
    state = {
        "file_title": "万用表RS-12的使用_new",
        "chunks": chunk_content,
        "file_dir": r"E:\zw\project\file\temp_dir\万用表RS-12的使用_online"
    }

    # 3. 实例化节点
    node = ItemNameRecognitionNode()

    # 4. 调用process
    result = node(state)

    # 5. 输出结果
    print(f"商品名: {result.get('item_name')}")
    print(f"chunks数量: {len(result.get('chunks', []))}")
    print(f"首个chunk是否含item_name: {'item_name' in result['chunks'][0]}")





