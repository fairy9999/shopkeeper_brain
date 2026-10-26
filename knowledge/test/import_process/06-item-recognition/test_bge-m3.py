from pymilvus.model.hybrid import BGEM3EmbeddingFunction

# 1. 加载模型
bge_m3 = BGEM3EmbeddingFunction(
    # model_name="BAAI/bge-m3", #在线模型
    # device="cuda:0",
    # use_fp16=True,
    model_name=r"E:\zw\ai_models\modelscope_cache\models\BAAI\bge-m3",
    device="cpu",
    use_fp16=False,
)

# 2. 生成嵌入
embeddings = bge_m3.encode_documents(["RS-12 数字万用表"])

# 3. 提取向量
dense_vector = embeddings["dense"][0].tolist()   # List[float], 长度 1024
sparse_matrix = embeddings["sparse"]              # CSR 稀疏矩阵

# 4. 从 CSR 矩阵提取稀疏向量
start_idx = sparse_matrix.indptr[0]
end_idx = sparse_matrix.indptr[1]
token_ids = sparse_matrix.indices[start_idx:end_idx].tolist()
weights = sparse_matrix.data[start_idx:end_idx].tolist()
sparse_vector = dict(zip(token_ids, weights))    # Dict[int, float]

print(start_idx, end_idx, token_ids, weights)
print(sparse_vector)