from langchain_text_splitters import RecursiveCharacterTextSplitter

text = "苹果的颜色是红色的[SEP]香蕉的颜色是黄色的[SEP]橘子的颜色是橙色的"

# chunk_size：每个分块的最大字符数（长度，不是"条数"）
#   太小 → 语义被切碎，检索到的上下文不完整
#   太大 → 单块信息冗余，向量精度下降，且容易超出模型 token 上限
# 经验取值（按场景）：
#   - FAQ / 短问答 / 对话：100 ~ 300
#   - 商品说明书、知识库文档问答（本项目场景）：300 ~ 800
#   - 长文 / 论文 / 书籍：800 ~ 1500
# 一般还需配合 chunk_overlap（重叠量）≈ chunk_size 的 10% ~ 20%，避免边界语义断裂
# 本示例为演示 [SEP] 分隔符的切割效果，故意设为 10
splitter_false = RecursiveCharacterTextSplitter(
    chunk_size=10,
    chunk_overlap=0,
    keep_separator=False,
    # 注意：separators 必须是 list；传字符串 "[SEP]" 会被逐字符拆分
    separators=["[SEP]"]
)

print(splitter_false.split_text(text))
