# 问器知识库（shopkeeper_brain）

面向设备文档的**知识库导入 + 智能问答**系统：把 PDF/说明书自动转成 Markdown、切片、向量化入库，用户提问时通过多路检索（向量 / HyDE / 联网）+ RRF 融合 + Rerank 精排，生成带溯源图片的专业答案。

核心链路全部用 **LangGraph** 编排，两个独立流程：

| 流程 | 作用 | 入口 | 端口 |
|------|------|------|------|
| 导入流程 `import_process` | 文档 → Markdown → 切片 → 向量 → Milvus | `knowledge/api/import_router.py` | 8000 |
| 查询流程 `query_process` | 问题 → 多路检索 → 融合精排 → 答案 | `knowledge/api/query_router.py` | 8001 |

---

## 一、技术栈

- **编排**：LangGraph（StateGraph，支持条件路由与并行分支）
- **服务**：FastAPI + Uvicorn，SSE 流式输出
- **存储**：Milvus（向量）、MongoDB（会话历史）、MinIO（图片/文件）
- **模型**：DashScope（LLM / VLM，OpenAI 兼容模式）、BGE-M3（本地稠密+稀疏向量）、BGE-Reranker-Large（本地精排）
- **文档解析**：MinerU（PDF → Markdown）

---

## 二、目录结构

```
shopkeeper_brain/
├── main.py                     # PyCharm 脚手架文件，非项目入口（可忽略）
├── README.md
└── knowledge/
    ├── .env                    # 环境变量配置（已被 .gitignore 忽略，不入仓库）
    ├── requirements.txt
    │
    ├── api/                    # 接口层：只做前后端对接，不含业务逻辑
    │   ├── import_router.py    # 导入服务：/import /upload /status/{task_id}
    │   └── query_router.py     # 查询服务：/chat /query /stream/{task_id} /history/{session_id}
    │
    ├── service/                # 业务层：上传落盘、图谱调度
    │   ├── import_file_service.py
    │   └── query_service.py
    │
    ├── processor/
    │   ├── import_process/     # 导入流程图谱
    │   │   ├── main_graph.py   # 图的装配与编译
    │   │   ├── state.py        # ImportGraphState
    │   │   └── nodes/          # 各节点实现
    │   └── query_process/      # 查询流程图谱
    │       ├── main_graph.py
    │       ├── state.py        # QueryGraphState
    │       ├── config.py       # 阈值与集合名等配置
    │       └── nodes/
    │
    ├── prompt/                 # 提示词模板
    ├── schema/                 # 请求/响应 Pydantic 模型
    ├── utils/                  # 客户端与工具
    │   ├── client/             # ai_clients（LLM/BGE）、storage_clients（Milvus/MinIO/Mongo）
    │   ├── milvus_util.py      # 混合检索封装
    │   ├── mongo_history_util.py
    │   ├── task_util.py        # 任务进度与节点中文名映射
    │   └── sse_util.py
    ├── front/                  # 前端页面 import.html / chat.html
    └── test/                   # 分阶段测试用例（01-env、02-mineru、... day12-rrf）
```

---

## 三、环境准备

```bash
cd knowledge
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 配置 `knowledge/.env`

项目通过 `python-dotenv` 加载，**配置缺失时 `StorageClients` / `AIClients` 会直接抛错**。

| 分类 | 变量 | 说明 |
|------|------|------|
| 大模型 | `DASHSCOPE_API_KEY` | DashScope API Key（LLM、VLM、MCP 共用） |
| | `OPENAI_API_BASE` | OpenAI 兼容模式 base url |
| | `LLM_DEFAULT_MODEL` / `ITEM_MODEL` / `VL_MODEL` / `KG_MODEL` | 各场景模型名 |
| 本地模型 | `BGE_M3_PATH` | BGE-M3 本地目录（离线） |
| | `BGE_RERANKER_LARGE` | bge-reranker-large 本地目录 |
| | `BGE_DEVICE` / `BGE_FP16` | 如 `cuda:0` / `1` |
| 向量库 | `MILVUS_URL` | 如 `http://192.168.6.150:19530` |
| | `CHUNKS_COLLECTION` / `ITEM_NAME_COLLECTION` / `ENTITY_NAME_COLLECTION` | 集合名 |
| | `MILVUS_METRIC_TYPE` / `MILVUS_MIN_COSINE_SCORE` | 度量方式与最低分阈值 |
| 会话库 | `MONGO_URL` / `MONGO_DB_NAME` | 历史会话存储 |
| 对象存储 | `MINIO_ENDPOINT` / `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` / `MINIO_BUCKET_NAME` | 图片与文件 |
| 文档解析 | `MINERU_API_BASE` | MinerU 服务地址 |
| 联网搜索 | `MCP_DASHSCOPE_BASE_URL` | MCP 服务地址 |
| 检索调优 | `RRF_K`(60) / `RRF_MAX_RESULTS`(10) / `MAX_CONTEXT_CHARS`(12000) / `EMBEDDING_SEARCH_LIMIT` / `HYDE_SEARCH_LIMIT` | 括号内为默认值 |

> 若 Milvus 开启了认证，连接串需带凭据（当前 `StorageClients.get_milvus_client()` 只传 uri，认证环境下需自行补充 token）。

---

## 四、启动服务

```bash
# 在项目根目录执行（模块内使用 knowledge.xxx 绝对导入，必须把根目录加入 sys.path）
python -m knowledge.api.import_router     # 导入服务 http://localhost:8000
python -m knowledge.api.query_router      # 查询服务 http://localhost:8001
```

前端页面：`http://localhost:8000/import`（上传）、`http://localhost:8001/chat`（问答）。

---

## 五、导入流程

```
entry_node                     检查文件类型
      │
      ├── (PDF) ──> pdf_to_md_node ──┐   MinerU 转换，按文件名重命名产物
      │                              │
      └── (MD) ────────────────────>├──> md_img_node        图片上传 MinIO + VLM 摘要回填
                                     │
                                     v
                             document_split_node      按二级标题切分
                                     │
                                     v
                            item_name_rec_node        LLM 识别商品名
                                     │
                                     v
                             bge_embedding_node       BGE-M3 稠密+稀疏向量
                                     │
                                     v
                            import_milvus_node        写入 Milvus
                                     │
                                     v
                                    END
```

关键说明：

- **`md_img_node`**：扫描 Markdown 中的 `![](images/xxx.jpg)`，把图片上传到 MinIO，用可访问 URL 替换相对路径，并在图片下方插入 VLM 生成的摘要。切片里的图片链接即由此产生，是问答阶段展示图片的数据来源。
- **`pdf_to_md_node`**：MinerU 产物默认带 UUID 文件名，节点内会统一重命名为 `{文件标题}_*`，并同步修正 Markdown 内的图片引用。
- 上传接口立即返回 `task_id`，图谱在后台执行，前端轮询 `/status/{task_id}` 获取进度。

---

## 六、查询流程

```
item_name_confirm              结合历史识别商品名，确认/澄清
      │
      ├── (已有答案：需澄清或无法识别) ─────────────> answer_output
      │
      └── (无答案) ──> multi_search  ── 并行分发 ──┬──> search_embedding        向量混合检索
                                                   ├──> search_embedding_hyde   HyDE 假设性文档检索
                                                   └──> web_search_mcp         联网搜索（不参与融合）
                                                        │
                                                        v
                                                       join
                                                        │
                                                        v
                                                       rrf         倒排融合（1/(k+rank)）
                                                        │
                                                        v
                                                      rerank       BGE-Reranker 精排
                                                        │
                                                        v
                                                  answer_output    生成答案 + 写历史
                                                        │
                                                        v
                                                       END
```

关键说明：

- **`item_name_confirm`**：拉取最近 10 条会话，LLM 提取商品名并改写问题；置信度中等时返回候选让用户确认，低置信度直接致歉，此时跳过检索直达 `answer_output`。
- **`multi_search` / `join`**：LangGraph 的虚拟分发与汇合节点，用于实现三路并行。
- **`rrf`**：只依赖各路**名次**（不看分数绝对值），因此各路结果必须按相关度降序——Milvus 混合检索返回时已保证该顺序。
- **`rerank`**：本地 `FlagReranker` 对融合结果精排，按分差过滤后输出最终上下文。
- **`answer_output`**：组装提示词 → LLM 生成（流式 SSE / 非流式）→ 写入 MongoDB 历史。

### 接口一览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/chat` | 问答页面 |
| POST | `/query` | 提交问题；`is_stream=true` 返回 `task_id`，走 SSE |
| GET | `/stream/{task_id}` | SSE 流式推送进度与答案 |
| GET | `/history/{session_id}` | 拉取会话历史（时间正序） |
| DELETE | `/history/{session_id}` | 清空会话历史 |
| POST | `/upload` | 上传文档（导入服务） |
| GET | `/status/{task_id}` | 导入/处理进度 |

---

## 七、常见坑与排查

| 现象 | 原因 | 处理 |
|------|------|------|
| `Repo id must use alphanumeric chars ... 'D:\xxx\bge-reranker-large'` | **本地模型目录不存在**，transformers 回退成 HuggingFace repo id 解析 | 检查 `.env` 里模型路径是否真实存在、盘符是否正确 |
| `Fail connecting to server on ...19530`，`DEADLINE_EXCEEDED` | Milvus 地址不可达或服务未启动（认证失败则是另一种报错） | ping 目标机、确认容器运行；内网地址需在同一局域网/VPN |
| `'HumanMessage' object has no attribute 'items'` | `llm.invoke()` 第二个位置参数是 `config`，多条消息必须作为 **list** 传第一个参数 | `invoke([SystemMessage(...), HumanMessage(...)])` |
| `'module' object is not callable`（`@cache`） | 误导入 `langgraph_sdk` 的 `cache` 模块 | 用 `from functools import cache` |
| 接口 404 | 路由字符串拼写错误（如 `"/upl oad"` 混入空格） | 核对 `@app.post("/upload")` |
| 答案里出现 `example.com/xxx.png` 这类假图 | 模型幻觉；提示词已限制只能原样复制参考内容中的图片链接 | 若仍出现，需在代码侧用正则校验 URL 是否存在于上下文中 |
| 刷新后历史消息顺序颠倒 | `get_recent_messages` 按 `ts` 降序取最近 N 条 | 已改为取回后反转，返回时间正序 |
| 阶段进度显示英文节点名 | `task_util._NODE_NAME_TO_CN` 的 key 必须与节点类的 `name` 完全一致 | 新增节点时同步补映射 |

---

## 八、开发约定

1. **接口层不含业务逻辑**：`api/` 只做参数校验与响应封装，业务一律下沉到 `service/`。
2. **节点职责单一**：每个节点实现 `process(state) -> state`，通过 `state` 传递数据，不跨节点共享可变全局。
3. **客户端单例**：`AIClients` / `StorageClients` 内部用锁 + 缓存复用连接，勿自行 new。
4. **配置走环境变量**：不写死地址、密钥、模型路径；`.env` 不入仓库。
5. **节点中文名同步维护**：新增查询流程节点时，务必在 `knowledge/utils/task_util.py` 的 `_NODE_NAME_TO_CN` 中补充，否则前端进度显示英文。
