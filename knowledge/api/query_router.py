"""查询路由"""

import os
import asyncio
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, Depends
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from knowledge.core.paths import get_front_page_dir
from knowledge.core.deps import get_query_service
from knowledge.schema.query_schema import QueryRequest, QueryResponse, StreamSubmitResponse
from knowledge.service.query_service import QueryService
from knowledge.utils.sse_util import sse_generator, create_sse_queue
from knowledge.processor.query_process.base import setup_logging


def create_app() -> FastAPI:
    app = FastAPI(title="Query Service", description="知识库查询服务")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # http://localhost:8001/front/chat.html
    front_page_dir = get_front_page_dir()
    if front_page_dir and os.path.exists(front_page_dir):
        app.mount("/front", StaticFiles(directory=front_page_dir))
    register_routes(app)
    return app


def register_routes(app: FastAPI):
    @app.get("/chat")
    async def chat_page():
        path = os.path.join(get_front_page_dir(), "chat.html")
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="chat.html not found")
        return FileResponse(path)

    @app.post("/query", response_model=QueryResponse | StreamSubmitResponse)
    async def query(
            request: QueryRequest,
            background_tasks: BackgroundTasks,
            service: QueryService = Depends(get_query_service),
    ):

        # 1. 获取session_id
        session_id = request.session_id or service.generate_session_id()

        # 2. 获取任务_id
        task_id = service.generate_task_id()

        # 3. 开启流式
        if request.is_stream:
            # 3.1 必须在返回响应前创建队列，否则前端请求 /stream 时队列不存在
            create_sse_queue(task_id)

            # 3.2 后台运行graph
            background_tasks.add_task(
                service.run_query_graph, task_id, session_id, request.query, True
            )

            # 3.3 返回响应
            return StreamSubmitResponse(
                message="Query submitted", session_id=session_id, task_id=task_id
            )

        # 4. 非流式：丢到线程池[默认]避免阻塞事件循环
        loop = asyncio.get_running_loop()
        #None 表示使用默认的线程池:ThreadPoolExecutor
        await loop.run_in_executor(
            None, service.run_query_graph, task_id, session_id, request.query, False
        )

        # 5. 获取答案
        answer = service.get_answer(task_id)

        # 6. 返回答案
        return QueryResponse(message="处理完成", session_id=session_id, task_id=task_id, answer=answer)

    @app.get("/stream/{task_id}")
    async def stream(task_id: str, request: Request) -> StreamingResponse:
        return StreamingResponse(
            sse_generator(task_id, request), media_type="text/event-stream",
        )

    @app.get("/history/{session_id}")
    async def get_history(
            session_id: str, limit: int = 50,
            service: QueryService = Depends(get_query_service),
    ):
        try:
            items = service.get_history(session_id, limit)
            return {"session_id": session_id, "items": items}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"history error: {e}")

    @app.delete("/history/{session_id}")
    async def clear_chat_history(
            session_id: str,
            service: QueryService = Depends(get_query_service),
    ):
        count = service.clear_history(session_id)
        return {"message": "History cleared", "deleted_count": count}


if __name__ == "__main__":
    setup_logging()
    uvicorn.run(app=create_app(), host="0.0.0.0", port=8001)