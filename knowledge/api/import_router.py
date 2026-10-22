import os.path
import uvicorn
from fastapi import FastAPI, File, UploadFile, Depends, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from knowledge.core.paths import get_front_page_dir
from knowledge.schema.upload_schema import UploadResponse, TaskStatusResponse
from knowledge.utils.task_util import get_task_info
from knowledge.core.deps import get_import_file_service
from knowledge.service.import_file_service import ImportFileService



def create_app() -> FastAPI:
    app = FastAPI(description="知识库导入",version="v1.0")

    # 跨域配置（允许前端跨域访问）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 挂载前端静态资源 image css
    front_page_dir = get_front_page_dir()
    if front_page_dir and os.path.exists(front_page_dir):
        app.mount("/front", StaticFiles(directory=front_page_dir))

    register_router(app)
    return app
def register_router(app: FastAPI):

    @app.get("/import")
    async def import_root():
        """返回导入页面"""
        return FileResponse(path=os.path.join(get_front_page_dir(), "import.html"))

    #fastAPI只管与前端进行接口对接，并不负责业务逻辑处理。需要定义专门业务层类来处理文件上传，开启langgraph流程
    @app.post("/upload", response_model=UploadResponse)
    async def upload_file_endpoint(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        service: ImportFileService = Depends(get_import_file_service)
    ):
        # 1. 上传文件（本地 + MinIO）
        task_id, file_dir, import_file_path = service.process_upload_file(file)

        # 2. 将耗时的图谱流程放入后台任务
        background_tasks.add_task(service.run_import_graph, task_id, file_dir, import_file_path)

        # 3. 立即返回 task_id，前端开始轮询
        return UploadResponse(message="文件上传成功", task_id=task_id)

    @app.get("/status/{task_id}", response_model=TaskStatusResponse)
    async def get_status_endpoint(task_id: str):
        """前端轮询此接口获取任务进度"""
        task_info = get_task_info(task_id)
        return TaskStatusResponse(**task_info)

if __name__ == '__main__':
    # param1:fastapi实例
    # param2:启动的服务器地址
    # param3:启动的服务端口
    uvicorn.run(app=create_app(), host="0.0.0.0", port=8000, log_level="info")