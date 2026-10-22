import logging, os.path, shutil, uuid
from datetime import datetime
from typing import Tuple
from fastapi import UploadFile
from knowledge.core.paths import get_local_base_dir
from knowledge.processor.import_process.exceptions import FileProcessingError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.task_util import add_running_task, add_done_task, update_task_status
from knowledge.processor.import_process.main_graph import kb_import_graph_app

logger = logging.getLogger(__name__)


class ImportFileService:

    def _get_date_dir(self) -> str:
        # %Y%m%d:年月日
        # %Y:四位 %y:两位
        return os.path.join(get_local_base_dir(), datetime.now().strftime("%Y%m%d"))

    def process_upload_file(self, file: UploadFile) -> Tuple[str, str, str]:
        """
        处理上传文件：
        1. 生成 task_id，构建归档目录
        2. 标记 upload_file 节点为运行中
        3. 保存文件到本地磁盘
        4. 同步上传到 MinIO
        5. 标记 upload_file 节点完成
        6. 返回 (task_id, file_dir, import_file_path)
        """
        date_dir = self._get_date_dir()
        task_id = str(uuid.uuid4().hex[:8])  # 真正的随机 获取前8个随机数
        file_dir = os.path.join(date_dir, task_id)

        add_running_task(task_id, "upload_file")
        import_file_path = self._save_upload_file_to_local(file, file_dir)
        self._save_upload_file_to_minio(import_file_path, file.filename)
        add_done_task(task_id, "upload_file")

        return task_id, file_dir, import_file_path

    def _save_upload_file_to_local(self, file: UploadFile, file_dir: str) -> str:
        """
        保存文件到临时目录
        Args:
            file: 文件上传对象
            file_dir: 上传文件的目录
        Returns:
        """
        # 1. 创建文件的归属目录
        os.makedirs(file_dir, exist_ok=True)

        # 2. 构建导入文件的路径
        import_file_path = os.path.join(file_dir, file.filename)

        # 3. 写入
        try:
            with  open(import_file_path, "wb") as f:
                # 不同的操作系统以及不同python版本都可以分批次的写入（windows版本以及3.7以上的sdk版本:1m）
                shutil.copyfileobj(file.file, f)
        except IOError as e:
            logger.info(f"{file.filename}写入临时目录失败 原因:{str(e)}")
            raise FileProcessingError(message=f"{file.filename}写入临时目录失败 原因:{str(e)}")

        # 4. 返回导入的文件路径
        return import_file_path

    def _save_upload_file_to_minio(self, import_file_path: str, filename: str):
        """
        Args:
            import_file_path:  上传文件的地址
            filename: 上传文件的名字
        Returns:
        """

        # 1. 获取minio客户端
        try:
            minio_client = StorageClients.get_minio_client()
        except ConnectionError as e:
            logger.error(f"MinIO客户端获取失败 原因:{str(e)}")
            return

        # 2. 获取minio相关信息
        bucket_name = os.getenv('MINIO_BUCKET_NAME')
        object_name = f"origin_files/{datetime.now().strftime('%Y%m%d')}/{filename}"

        # 3. 上传
        try:
            minio_client.fput_object(bucket_name, object_name, import_file_path)
        except Exception as e:
            logger.error(f"{filename}上传到MinIO失败 原因：{str(e)}")

    def run_import_graph(self, task_id: str, file_dir: str, import_file_path: str):
        """ 运行导入 LangGraph 流水线（在后台任务中执行） """
        try:
            update_task_status(task_id, "processing")

            global_graph_init_status: ImportGraphState = {
                "task_id": task_id,
                "file_dir": file_dir,
                "import_file_path": import_file_path
            }

            for event in kb_import_graph_app.stream(global_graph_init_status):
                for node_name, state in event.items():
                    print(f"[{task_id}] Completed Node: {node_name}")

            update_task_status(task_id, "completed")
        except Exception as e:
            update_task_status(task_id, "failed")
            print(f"[{task_id}] Error: {e}")