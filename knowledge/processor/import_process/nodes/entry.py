import json
from pathlib import Path

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState


class EntryNode(BaseNode):
    """
    入口节点
    根据上传文件后缀，来决定走哪个分支节点（.pdf, .md）
    """
    # 实现父类中继承的方法，每个子节点都需要处理自己那一步任务。任务日志、任务执行时间都由父类处理
    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        入口节点
        根据文件后缀决定处理流程
        .pdf -> pdf_to_md_node -> md_img_node
        .md -> md_img_node
        :param state:
            import_file_path = "E:/zw/project/file/doc/万用表RS-12的使用.pdf"
            file_dir = "E:/zw/project/file/temp_dir"
        :return:
            "is_pdf_enabled": True,
            "is_md_enabled": False,
            "import_file_path": "E:/zw/project/file/doc/万用表RS-12的使用.pdf",
            "file_dir": "E:/zw/project/file/temp_dir"
        """
        self.log_step("step1", "获取上传的文件以及路径")
        # 1.获取上传的文件以及路径
        import_file_path = state.get("import_file_path", "")
        file_dir = state.get("file_dir", "")
        self.log_step("step2", "判断文件是否为空")

        # 2.判断文件是否为空
        if not import_file_path:
            raise ValidationError("文件路径为空",self.name)

        self.log_step("step3", "获取文件后缀，并变成小写")
        # 3.获取文件后缀，并变成小写
        # file_ext = os.path.splitext(file_path)[1].lower()
        path = Path(import_file_path)
        if not path.exists():
            raise ValidationError(f"文件不存在：{import_file_path}",self.name)
        suffix = path.suffix.lower()

        # 4.不同文件类型做不同状态修改（字段名与 state.py / main_graph.py 路由保持一致）
        if suffix == ".pdf":
            state["is_pdf_read_enabled"] = True
            state["is_md_read_enabled"] = False
            state["pdf_path"] = str(path.parent)
        elif suffix == ".md":
            state["is_md_read_enabled"] = True
            state["is_pdf_read_enabled"] = False
            state["md_path"] = import_file_path
        else:
            raise ValidationError(f"不支持的文件类型：{import_file_path}-{suffix}",self.name)
        # 5.获取文件标题,不带扩展名
        file_name = path.stem
        state["file_title"] = file_name
        # 6.更新状态并返回状态
        return state

if __name__ == "__main__":
    setup_logging()
    import_file_path = r"E:\zw\project\file\doc\万用表RS-12的使用.pdf"
    file_dir = r"E:\zw\project\file\temp_dir"
    state = {
        "is_pdf_read_enabled": False,
        "is_md_read_enabled": False,
        "import_file_path": import_file_path,
        "file_dir": file_dir
    }

    entry = EntryNode()
    processed_state = entry(state)
    print(json.dumps(processed_state, indent=4, ensure_ascii=False))