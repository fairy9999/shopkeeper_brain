from functools import cache

from knowledge.service.import_file_service import ImportFileService

#专业创建对象，并进行缓存管理。可以实现对象共享。
# @lru_cache # 缓存满了，可以根据Lru(最近最少使用)策略进行缓存清理
@cache  # functools.cache 等价于 maxsize=None 的 lru_cache，缓存满了不清理
def get_import_file_service() -> ImportFileService:
    return ImportFileService()