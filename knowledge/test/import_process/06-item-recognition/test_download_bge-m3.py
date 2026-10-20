from modelscope import snapshot_download
local_dir = snapshot_download(model_id="BAAI/bge-m3",
local_dir=r"E:\zw\ai_models\modelscope_cache\models\BAAI\bge-m3")
print(local_dir)