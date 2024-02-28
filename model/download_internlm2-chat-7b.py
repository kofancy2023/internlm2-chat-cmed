# pip install modelscope
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os

# cache_dir: 模型下载的缓存目录
model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm2-chat-7b', cache_dir='./Shanghai_AI_Laboratory')