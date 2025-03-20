import os

# Model paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
LOCAL_EMBED_PATH = os.path.join(MODEL_DIR, "embed")
LOCAL_EMBED_MODEL_PATH = os.path.join(LOCAL_EMBED_PATH, "embed.onnx")
LOCAL_RERANK_PATH = os.path.join(MODEL_DIR, "reranker")
LOCAL_RERANK_MODEL_PATH = os.path.join(LOCAL_RERANK_PATH, "rerank.onnx")

# Model parameters
LOCAL_EMBED_BATCH = 1
EMBED_MAX_LENGTH = 384
RERANK_BATCH_SIZE = 8
LOCAL_RERANK_MAX_LENGTH = 512
RERANK_THREADS = 4

# Server settings
EMBED_SERVER_PORT = 9001
RERANK_SERVER_PORT = 8001
SERVER_WORKERS = 1