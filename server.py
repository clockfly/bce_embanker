from sanic import Sanic
from sanic.response import json
from typing import List, Union, Dict, Any
import time

from backends.embed import EmbeddingOnnxBackend
from backends.rerank import RerankOnnxBackend
from timeutil import get_time_async
from config import EMBED_SERVER_PORT
import argparse
from logger import setup_loggers
import os

# Create a global app instance
app = Sanic("embedding_server")

# OpenAI-compatible embedding endpoint
@get_time_async
@app.route("/v1/embeddings", methods=["POST"])
async def openai_embedding(request):
    data = request.json
    input_texts = data.get('input')
    model = data.get('model', 'text-embedding-ada-002')  # Not used, but tracked for response
    
    # Handle both single text and array of texts
    if isinstance(input_texts, str):
        input_texts = [input_texts]
    
    
    onnx_backend = request.app.ctx.embedding_backend
    embeddings = onnx_backend.predict(input_texts)
    
    # Format response according to OpenAI spec
    data_response = []
    for i, embedding in enumerate(embeddings):
        data_response.append({
            "object": "embedding",
            "embedding": embedding,
            "index": i
        })
    
    result = {
        "object": "list",
        "data": data_response,
        "model": model
    }
    
    return json(result)

# Legacy embedding endpoint (keep for backward compatibility)
@get_time_async
@app.route("/embedding", methods=["POST"])
async def embedding(request):
    data = request.json
    texts = data.get('texts')
    
    onnx_backend = request.app.ctx.embedding_backend
    result_data = onnx_backend.predict(texts)
    
    return json(result_data)

# Jina-compatible reranker endpoint
@get_time_async
@app.route("/v1/rerank", methods=["POST"])
async def jina_rerank(request):
    data = request.json
    query = data.get('query')
    documents = data.get('documents', [])
    model = data.get('model', 'jina-reranker-v1')  # Not used functionally, just for response
    top_n = data.get('top_n', len(documents))
    
    # Extract text from documents if they are dictionaries
    passages = []
    for doc in documents:
        if isinstance(doc, dict) and 'text' in doc:
            passages.append(doc['text'])
        elif isinstance(doc, str):
            passages.append(doc)
    
    # If no passages provided, return empty results
    if not passages:
        return json({
            "model": model,
            "usage": {"total_tokens": 0},
            "results": []
        })
    
    
    onnx_backend = request.app.ctx.rerank_backend
    scores = onnx_backend.get_rerank(query, passages)
    
    # Create result items with index, document, and score
    results = []
    for i, (score, doc) in enumerate(zip(scores, documents)):
        doc_text = doc if isinstance(doc, str) else doc.get('text', '')
        results.append({
            "index": i,
            "document": {"text": doc_text},
            "relevance_score": score
        })
    
    # Sort by score descending and limit to top_n
    results = sorted(results, key=lambda x: x["relevance_score"], reverse=True)[:top_n]
    
    response = {
        "model": model,
        "results": results
    }
    
    return json(response)

# Legacy rerank endpoint (keep for backward compatibility)
@get_time_async
@app.route("/rerank", methods=["POST"])
async def rerank(request):
    data = request.json
    query = data.get('query')
    passages = data.get('passages')

    onnx_backend = request.app.ctx.rerank_backend
    result_data = onnx_backend.get_rerank(query, passages)
    
    return json(result_data)    

@app.listener('before_server_start')
async def setup_onnx_backend(app, loop):
    # These will be initialized with proper values in main()
    app.ctx.use_gpu = False
    app.ctx.embedding_backend = EmbeddingOnnxBackend(use_cpu=not app.ctx.use_gpu)
    app.ctx.rerank_backend = RerankOnnxBackend(use_cpu=not app.ctx.use_gpu)

def main():
    parser = argparse.ArgumentParser(description="Run embedding and reranking servers")
    parser.add_argument('--use_gpu', action="store_true", help='Use GPU for inference')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers per server')
    parser.add_argument('--service', type=str, default='all', choices=['all', 'embedding', 'rerank'], 
                        help='Which service to run')
    
    args = parser.parse_args()
    logger = setup_loggers()
    logger.info(f"Starting with args: {args}")
    
    # Check if model files exist
    from config import LOCAL_EMBED_MODEL_PATH, LOCAL_RERANK_MODEL_PATH
    if not os.path.exists(LOCAL_EMBED_MODEL_PATH):
        logger.error(f"Embedding model not found at {LOCAL_EMBED_MODEL_PATH}")
        return
    if not os.path.exists(LOCAL_RERANK_MODEL_PATH):
        logger.error(f"Reranking model not found at {LOCAL_RERANK_MODEL_PATH}")
        return
    
    # Store use_gpu setting in app context for access in listener
    app.ctx.use_gpu = args.use_gpu
    
    # Handle service selection
    if args.service != 'all':
        # Only keep routes for the specified service
        routes_to_keep = []
        for route in app.router.routes:
            if args.service == 'embedding' and ('embedding' in route.uri or 'embeddings' in route.uri):
                routes_to_keep.append(route)
            elif args.service == 'rerank' and 'rerank' in route.uri:
                routes_to_keep.append(route)
        
        # Create a new router with only the desired routes
        old_routes = list(app.router.routes)
        for route in old_routes:
            if route not in routes_to_keep:
                app.router.remove(route)
    
    # Run the app
    app.run(host="0.0.0.0", port=EMBED_SERVER_PORT, workers=args.workers)

if __name__ == "__main__":
    main()