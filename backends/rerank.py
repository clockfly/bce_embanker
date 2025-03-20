import numpy as np
import onnxruntime
from transformers import AutoTokenizer
from copy import deepcopy
from typing import List
import concurrent.futures

from config import LOCAL_RERANK_MODEL_PATH, LOCAL_RERANK_PATH, LOCAL_RERANK_MAX_LENGTH, RERANK_BATCH_SIZE, RERANK_THREADS
from logger import debug_logger
from timeutil import get_time

def sigmoid(x):
    """Apply sigmoid function with scaling and clipping"""
    x = x.astype('float32')
    scores = 1/(1+np.exp(-x))
    scores = np.clip(1.5*(scores-0.5)+0.5, 0, 1)
    return scores

class RerankOnnxBackend:
    def __init__(self, use_cpu: bool = False):
        self.use_cpu = use_cpu
        self._tokenizer = AutoTokenizer.from_pretrained(LOCAL_RERANK_PATH)
        self.spe_id = self._tokenizer.sep_token_id
        self.overlap_tokens = 80
        self.batch_size = RERANK_BATCH_SIZE
        self.max_length = LOCAL_RERANK_MAX_LENGTH
        self.return_tensors = "np"
        self.workers = RERANK_THREADS
        
        # Configure ONNX session
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 0
        sess_options.inter_op_num_threads = 0
        
        # Set providers based on CPU/GPU availability
        if use_cpu:
            providers = ['CPUExecutionProvider']
        else:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            
        self.session = onnxruntime.InferenceSession(LOCAL_RERANK_MODEL_PATH, sess_options, providers=providers)
        debug_logger.info(f"RerankClient: model_path: {LOCAL_RERANK_MODEL_PATH}")

    def inference(self, batch):
        """Run inference on a batch of inputs"""
        # Prepare input data
        inputs = {
            self.session.get_inputs()[0].name: batch['input_ids'],
            self.session.get_inputs()[1].name: batch['attention_mask']
        }

        if 'token_type_ids' in batch and len(self.session.get_inputs()) > 2:
            inputs[self.session.get_inputs()[2].name] = batch['token_type_ids']

        # Run inference
        result = self.session.run(None, inputs)
        
        # Apply sigmoid function
        sigmoid_scores = sigmoid(np.array(result[0]))

        return sigmoid_scores.reshape(-1).tolist()

    def merge_inputs(self, chunk1_raw, chunk2):
        """Merge query and passage chunks with separators"""
        chunk1 = deepcopy(chunk1_raw)

        # Add separator at the end of chunk1
        chunk1['input_ids'].append(self.spe_id)
        chunk1['attention_mask'].append(1)

        # Add chunk2 content
        chunk1['input_ids'].extend(chunk2['input_ids'])
        chunk1['attention_mask'].extend(chunk2['attention_mask'])

        # Add separator at the end of the sequence
        chunk1['input_ids'].append(self.spe_id)
        chunk1['attention_mask'].append(1)

        if 'token_type_ids' in chunk1:
            # Add token_type_ids for chunk2 and separators
            token_type_ids = [1 for _ in range(len(chunk2['token_type_ids']) + 2)]
            chunk1['token_type_ids'].extend(token_type_ids)

        return chunk1

    def tokenize_preproc(self, query: str, passages: List[str]):
        """Tokenize and preprocess query and passages"""
        query_inputs = self._tokenizer.encode_plus(query, truncation=False, padding=False)
        max_passage_inputs_length = self.max_length - len(query_inputs['input_ids']) - 2  # Account for separators

        assert max_passage_inputs_length > 10, "Query is too long, not enough space for passage"
        overlap_tokens = min(self.overlap_tokens, max_passage_inputs_length * 2 // 7)

        # Create [query, passage] pairs
        merge_inputs = []
        merge_inputs_idxs = []
        for pid, passage in enumerate(passages):
            passage_inputs = self._tokenizer.encode_plus(
                passage, 
                truncation=False, 
                padding=False,
                add_special_tokens=False
            )
            passage_inputs_length = len(passage_inputs['input_ids'])

            if passage_inputs_length <= max_passage_inputs_length:
                if passage_inputs['attention_mask'] is None or len(passage_inputs['attention_mask']) == 0:
                    continue
                qp_merge_inputs = self.merge_inputs(query_inputs, passage_inputs)
                merge_inputs.append(qp_merge_inputs)
                merge_inputs_idxs.append(pid)
            else:
                # Handle long passages by splitting them with overlap
                start_id = 0
                while start_id < passage_inputs_length:
                    end_id = start_id + max_passage_inputs_length
                    sub_passage_inputs = {k: v[start_id:end_id] for k, v in passage_inputs.items()}
                    start_id = end_id - overlap_tokens if end_id < passage_inputs_length else end_id

                    qp_merge_inputs = self.merge_inputs(query_inputs, sub_passage_inputs)
                    merge_inputs.append(qp_merge_inputs)
                    merge_inputs_idxs.append(pid)

        return merge_inputs, merge_inputs_idxs

    @get_time
    def get_rerank(self, query: str, passages: List[str]):
        """Get reranking scores for query-passage pairs"""
        tot_batches, merge_inputs_idxs_sort = self.tokenize_preproc(query, passages)

        tot_scores = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = []
            for k in range(0, len(tot_batches), self.batch_size):
                batch = self._tokenizer.pad(
                    tot_batches[k:k + self.batch_size],
                    padding=True,
                    max_length=None,
                    pad_to_multiple_of=None,
                    return_tensors=self.return_tensors
                )
                future = executor.submit(self.inference, batch)
                futures.append(future)
                
            for future in futures:
                scores = future.result()
                tot_scores.extend(scores)

        # Merge scores for the same passage (take max score)
        merge_tot_scores = [0 for _ in range(len(passages))]
        for pid, score in zip(merge_inputs_idxs_sort, tot_scores):
            merge_tot_scores[pid] = max(merge_tot_scores[pid], score)
            
        return merge_tot_scores