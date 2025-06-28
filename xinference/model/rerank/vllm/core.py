# Copyright 2022-2025 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import logging
import math
import uuid
from typing import List, Optional

import numpy as np
import torch

from ....types import Document, DocumentObj, Rerank, RerankTokens
from ..core import RerankModel, RerankModelSpec
from ..utils import preprocess_sentence

logger = logging.getLogger(__name__)


# only support qwen3 reranker for now. Date:2025-0628 By Pengjunfeng
class VLLMRerankModel(RerankModel):
    def load(self):
        try:
            from vllm import LLM
            from transformers import AutoTokenizer, is_torch_npu_available
        except ImportError:
            error_message = "Failed to import module 'vllm'"
            installation_guide = [
                "Please make sure 'vllm' is installed. ",
                "You can install it by `pip install vllm`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        self._model = LLM(
            model=self._model_path,
            tensor_parallel_size=number_of_gpu,
            max_model_len=10000,
            enable_prefix_caching=True,
            gpu_memory_utilization=0.8,
        )

        _tokenizer = AutoTokenizer.from_pretrained(self._model_path)

        number_of_gpu = torch.cuda.device_count()
        _tokenizer.padding_side = "left"
        _tokenizer.pad_token = _tokenizer.eos_token
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        max_length = 8192
        suffix_tokens = _tokenizer.encode(suffix, add_special_tokens=False)
        true_token = _tokenizer("yes", add_special_tokens=False).input_ids[0]
        false_token = _tokenizer("no", add_special_tokens=False).input_ids[0]
        _sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            logprobs=20,
            allowed_token_ids=[true_token, false_token],
        )

        # def format_instruction(instruction, query, doc):
        #     text = [
        #         {
        #             "role": "system",
        #             "content": 'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".',
        #         },
        #         {
        #             "role": "user",
        #             "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}",
        #         },
        #     ]
        #     return text

        # def process_inputs(pairs, instruction, max_length, suffix_tokens):
        def process_inputs(pairs, **kwargs):
            messages = [
                format_instruction(instruction, query, doc) for query, doc in pairs
            ]
            messages = _tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            messages = [ele[:max_length] + suffix_tokens for ele in messages]
            messages = [TokensPrompt(prompt_token_ids=ele) for ele in messages]
            return messages

        # def compute_logits(model, messages, _sampling_params, true_token, false_token):
        def compute_logits(model, **kwargs):
            outputs = model.generate(messages, _sampling_params, use_tqdm=False)
            scores = []
            for i in range(len(outputs)):
                final_logits = outputs[i].outputs[0].logprobs[-1]
                token_count = len(outputs[i].outputs[0].token_ids)
                if true_token not in final_logits:
                    true_logit = -10
                else:
                    true_logit = final_logits[true_token].logprob
                if false_token not in final_logits:
                    false_logit = -10
                else:
                    false_logit = final_logits[false_token].logprob
                true_score = math.exp(true_logit)
                false_score = math.exp(false_logit)
                score = true_score / (true_score + false_score)
                scores.append(score)
            return scores

        self.process_inputs = process_inputs
        self.compute_logits = compute_logits

    @staticmethod
    def _get_detailed_instruct(task_description: str, query: str) -> str:
        return f"Instruct: {task_description}\nQuery:{query}"

    def rerank(
        self,
        documents: List[str],
        query: str,
        top_n: Optional[int],
        max_chunks_per_doc: Optional[int],
        return_documents: Optional[bool],
        return_len: Optional[bool],
        **kwargs,
    ):
        assert self._model is not None
        assert "qwen3" in self._model_spec.model_name.lower()
        if max_chunks_per_doc is not None:
            raise ValueError("rerank hasn't support `max_chunks_per_doc` parameter.")
        logger.info("Rerank with kwargs: %s, model: %s", kwargs, self._model)

        pre_query = preprocess_sentence(
            query, kwargs.get("instruction", None), self._model_spec.model_name
        )
        sentence_combinations = [[pre_query, doc] for doc in documents]
        if "qwen3" in self._model_spec.model_name.lower():

            def format_instruction(instruction, query, doc):
                if instruction is None:
                    instruction = "Given a web search query, retrieve relevant passages that answer the query"
                output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
                    instruction=instruction, query=query, doc=doc
                )
                return output

            # reduce memory usage.
            micro_bs = 4
            similarity_scores = []
            for i in range(0, len(documents), micro_bs):
                sub_docs = documents[i : i + micro_bs]
                pairs = [
                    format_instruction(kwargs.get("instruction", None), query, doc)
                    for doc in sub_docs
                ]
                # Tokenize the input texts
                inputs = self.process_inputs(pairs)
                similarity_scores.extend(self.compute_logits(inputs))
        sim_scores_argsort = list(reversed(np.argsort(similarity_scores)))
        if top_n is not None:
            sim_scores_argsort = sim_scores_argsort[:top_n]
        if return_documents:
            docs = [
                DocumentObj(
                    index=int(arg),
                    relevance_score=float(similarity_scores[arg]),
                    document=Document(text=documents[arg]),
                )
                for arg in sim_scores_argsort
            ]
        else:
            docs = [
                DocumentObj(
                    index=int(arg),
                    relevance_score=float(similarity_scores[arg]),
                    document=None,
                )
                for arg in sim_scores_argsort
            ]
        if return_len:
            input_len = self._model.model.n_tokens
            # Rerank Model output is just score or documents
            # while return_documents = True
            output_len = input_len

        # api_version, billed_units, warnings
        # is for Cohere API compatibility, set to None
        metadata = {
            "api_version": None,
            "billed_units": None,
            "tokens": (
                RerankTokens(input_tokens=input_len, output_tokens=output_len)
                if return_len
                else None
            ),
            "warnings": None,
        }

        del similarity_scores
        # clear cache if possible
        self._counter += 1
        if self._counter % RERANK_EMPTY_CACHE_COUNT == 0:
            logger.debug("Empty rerank cache.")
            gc.collect()
            empty_cache()

        return Rerank(id=str(uuid.uuid1()), results=docs, meta=metadata)

    @classmethod
    def check_lib(cls) -> bool:
        return importlib.util.find_spec("vllm") is not None

    @classmethod
    def match_json(cls, model_spec: EmbeddingModelSpec) -> bool:
        prefix = model_spec.model_name.split("-", 1)[0]
        if prefix in SUPPORTED_MODELS_PREFIXES:
            return True
        return False

    # @classmethod
    # def check_lib(cls) -> bool:
    #     return importlib.util.find_spec("transformers") is not None

    # @classmethod
    # def match_json(cls, model_spec: RerankModelSpec) -> bool:
    #     # transformers 引擎支持所有类型的模型
    #     return True
