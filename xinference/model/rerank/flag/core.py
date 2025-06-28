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
import uuid
from typing import List, Optional

import numpy as np
import torch

from ....types import Document, DocumentObj, Rerank, RerankTokens
from ..core import RerankModel, RerankModelSpec
from ..utils import preprocess_sentence

logger = logging.getLogger(__name__)


class FlagRerankModel(RerankModel):
    def load(self):

        try:
            if self._model_spec.type == "LLM-based":
                from FlagEmbedding import FlagLLMReranker as FlagReranker
            elif self._model_spec.type == "LLM-based layerwise":
                from FlagEmbedding import LayerWiseFlagLLMReranker as FlagReranker
            else:
                raise RuntimeError(
                    f"Unsupported Rank model type: {self._model_spec.type}"
                )
        except ImportError:
            error_message = "Failed to import module 'FlagEmbedding'"
            installation_guide = [
                "Please make sure 'FlagEmbedding' is installed. ",
                "You can install it by `pip install FlagEmbedding`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")
        self._model = FlagReranker(self._model_path, use_fp16=self._use_fp16)
        # Wrap transformers model to record number of tokens
        self._model.model = _ModelWrapper(self._model.model)

    def _load_qwen3_model(self, flash_attn_installed: bool):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self._model_path, padding_side="left")
        enable_flash_attn = self._model_config.get("enable_flash_attn", True)
        model_kwargs = {"device_map": "auto"}

        if flash_attn_installed and enable_flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            model_kwargs["torch_dtype"] = torch.float16

        model_kwargs.update(self._model_config)
        logger.debug("Loading qwen3 rerank with kwargs %s", model_kwargs)

        model = AutoModelForCausalLM.from_pretrained(
            self._model_path, **model_kwargs
        ).eval()

        max_length = getattr(self._model_spec, "max_tokens")

        # 设置 qwen3 特定的处理逻辑
        self._setup_qwen3_processing(tokenizer, model, max_length)

    def _load_general_model(self, flash_attn_installed: bool):
        # 通用的 transformers 模型加载逻辑
        from transformers import AutoModel, AutoTokenizer

        model_kwargs = {"device_map": "auto"}
        if self._use_fp16:
            model_kwargs["torch_dtype"] = torch.float16

        model_kwargs.update(self._model_config)

        self._model = AutoModel.from_pretrained(self._model_path, **model_kwargs).eval()

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)

    def _setup_qwen3_processing(self, tokenizer, model, max_length):
        prefix = (
            "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
            'and the Instruct provided. Note that the answer can only be "yes" or "no".'
            "<|im_end|>\n<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

        def process_inputs(pairs):
            inputs = tokenizer(
                pairs,
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
            )
            for i, ele in enumerate(inputs["input_ids"]):
                inputs["input_ids"][i] = prefix_tokens + ele + suffix_tokens
            inputs = tokenizer.pad(
                inputs, padding=True, return_tensors="pt", max_length=max_length
            )
            for key in inputs:
                inputs[key] = inputs[key].to(model.device)
            return inputs

        token_false_id = tokenizer.convert_tokens_to_ids("no")
        token_true_id = tokenizer.convert_tokens_to_ids("yes")

        def compute_logits(inputs, **kwargs):
            batch_scores = model(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, token_true_id]
            false_vector = batch_scores[:, token_false_id]
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().tolist()
            return scores

        self._model = model
        self._tokenizer = tokenizer
        self.process_inputs = process_inputs
        self.compute_logits = compute_logits

    def rerank(
        self,
        documents: List[str],
        query: str,
        top_n: Optional[int],
        max_chunks_per_doc: Optional[int],
        return_documents: Optional[bool],
        return_len: Optional[bool],
        **kwargs,
    ) -> Rerank:
        assert self._model is not None
        if max_chunks_per_doc is not None:
            raise ValueError("rerank hasn't support `max_chunks_per_doc` parameter.")

        logger.info("Rerank with kwargs: %s, model: %s", kwargs, self._model)

        if "qwen3" in self._model_spec.model_name.lower():
            similarity_scores = self._rerank_qwen3(documents, query, **kwargs)
        else:
            similarity_scores = self._rerank_general(documents, query, **kwargs)

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
            # 简化的 token 计算
            input_len = sum(len(self._tokenizer.encode(doc)) for doc in documents)
            output_len = input_len
        else:
            input_len = output_len = 0

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
        self._clean_cache_if_needed()

        return Rerank(id=str(uuid.uuid1()), results=docs, meta=metadata)

    def _rerank_qwen3(self, documents: List[str], query: str, **kwargs):
        def format_instruction(instruction, query, doc):
            if instruction is None:
                instruction = "Given a web search query, retrieve relevant passages that answer the query"
            output = (
                "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
                    instruction=instruction, query=query, doc=doc
                )
            )
            return output

        pairs = [
            format_instruction(kwargs.get("instruction", None), query, doc)
            for doc in documents
        ]
        inputs = self.process_inputs(pairs)
        similarity_scores = self.compute_logits(inputs)
        return similarity_scores

    def _rerank_general(self, documents: List[str], query: str, **kwargs):
        # 通用的重排序逻辑
        pre_query = preprocess_sentence(
            query, kwargs.get("instruction", None), self._model_spec.model_name
        )

        scores = []
        for doc in documents:
            # 简化的相似度计算
            inputs = self._tokenizer(
                [pre_query, doc], return_tensors="pt", padding=True, truncation=True
            )

            with torch.no_grad():
                outputs = self._model(**inputs)
                # 简化的评分逻辑
                score = torch.cosine_similarity(
                    outputs.last_hidden_state[0].mean(dim=0),
                    outputs.last_hidden_state[1].mean(dim=0),
                    dim=0,
                ).item()
                scores.append(score)

        return scores

    @classmethod
    def check_lib(cls) -> bool:
        return importlib.util.find_spec("transformers") is not None

    @classmethod
    def match_json(cls, model_spec: RerankModelSpec) -> bool:
        # transformers 引擎支持所有类型的模型
        return True
