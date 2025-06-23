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
from ..core import RerankModel, RerankModelSpec, _ModelWrapper
from ..utils import preprocess_sentence

logger = logging.getLogger(__name__)


class SentenceTransformerRerankModel(RerankModel):
    def load(self):
        logger.info(
            "Loading rerank model with sentence-transformers: %s", self._model_path
        )

        try:
            import sentence_transformers
            from sentence_transformers.cross_encoder import CrossEncoder

            if sentence_transformers.__version__ < "3.1.0":
                raise ValueError(
                    "The sentence_transformers version must be greater than 3.1.0. "
                    "Please upgrade your version via `pip install -U sentence_transformers` or refer to "
                    "https://github.com/UKPLab/sentence-transformers"
                )
        except ImportError:
            error_message = "Failed to import module 'sentence-transformers'"
            installation_guide = [
                "Please make sure 'sentence-transformers' is installed. ",
                "You can install it by `pip install sentence-transformers`\n",
            ]
            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        self._model = CrossEncoder(
            self._model_path,
            device=self._device,
            trust_remote_code=True,
            max_length=getattr(self._model_spec, "max_tokens"),
            **self._model_config,
        )

        if self._use_fp16:
            self._model.model.half()

        # Wrap transformers model to record number of tokens
        self._model.model = _ModelWrapper(self._model.model)

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

        pre_query = preprocess_sentence(
            query, kwargs.get("instruction", None), self._model_spec.model_name
        )
        sentence_combinations = [[pre_query, doc] for doc in documents]

        # reset n tokens
        self._model.model.n_tokens = 0

        logger.debug("Passing processed sentences: %s", sentence_combinations)
        similarity_scores = self._model.predict(
            sentence_combinations,
            convert_to_numpy=False,
            convert_to_tensor=True,
            **kwargs,
        ).cpu()

        if similarity_scores.dtype == torch.bfloat16:
            similarity_scores = similarity_scores.float()

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

    @classmethod
    def check_lib(cls) -> bool:
        return importlib.util.find_spec("sentence_transformers") is not None

    @classmethod
    def match_json(cls, model_spec: RerankModelSpec) -> bool:
        # sentence-transformers 支持 normal 类型的模型
        return model_spec.type == "normal"
