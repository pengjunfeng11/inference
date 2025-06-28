# Copyright 2022-2023 XProbe Inc.
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

import shutil

from ...core import RerankModelSpec, cache, create_rerank_model_instance

TEST_MODEL_SPEC = RerankModelSpec(
    model_name="bge-reranker-base",
    type="normal",
    max_tokens=512,
    language=["en", "zh"],
    model_id="BAAI/bge-reranker-base",
    model_revision="465b4b7ddf2be0a020c8ad6e525b9bb1dbb708ae",
)


def test_rerank_model_with_transformers():
    model_path = None

    try:
        model_path = cache(TEST_MODEL_SPEC)

        model, _ = create_rerank_model_instance(
            "mock",
            None,
            "mock",
            "bge-reranker-base",
            "flag",
            model_path=model_path,
        )
        model.load()

        # 测试单个查询和文档列表
        query = "what is the capital of China?"
        documents = [
            "Beijing is the capital of China.",
            "Shanghai is the largest city in China.",
            "The Great Wall of China is a famous landmark.",
            "China has a population of over 1.4 billion people.",
        ]

        # 测试rerank功能
        result = model.rerank(query, documents, top_n=3)
        assert len(result.results) == 3
        assert all(hasattr(doc, "relevance_score") for doc in result.results)
        assert all(hasattr(doc, "index") for doc in result.results)
        assert all(hasattr(doc, "document") for doc in result.results)

        # 验证分数是降序排列的
        scores = [doc.relevance_score for doc in result.results]
        assert scores == sorted(scores, reverse=True)

        # 测试不返回文档内容
        result_no_docs = model.rerank(query, documents, top_n=2, return_documents=False)
        assert len(result_no_docs.results) == 2
        assert all(doc.document is None for doc in result_no_docs.results)

        # 测试返回所有文档（不指定top_n）
        result_all = model.rerank(query, documents)
        assert len(result_all.results) == len(documents)

        # 测试返回token信息
        result_with_tokens = model.rerank(query, documents, top_n=2, return_len=True)
        assert result_with_tokens.meta["tokens"] is not None
        assert hasattr(result_with_tokens.meta["tokens"], "input_tokens")
        assert hasattr(result_with_tokens.meta["tokens"], "output_tokens")

    finally:
        if model_path is not None:
            shutil.rmtree(model_path, ignore_errors=True)


def test_rerank_model_edge_cases():
    model_path = None

    try:
        model_path = cache(TEST_MODEL_SPEC)

        model, _ = create_rerank_model_instance(
            "mock",
            None,
            "mock",
            "bge-reranker-base",
            "transformers",
            model_path=model_path,
        )
        model.load()

        query = "test query"

        # 测试空文档列表
        result_empty = model.rerank(query, [])
        assert len(result_empty.results) == 0

        # 测试单个文档
        single_doc = ["This is a single document."]
        result_single = model.rerank(query, single_doc)
        assert len(result_single.results) == 1

        # 测试top_n大于文档数量
        few_docs = ["doc1", "doc2"]
        result_large_topn = model.rerank(query, few_docs, top_n=10)
        assert len(result_large_topn.results) == 2

    finally:
        if model_path is not None:
            shutil.rmtree(model_path, ignore_errors=True)
