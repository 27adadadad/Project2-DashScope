"""共享语料配对评测：推理只消费问题/语料，标注只在推理完成后评分。"""

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
from statistics import mean

from app.evaluation.retrieval_metrics import METRIC_NAMES, aggregate_metrics, calculate_retrieval_metrics
from app.services.hybrid_retrieval import HybridTextUnitRetriever
from app.services.rag_chat_service import RAGChatService

DATASET_DIR = Path(__file__).resolve().parents[2] / "evals" / "datasets"
DEFAULT_DATASET_PATH = DATASET_DIR / "rag_eval.jsonl"
DEFAULT_CORPUS_PATH = DATASET_DIR / "shared_corpus.jsonl"
# 兼容现有数据集；空结果为线上系统错误，不生成该回答。
INSUFFICIENT_EVIDENCE_ANSWER = "资料中没有足够依据，无法确认。"


def fingerprint(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_samples(path=DEFAULT_DATASET_PATH):
    samples = _load_jsonl(path)
    required = ("id", "category", "question", "reference_answer", "reference_context_ids", "reference_contexts")
    for sample in samples:
        if any(key not in sample for key in required) or "contexts" in sample:
            raise ValueError("invalid_sample_schema")
        if not isinstance(sample["question"], str) or not sample["question"].strip():
            raise ValueError("empty_question")
        if not isinstance(sample["reference_context_ids"], list) or not isinstance(sample["reference_contexts"], list):
            raise ValueError("invalid_reference_schema")
    if len({sample["id"] for sample in samples}) != len(samples):
        raise ValueError("duplicate_sample_id")
    return samples


def load_corpus(path=DEFAULT_CORPUS_PATH):
    corpus = _load_jsonl(path)
    if any(not {"id", "text", "source"}.issubset(item) or not item["text"].strip() for item in corpus):
        raise ValueError("invalid_corpus_schema")
    if len({item["id"] for item in corpus}) != len(corpus):
        raise ValueError("duplicate_corpus_id")
    return corpus


def validate_references(samples, corpus):
    by_id = {item["id"]: item["text"] for item in corpus}
    for sample in samples:
        ids = sample["reference_context_ids"]
        if len(set(ids)) != len(ids) or any(item not in by_id for item in ids):
            raise ValueError("invalid_reference_ids")
        if [by_id[item] for item in ids] != sample["reference_contexts"]:
            raise ValueError("reference_text_mismatch")


def safe_error(error, stage):
    # 不保存可能携带请求头/密钥的供应商异常原文。
    return {"stage": stage, "reason": "timeout" if isinstance(error, TimeoutError) else "exception",
            "error_type": type(error).__name__}


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def _status(rows):
    if not rows:
        return "not_run"
    good = sum(row["retrieval_status"] == "completed" and row["generation_status"] != "failed" for row in rows)
    return "ok" if good == len(rows) else "partial" if good else "failed"


class EvaluationRunner:
    def __init__(self, *, embed_query, embed_documents, answer_generator=None, k=5,
                 candidate_limit=20, rrf_k=60, timeout=120, mode="offline_test", metadata=None):
        if k <= 0 or candidate_limit < k or rrf_k <= 0 or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("invalid_evaluation_parameters")
        if mode not in {"offline_test", "offline_smoke", "online"}:
            raise ValueError("invalid_run_mode")
        self.embed_query, self.embed_documents = embed_query, embed_documents
        self.answer_generator = answer_generator
        self.k, self.candidate_limit, self.rrf_k = k, candidate_limit, rrf_k
        self.timeout, self.mode, self.metadata = timeout, mode, metadata or {}

    async def _call(self, function, *args):
        async def invoke():
            if inspect.iscoroutinefunction(function):
                value = function(*args)
            else:
                value = await asyncio.to_thread(function, *args)
            return await value if inspect.isawaitable(value) else value
        return await asyncio.wait_for(invoke(), timeout=self.timeout)

    async def run(self, samples, corpus):
        validate_references(samples, corpus)
        if len({s["id"] for s in samples}) != len(samples) or len({c["id"] for c in corpus}) != len(corpus):
            raise ValueError("duplicate_id")
        strategies = {"dense_baseline": [], "hybrid_rrf": []}
        corpus_error = None
        document_vectors = []
        try:
            if corpus:
                document_vectors = await self._call(self.embed_documents, [item["text"] for item in corpus])
                if len(document_vectors) != len(corpus):
                    raise ValueError("document_vector_count_mismatch")
                fingerprint(document_vectors)
        except Exception as error:
            corpus_error = safe_error(error, "corpus_embedding")

        query_vector_hashes = {}
        for index, sample in enumerate(samples):
            if corpus_error:
                for rows in strategies.values():
                    rows.append(self._failed_row(sample, corpus_error))
                continue
            try:
                query_vector = await self._call(self.embed_query, sample["question"]) if corpus else []
                query_vector_hashes[sample["id"]] = fingerprint(query_vector)
            except Exception as error:
                for rows in strategies.values():
                    rows.append(self._failed_row(sample, safe_error(error, "query_embedding")))
                continue
            # 向量冻结共享，两策略中的回调不再发起模型请求。
            retriever = HybridTextUnitRetriever(corpus, lambda _: deepcopy(query_vector),
                lambda _: deepcopy(document_vectors), candidate_limit=self.candidate_limit, rrf_k=self.rrf_k)
            methods = {"dense_baseline": retriever.retrieve_dense, "hybrid_rrf": retriever.retrieve}
            order = list(methods) if index % 2 == 0 else list(reversed(methods))
            for name in order:
                try:
                    results = await asyncio.wait_for(methods[name](sample["question"], limit=self.k, strict=True), self.timeout)
                except Exception as error:
                    strategies[name].append(self._failed_row(sample, safe_error(error, "retrieval")))
                else:
                    strategies[name].append(await self._row(sample, results))
        return {
            "schema_version": 2, "status": _status([row for rows in strategies.values() for row in rows]),
            "sample_count": len(samples), "corpus_count": len(corpus),
            "run": {"mode": self.mode, "online_evaluation": "run" if self.mode == "online" else "not_run",
                "quality_claim_allowed": self.mode == "online", "created_at": datetime.now(timezone.utc).isoformat(),
                "dataset_role": "previously_exposed_regression_set", "tuning_performed": False,
                "dataset_sha256": fingerprint(samples), "corpus_sha256": fingerprint(corpus),
                "document_vectors_sha256": None if corpus_error else fingerprint(document_vectors),
                "query_vectors_sha256": query_vector_hashes, "k": self.k, "candidate_limit": self.candidate_limit,
                "rrf_k": self.rrf_k, "timeout_seconds": self.timeout, "strategy_order": "alternating_by_sample",
                "scope": "text_unit_dense_vs_bm25_rrf_without_graph_channel",
                "gate": "production_nonempty_context_only", "empty_gate_behavior": "retrieval_unavailable_error",
                "generation": "enabled" if self.answer_generator else "not_run",
                "prompt_sha256": fingerprint(RAGChatService.build_prompt_messages([], [])),
                "provider": deepcopy(self.metadata)},
            "strategies": {name: self._summary(rows) for name, rows in strategies.items()},
            "comparison": self._comparison(strategies),
            "limitations": ["Text retrieval subsystem only; production also appends GraphRAG Local Search output.",
                "Gold IDs score task answerability and annotated evidence coverage, not factual correctness.",
                "Final refusal/correctness require human review; no string heuristic is used.",
                "Regression questions were previously exposed; no held-out generalization claim.",
                "Timeout cancels the awaiting task; an SDK worker may continue until its own timeout."],
        }

    async def _row(self, sample, results):
        # 此阶段标准答案/证据标注没有进入门控或生成。
        allowed = RAGChatService.has_retrieval_context(results)
        row = {"id": sample["id"], "question": sample["question"], "category": sample["category"],
            "retrieved_ids": [item["id"] for item in results], "contexts": deepcopy(results),
            "retrieval_status": "completed", "gate_allowed": allowed, "answer": None,
            "generation_status": "not_run" if allowed else "skipped_gate", "errors": [],
            "answer_review": {"status": "not_generated", "is_refusal": None, "is_factually_correct": None,
                "is_supported_by_context": None, "reviewer": None, "rationale": None}}
        scores = [item["dense_score"] for item in results if isinstance(item.get("dense_score"), (int, float))
                  and not isinstance(item["dense_score"], bool) and math.isfinite(item["dense_score"])]
        row["top_dense_score"] = max(scores) if scores else None
        row["dense_score_valid_count"] = len(scores)
        row["dense_score_missing_count"] = len(results) - len(scores)
        if allowed and self.answer_generator is not None:
            try:
                answer = await self._call(self.answer_generator, sample["question"], deepcopy(results))
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError("empty_or_invalid_model_answer")
                row.update(answer=answer, generation_status="completed", answer_sha256=fingerprint(answer))
                row["answer_review"]["status"] = "pending_review"
            except Exception as error:
                row["generation_status"] = "failed"
                row["errors"].append(safe_error(error, "generation"))
        return self._score_row(sample, row)

    def _score_row(self, sample, row):
        # 仅评分消费标签；FP/FN 对照题目在共享库中的可答标注。
        reference = sample["reference_context_ids"]
        valid = row["retrieval_status"] == "completed"
        row["reference_context_ids"] = list(reference)
        row["reference_answer"] = sample["reference_answer"]
        row["answerable"] = bool(reference)
        row["metrics"] = calculate_retrieval_metrics(row["retrieved_ids"], reference, k=self.k) if valid and reference else None
        row["gate_false_accept"] = bool(not reference and row["gate_allowed"]) if valid else None
        row["gate_false_reject"] = bool(reference and not row["gate_allowed"]) if valid else None
        row["gate_allowed_with_incomplete_reference"] = bool(reference and row["gate_allowed"]
            and not set(reference).issubset(row["retrieved_ids"])) if valid else None
        return row

    def _failed_row(self, sample, error):
        return self._score_row(sample, {"id": sample["id"], "question": sample["question"], "category": sample["category"],
            "retrieved_ids": [], "contexts": [], "retrieval_status": "failed", "gate_allowed": None,
            "answer": None, "generation_status": "not_run", "top_dense_score": None,
            "dense_score_valid_count": 0, "dense_score_missing_count": 0, "errors": [deepcopy(error)],
            "answer_review": {"status": "not_generated", "is_refusal": None, "is_factually_correct": None,
                "is_supported_by_context": None, "reviewer": None, "rationale": None}})

    @staticmethod
    def _summary(rows, *, categories=True):
        valid = [row for row in rows if row["retrieval_status"] == "completed"]
        answerable = [row for row in valid if row["answerable"]]
        unanswerable = [row for row in valid if not row["answerable"]]
        generated = [row for row in rows if row["generation_status"] == "completed"]
        similarities = [row["top_dense_score"] for row in valid if row["top_dense_score"] is not None]
        refusal_reviews = [row for row in generated if row["answer_review"]["is_refusal"] is not None]
        factual_reviews = [row for row in generated if row["answer_review"]["is_refusal"] is False
                           and row["answer_review"]["is_factually_correct"] is not None]
        support_reviews = [row for row in generated if not row["answerable"] and row["answer_review"]["is_refusal"] is False
                           and row["answer_review"]["is_supported_by_context"] is not None]
        result = {**aggregate_metrics([row["metrics"] for row in answerable]), "status": _status(rows),
            "total_count": len(rows), "retrieval_valid_count": len(valid), "retrieval_failed_count": len(rows) - len(valid),
            "answerable_count": sum(row["answerable"] for row in rows),
            "unanswerable_count": sum(not row["answerable"] for row in rows),
            "retrieval_metric_denominator": len(answerable),
            "generation_completed_count": len(generated),
            "generation_failed_count": sum(row["generation_status"] == "failed" for row in rows),
            "generation_not_run_count": sum(row["generation_status"] == "not_run" for row in rows),
            "system_gate_blocked_count": sum(row["generation_status"] == "skipped_gate" for row in rows),
            "gate_false_accept_rate": _rate(sum(row["gate_false_accept"] for row in unanswerable), len(unanswerable)),
            "gate_false_accept_denominator": len(unanswerable),
            "gate_false_reject_rate": _rate(sum(row["gate_false_reject"] for row in answerable), len(answerable)),
            "gate_false_reject_denominator": len(answerable),
            "average_top_similarity": mean(similarities) if similarities else None,
            "similarity_valid_count": len(similarities), "similarity_missing_count": len(valid) - len(similarities),
            "model_refusal_rate": _rate(sum(row["answer_review"]["is_refusal"] for row in refusal_reviews), len(refusal_reviews)),
            "model_refusal_reviewed_count": len(refusal_reviews),
            "model_actual_error_rate": _rate(sum(not row["answer_review"]["is_factually_correct"] for row in factual_reviews), len(factual_reviews)),
            "model_actual_error_reviewed_count": len(factual_reviews),
            "unanswerable_unsupported_answer_rate": _rate(sum(not row["answer_review"]["is_supported_by_context"] for row in support_reviews), len(support_reviews)),
            "unanswerable_support_reviewed_count": len(support_reviews),
            "pending_review_count": sum(row["answer_review"]["status"] == "pending_review" for row in generated),
            "retrieval_failed_ids": [row["id"] for row in rows if row["retrieval_status"] == "failed"],
            "generation_failed_ids": [row["id"] for row in rows if row["generation_status"] == "failed"],
            "gate_false_accept_ids": [row["id"] for row in rows if row["gate_false_accept"]],
            "gate_false_reject_ids": [row["id"] for row in rows if row["gate_false_reject"]],
            "incomplete_reference_ids": [row["id"] for row in answerable if row["metrics"]["all_evidence_at_k"] < 1],
            "rows": rows}
        if categories:
            result["by_category"] = {category: EvaluationRunner._summary([row for row in rows if row["category"] == category], categories=False)
                                     for category in sorted({row["category"] for row in rows})}
            for summary in result["by_category"].values():
                summary.pop("rows")
        return result

    @staticmethod
    def _comparison(strategies):
        left = {row["id"]: row for row in strategies["dense_baseline"]}
        right = {row["id"]: row for row in strategies["hybrid_rrf"]}
        paired = [id for id in left if left[id]["retrieval_status"] == right[id]["retrieval_status"] == "completed"]
        scored = [id for id in paired if left[id]["answerable"]]
        def compare(ids):
            scores = {}
            for metric in METRIC_NAMES:
                dense = [left[id]["metrics"][metric] for id in ids]
                hybrid = [right[id]["metrics"][metric] for id in ids]
                delta = [b - a for a, b in zip(dense, hybrid)]
                scores[metric] = {"dense": mean(dense) if dense else None, "hybrid": mean(hybrid) if hybrid else None,
                    "delta_hybrid_minus_dense": mean(delta) if delta else None,
                    "wins": [id for id, value in zip(ids, delta) if value > 0],
                    "losses": [id for id, value in zip(ids, delta) if value < 0],
                    "tie_count": sum(value == 0 for value in delta)}
            return {"denominator": len(ids), "metrics": scores}
        return {"paired_valid_count": len(paired), "excluded_pair_ids": [id for id in left if id not in paired],
            **compare(scored), "by_category": {category: compare([id for id in scored if left[id]["category"] == category])
                                               for category in sorted({row["category"] for row in left.values()})}}


def write_report(report, directory):
    content = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    path = Path(directory) / f"rag-eval-{datetime.now():%Y%m%d-%H%M%S-%f}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
