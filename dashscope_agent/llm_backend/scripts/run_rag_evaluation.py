"""运行共享语料 RAG 对照；默认只打印计划，在线模式必须显式指定。"""

import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.rag_evaluator import (DEFAULT_CORPUS_PATH, DEFAULT_DATASET_PATH,
    EvaluationRunner, load_corpus, load_samples, write_report)
from app.services.rag_chat_service import RAGChatService


def _plan(sample_count=80, corpus_count=40, answerable_count=60, *, ragas=False, retrieval_only=False):
    return {"status": "plan_only", "online_evaluation": "not_run", "quality_claim_allowed": False,
        "sample_count": sample_count, "corpus_count": corpus_count, "top_k": 5,
        "embedding_calls": {"document_batches": math.ceil(corpus_count / 20), "query": sample_count},
        "generation_calls_max": 0 if retrieval_only else sample_count * 2,
        "ragas": {"requested": ragas, "expected_eligible_rows": answerable_count * 2 if ragas else 0,
                  "rows_max": sample_count * 2 if ragas else 0,
                  "metric_families": 3 if ragas else 0, "provider_call_count": "implementation-dependent" if ragas else 0},
        "network": "disabled_until_explicit_--online", "secrets": "not_read"}


def _safe_code_hash():
    paths = [Path(__file__), Path(__file__).resolve().parents[1] / "app/evaluation/rag_evaluator.py",
             Path(__file__).resolve().parents[1] / "app/evaluation/retrieval_metrics.py"]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.name).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


async def _offline_query(_question): return [1.0, 0.0]


async def _offline_documents(texts):
    return [[1.0, 0.0] if any(token in text for token in ("型号", "保修", "支持", "订单", "摄像头")) else [0.0, 1.0] for text in texts]


async def _offline_answer(_question, contexts): return f"离线冒烟回答：{contexts[0]['content']}"


class EvaluationGenerationError(RuntimeError):
    def __init__(self, reason_code):
        self.reason_code = reason_code
        super().__init__(reason_code)


async def dashscope_answer(question, contexts):
    """复用线上 Prompt；SSE 缺 error/done/正文时均作为失败。"""
    from app.services.model_service_factory import ModelServiceFactory
    messages = RAGChatService.build_prompt_messages([{"role": "user", "content": question}], contexts)
    parts, saw_done = [], False
    try:
        async for event in ModelServiceFactory.create_chat_service().generate_stream(messages, thinking=False):
            if not isinstance(event, str) or not event.startswith("data: "):
                raise EvaluationGenerationError("model_stream_invalid")
            try:
                payload = json.loads(event[6:].strip())
            except (TypeError, ValueError):
                raise EvaluationGenerationError("model_stream_invalid") from None
            kind = payload.get("type")
            if kind == "error": raise EvaluationGenerationError("model_stream_error")
            if kind == "content":
                content = payload.get("content")
                if isinstance(content, str): parts.append(content)
            elif kind == "done": saw_done = True
        if not saw_done: raise EvaluationGenerationError("model_stream_incomplete")
        answer = "".join(parts).strip()
        if not answer: raise EvaluationGenerationError("model_answer_empty")
        return answer
    except EvaluationGenerationError: raise
    except Exception as error: raise EvaluationGenerationError("model_stream_exception") from error


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="确定性本地冒烟，不调用模型")
    parser.add_argument("--online", action="store_true", help="显式调用真实 Embedding/生成接口")
    parser.add_argument("--retrieval-only", action="store_true", help="只测检索，不生成回答")
    parser.add_argument("--ragas", action="store_true", help="在线生成后请求 Ragas；必须与 --online 同用")
    parser.add_argument("--plan", action="store_true", help="只打印调用计划")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output-dir", type=Path)
    return parser


async def main(argv=None):
    args = _parser().parse_args(argv)
    if not args.offline and not args.online: args.plan = True
    if args.offline and args.online: raise SystemExit("只能选择 --offline 或 --online")
    if args.ragas and not args.plan and (not args.online or args.retrieval_only):
        raise SystemExit("--ragas 必须与 --online 一起使用，且不能 --retrieval-only")
    if args.k <= 0 or args.k > 20 or not math.isfinite(args.timeout) or args.timeout <= 0:
        raise SystemExit("--k 必须为 1..20，--timeout 必须为正有限数")
    samples, corpus = load_samples(DEFAULT_DATASET_PATH), load_corpus(DEFAULT_CORPUS_PATH)
    if args.plan:
        answerable_count = sum(bool(sample["reference_context_ids"]) for sample in samples)
        print(json.dumps(_plan(len(samples), len(corpus), answerable_count, ragas=args.ragas, retrieval_only=args.retrieval_only), ensure_ascii=False, indent=2))
        return 0
    output_dir = args.output_dir or (Path(__file__).resolve().parents[1] / "evals/reports" / ("offline-smoke" if args.offline else "online"))
    metadata = {"embedding_dimension": 1024, "thinking": False, "temperature": "provider_default", "code_sha256": _safe_code_hash()}
    if args.offline:
        query, documents = _offline_query, _offline_documents
        answer, mode = (None, "offline_smoke") if args.retrieval_only else (_offline_answer, "offline_smoke")
    else:
        from app.services.dashscope_embeddings import DashScopeEmbeddings
        embeddings = DashScopeEmbeddings()
        query, documents = embeddings.embed_query, embeddings.embed_documents
        answer, mode = (None, "online") if args.retrieval_only else (dashscope_answer, "online")
        metadata["model"] = "configured_runtime_model_name"
    report = await EvaluationRunner(embed_query=query, embed_documents=documents, answer_generator=answer,
        k=args.k, timeout=args.timeout, mode=mode, metadata=metadata).run(samples, corpus)
    if args.ragas:
        from app.evaluation.ragas_runner import evaluate_with_ragas
        report["ragas"] = {name: await evaluate_with_ragas(samples, data["rows"], timeout=args.timeout)
                           for name, data in report["strategies"].items()}
        statuses = [value["status"] for value in report["ragas"].values()]
        if any(status in {"failed", "unavailable"} for status in statuses): report["status"] = "failed"
        elif any(status != "ok" for status in statuses): report["status"] = "partial"
    path = write_report(report, output_dir)
    print(path)
    return 0 if report.get("status") in {"ok", "partial"} else 1


if __name__ == "__main__": raise SystemExit(asyncio.run(main()))
