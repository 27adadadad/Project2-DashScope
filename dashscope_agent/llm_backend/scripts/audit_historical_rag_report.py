"""对旧 RAG 报告做只读重评分，不重新检索、生成或调用 Ragas。"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.evaluation.retrieval_metrics import calculate_retrieval_metrics

BACKEND = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = BACKEND / "evals/datasets/rag_eval.jsonl"
DEFAULT_CORPUS = BACKEND / "evals/datasets/shared_corpus.jsonl"
DEFAULT_REPORT = BACKEND / "evals/reports/rag-eval-20260814-090731.json"
REVIEW_NOTES = [
    {"sample_id": "relation-01", "reason": "g01 appears to cover the answer alone; p01 may be redundant."},
    {"sample_id": "relation-16", "reason": "o09 and s08 do not explicitly establish the claimed verify-then-view dependency."},
    {"sample_id": "relation-19", "reason": "g03 says H1 speaker while g07 says H1 hub; identity equivalence is not explicit."},
]


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate(samples, corpus, old):
    if old.get("sample_count") != len(samples) or old.get("corpus_count") != len(corpus):
        raise ValueError("count_mismatch")
    sample_ids = [x["id"] for x in samples]
    corpus_ids = [x["id"] for x in corpus]
    if len(set(sample_ids)) != len(sample_ids) or len(set(corpus_ids)) != len(corpus_ids):
        raise ValueError("duplicate_id")
    by_id = {x["id"]: x["text"] for x in corpus}
    for sample in samples:
        ids, texts = sample["reference_context_ids"], sample["reference_contexts"]
        if len(ids) != len(set(ids)) or any(i not in by_id for i in ids) or [by_id[i] for i in ids] != texts:
            raise ValueError("reference_mismatch")
    expected = set(sample_ids)
    for name, strategy in old.get("strategies", {}).items():
        rows = strategy.get("rows", [])
        ids = [x.get("id") for x in rows]
        if set(ids) != expected or len(ids) != len(set(ids)):
            raise ValueError(f"historical_rows_mismatch:{name}")
        for row in rows:
            if len(row.get("retrieved_ids", [])) != len(row.get("contexts", [])):
                raise ValueError(f"retrieved_context_mismatch:{name}:{row.get('id')}")
            if not isinstance(row.get("retrieved_ids", []), list):
                raise ValueError(f"retrieved_count_mismatch:{name}:{row.get('id')}")
            if len(row.get("retrieved_ids", [])) != len(set(row.get("retrieved_ids", []))):
                raise ValueError(f"duplicate_retrieved:{name}:{row.get('id')}")
            for rid, context in zip(row["retrieved_ids"], row["contexts"]):
                if rid not in by_id or context.get("id") != rid or context.get("content") != by_id[rid]:
                    raise ValueError(f"retrieved_text_mismatch:{name}:{row.get('id')}")


def _strategy(samples, old_rows, old_summary=None):
    samples_by_id = {s["id"]: s for s in samples}
    rows = []
    for old in old_rows:
        sample = samples_by_id[old["id"]]
        metric = calculate_retrieval_metrics(old["retrieved_ids"], sample["reference_context_ids"], k=5) if sample["reference_context_ids"] else None
        rows.append({"id": old["id"], "category": sample["category"], "metrics": metric,
                     "retrieved_ids": old["retrieved_ids"], "legacy_gate_allowed": old.get("evidence_sufficient")})
    answerable = [r for r in rows if r["metrics"] is not None]
    summary = {key: (mean(r["metrics"][key] for r in answerable) if answerable else None)
               for key in ("recall_at_k", "hit_at_k", "all_evidence_at_k", "mrr", "id_precision_at_k")}
    summary["answerable_count"] = len(answerable)
    summary["unanswerable_count"] = len(rows) - len(answerable)
    return {"summary": summary, "by_category": {
        category: {key: (mean(r["metrics"][key] for r in group) if group else None)
                   for key in ("recall_at_k", "hit_at_k", "all_evidence_at_k", "mrr", "id_precision_at_k")}
        for category in sorted({r["category"] for r in rows})
        for group in [[x for x in answerable if x["category"] == category]]
    }, "missing_multi_evidence_ids": [r["id"] for r in answerable if r["metrics"]["all_evidence_at_k"] < 1],
        "legacy_gate_proxy": {"source_false_answer_rate": (old_summary or {}).get("false_answer_rate"),
                               "is_answer_accuracy": False},
        "model_actual_error_rate": None, "rows": rows}


def audit_report(dataset_path=DEFAULT_DATASET, corpus_path=DEFAULT_CORPUS, report_path=DEFAULT_REPORT):
    samples, corpus = _read_jsonl(dataset_path), _read_jsonl(corpus_path)
    old = json.loads(Path(report_path).read_text(encoding="utf-8"))
    _validate(samples, corpus, old)
    strategies = {name: _strategy(samples, data["rows"], data) for name, data in old["strategies"].items()}
    dense = {x["id"]: x for x in strategies["dense_baseline"]["rows"]}
    hybrid = {x["id"]: x for x in strategies["hybrid_rrf"]["rows"]}
    answerable = [s["id"] for s in samples if s["reference_context_ids"]]
    comparison = {"overall": {}}
    for metric in ("recall_at_k", "hit_at_k", "all_evidence_at_k", "mrr", "id_precision_at_k"):
        delta = {sid: hybrid[sid]["metrics"][metric] - dense[sid]["metrics"][metric] for sid in answerable}
        comparison["overall"][metric] = {"improved_ids": [i for i in answerable if delta[i] > 0],
            "regressed_ids": [i for i in answerable if delta[i] < 0], "equal_count": sum(v == 0 for v in delta.values()),
            "delta": mean(delta.values()) if delta else None}
    old_ragas = old.get("ragas", {})
    ragas = {}
    for name, value in old_ragas.items():
        scores = {k: (float(v) if _finite(v) else None) for k, v in value.get("scores", {}).items()}
        ragas[name] = {"source_status": value.get("status"), "status": "ok" if scores and all(v is not None for v in scores.values()) else "failed", "scores": scores}
    return {"schema_version": 1, "mode": "historical_rescore", "online_evaluation": "not_run",
        "historical_versions": {"generation_model": None, "embedding_model": None},
        "sources": {"dataset": {"path": str(dataset_path), "sha256": _sha(dataset_path)}, "corpus": {"path": str(corpus_path), "sha256": _sha(corpus_path)}, "report": {"path": str(report_path), "sha256": _sha(report_path)}},
        "integrity": {"sample_count": len(samples), "corpus_count": len(corpus), "category_counts": {c: sum(s["category"] == c for s in samples) for c in sorted({s["category"] for s in samples})}},
        "strategies": strategies, "comparison": comparison, "ragas_validity": ragas, "annotation_review": REVIEW_NOTES,
        "limitations": ["This is a read-only re-score of old retrieved_ids; no retrieval, embedding, generation, or Ragas run.", "Legacy false_answer_rate measured gate proxy only, not model factual error." ]}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-dir", type=Path, default=BACKEND / "evals/reports/historical-audit")
    args = parser.parse_args(argv)
    result = audit_report(args.dataset, args.corpus, args.report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "historical-rescore.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(main())
