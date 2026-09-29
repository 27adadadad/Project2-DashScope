"""手动评测使用的 Ragas 适配器；离线注入 scorer 不加载配置或模型依赖。"""

import asyncio
import inspect
import math
from collections.abc import Mapping
from numbers import Real


METRIC_NAMES = ("faithfulness", "llm_context_precision_with_reference", "context_recall")


def build_ragas_payload(samples, strategy_rows):
    rows_by_id = {row["id"]: row for row in strategy_rows}
    payload = []
    for sample in samples:
        if not sample["reference_context_ids"]:
            continue
        row = rows_by_id[sample["id"]]
        payload.append({
            "user_input": sample["question"], "response": row["answer"],
            "reference": sample["reference_answer"],
            "retrieved_contexts": [context["content"] for context in row["contexts"]],
        })
    return payload


def _score_with_ragas(payload, *, timeout):
    """在线路径，在线程中运行；不让 Ragas 默认错误日志输出供应商异常原文。"""
    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall
    from ragas.run_config import RunConfig
    from app.core.config import settings

    # Ragas 0.3 uses `is not None` to enable tenacity logging, even for False.
    run_config = RunConfig(timeout=timeout, max_retries=0, max_workers=3, log_tenacity=None)
    llm = LangchainLLMWrapper(ChatOpenAI(
        model=settings.DASHSCOPE_CHAT_MODEL, api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.DASHSCOPE_COMPATIBLE_BASE_URL, temperature=0,
        timeout=timeout, max_retries=0,
    ), run_config=run_config)
    dataset = EvaluationDataset(samples=[SingleTurnSample(**payload)])
    result = evaluate(
        dataset,
        metrics=[Faithfulness(llm=llm), LLMContextPrecisionWithReference(llm=llm), LLMContextRecall(llm=llm)],
        run_config=run_config, raise_exceptions=True, show_progress=False, allow_nest_asyncio=False,
    )
    # Read individual metric results; never let DataFrame.mean silently drop NaN.
    return result.scores[0] if len(result.scores) == 1 else None


async def _invoke_scorer(scorer, payload):
    if inspect.iscoroutinefunction(scorer):
        return await scorer(payload)
    result = await asyncio.to_thread(scorer, payload)
    return await result if inspect.isawaitable(result) else result


def _excluded_reason(sample, row):
    if not sample.get("reference_context_ids"):
        return "no_reference_context"
    if row is None:
        return "strategy_row_missing"
    generation = row.get("generation_status")
    if generation != "completed":
        return {
            "failed": "generation_failed", "not_run": "generation_not_run",
            "skipped_gate": "generation_skipped_gate", None: "generation_status_missing",
        }.get(generation, "generation_status_invalid")
    retrieval = row.get("retrieval_status")
    if retrieval != "completed":
        return {
            "failed": "retrieval_failed", None: "retrieval_status_missing",
        }.get(retrieval, "retrieval_status_invalid")
    if not isinstance(row.get("answer"), str) or not row["answer"].strip():
        return "answer_empty"
    return None


def _metric_result(value=None, reason=None):
    return {"score": value, "status": "valid" if reason is None else "failed", "reason_code": reason}


def _validate_metric(raw_scores, metric):
    if metric not in raw_scores:
        return _metric_result(reason="metric_missing")
    value = raw_scores[metric]
    if value is None:
        return _metric_result(reason="metric_null")
    if isinstance(value, bool) or not isinstance(value, Real):
        return _metric_result(reason="metric_not_numeric")
    value = float(value)
    if not math.isfinite(value):
        return _metric_result(reason="metric_non_finite")
    if not 0 <= value <= 1:
        return _metric_result(reason="metric_out_of_range")
    return _metric_result(value)


def _mark_failure(entry, reason, error=None, *, status="failed"):
    entry.update(status=status, reason_codes=[reason])
    entry["metrics"] = {name: _metric_result(reason=reason) for name in METRIC_NAMES}
    if error is not None:
        # Do not stringify errors or attach tracebacks: providers can include keys.
        name = type(error).__name__
        entry["error_type"] = name if name.isidentifier() and len(name) <= 80 else "Exception"


def _summarize(per_sample):
    expected = sum(entry["expected"] for entry in per_sample)
    eligible = sum(entry["eligible"] for entry in per_sample)
    valid = sum(entry["status"] == "ok" for entry in per_sample)
    metrics = {}
    for name in METRIC_NAMES:
        values = [entry["metrics"][name]["score"] for entry in per_sample
                  if entry["metrics"][name]["status"] == "valid"]
        metrics[name] = {
            "mean": sum(values) / len(values) if values else None,
            "denominator": len(values), "expected_count": expected,
            "eligible_count": eligible, "valid_count": len(values),
            "failed_count": expected - len(values),
        }
    if expected and valid == expected:
        status = "ok"
    elif any(metric["denominator"] for metric in metrics.values()):
        status = "partial"
    elif eligible and all(entry["status"] == "unavailable" for entry in per_sample if entry["eligible"]):
        status = "unavailable"
    elif any(entry["status"] == "failed" for entry in per_sample):
        status = "failed"
    else:
        status = "not_run"
    return {
        "status": status, "scores": {name: metric["mean"] for name, metric in metrics.items()},
        "counters": {
            "total_count": len(per_sample), "expected_count": expected,
            "eligible_count": eligible, "valid_count": valid,
            "failed_count": expected - valid, "excluded_count": expected - eligible,
        },
        "metrics": metrics, "per_sample": per_sample,
    }


async def evaluate_with_ragas(samples, strategy_rows, *, timeout=120, score_runner=None):
    """逐样本评分，只有三指标全有效且覆盖全部应评样本时返回 ok。

    ``expected_count`` 只统计有 reference_context_ids 的样本；未生成、失败或
    门控跳过的应评样本仍在 expected 中，绝不填 0。每指标的 denominator 只统计
    有效分数。注入 scorer 接受单条 Ragas payload，返回指标名到分数的 mapping。
    ``timeout`` 限制每条样本；取消等待不会强杀同步线程，在线 HTTP/RunConfig 同样
    有超时且无重试。迟到结果不会写入报告。
    """
    if isinstance(timeout, bool) or not isinstance(timeout, Real) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a positive finite number")
    rows_by_id = {row["id"]: row for row in strategy_rows}
    per_sample = []
    scorer = score_runner if score_runner is not None else lambda payload: _score_with_ragas(payload, timeout=timeout)
    for sample in samples:
        row = rows_by_id.get(sample["id"])
        reason = _excluded_reason(sample, row)
        entry = {
            "id": sample["id"], "expected": bool(sample.get("reference_context_ids")),
            "eligible": reason is None, "status": "not_run", "reason_codes": [], "metrics": {},
        }
        per_sample.append(entry)
        if reason is not None:
            status = "not_run" if reason in ("no_reference_context", "generation_not_run", "generation_skipped_gate") else "failed"
            _mark_failure(entry, reason, status=status)
            continue
        try:
            payload = build_ragas_payload([sample], [row])[0]
            raw_scores = await asyncio.wait_for(_invoke_scorer(scorer, payload), timeout=float(timeout))
            if not isinstance(raw_scores, Mapping):
                _mark_failure(entry, "scorer_result_invalid")
                continue
            entry["metrics"] = {name: _validate_metric(raw_scores, name) for name in METRIC_NAMES}
            entry["reason_codes"] = list(dict.fromkeys(
                metric["reason_code"] for metric in entry["metrics"].values() if metric["reason_code"]
            ))
            valid_count = sum(metric["status"] == "valid" for metric in entry["metrics"].values())
            entry["status"] = "ok" if valid_count == len(METRIC_NAMES) else ("partial" if valid_count else "failed")
        except (asyncio.TimeoutError, TimeoutError) as error:
            _mark_failure(entry, "scorer_timeout", error)
        except ImportError as error:
            _mark_failure(entry, "dependency_unavailable", error, status="unavailable")
        except Exception as error:
            _mark_failure(entry, "scorer_exception", error)
    return _summarize(per_sample)
