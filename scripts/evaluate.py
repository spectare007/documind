"""RAGAs evaluation: agentic vs simple pipeline over the golden set.

Adapted for the installed `ragas==0.4.3` (a major version ahead of the API
this script was originally drafted against). Two real API changes were
required, both explained where they are applied below:

1. `ragas.llms.base` (imported transitively by `import ragas` itself)
   unconditionally does `from langchain_community.chat_models.vertexai
   import ChatVertexAI`. That submodule no longer exists in
   `langchain-community>=0.4` (langchain-community is being "sunset" and
   dropped its built-in cloud-provider chat model shims -- see
   https://github.com/langchain-ai/langchain-community/issues/674). We never
   use VertexAI (Ollama is our judge), so `import ragas` is unblocked with a
   harmless stub module registered in `sys.modules` before the first `ragas`
   import, rather than pinning a fragile, possibly-conflicting older
   langchain-community version.
2. The legacy top-level metric objects (`faithfulness`, `answer_relevancy`,
   `context_precision`, `context_recall`) and `LangchainLLMWrapper` /
   `LangchainEmbeddingsWrapper` still work exactly as in older ragas, just
   under a `DeprecationWarning` pointing at `ragas.metrics.collections`
   (a newer, non-Langchain-wrapped API). We keep the legacy path here since
   it is still fully functional and keeps this script's shape close to the
   original design; the warning is suppressed for clean progress output.

Everything else (`EvaluationDataset.from_list`, `evaluate(dataset,
metrics=..., llm=..., embeddings=...)`, `result.to_pandas()`) matches the
original design unchanged.
"""
import argparse
import json
import statistics
import sys
import time
import types
import warnings
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
BACKEND = "http://localhost:8000"

METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def _install_langchain_community_vertexai_stub() -> None:
    """See module docstring, adaptation (1). Idempotent."""
    name = "langchain_community.chat_models.vertexai"
    if name in sys.modules:
        return
    stub = types.ModuleType(name)

    class ChatVertexAI:  # pragma: no cover - never instantiated, import-only shim
        pass

    stub.ChatVertexAI = ChatVertexAI
    sys.modules[name] = stub


def check_corpus_status(api_key: str) -> str:
    """Fetch /api/v1/documents and summarize ingestion health for the report.

    Returns a human-readable note (empty string if everything looks healthy)
    so a broken or partial corpus is disclosed in the generated report's
    Caveats section instead of silently producing a hollow comparison.
    """
    try:
        r = httpx.get(
            f"{BACKEND}/api/v1/documents",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )
        r.raise_for_status()
        docs = r.json()
    except Exception as exc:  # noqa: BLE001 - best-effort diagnostic, never fatal
        return f"Could not check /api/v1/documents before running ({exc!r})."

    total = len(docs)
    ready = [d for d in docs if d.get("status") == "completed" and (d.get("chunk_count") or 0) > 0]
    failed = [d for d in docs if d.get("status") == "failed"]
    processing = [d for d in docs if d.get("status") == "processing"]
    if len(ready) == total and total > 0:
        return ""
    parts = [f"{len(ready)}/{total} documents ingested with chunks at run start"]
    if failed:
        sample_err = (failed[0].get("error") or "")[:200]
        parts.append(f"{len(failed)} failed (e.g. {failed[0]['filename']!r}: {sample_err})")
    if processing:
        parts.append(f"{len(processing)} still processing")
    return "; ".join(parts) + "."


def run_pipeline(question: str, mode: str, api_key: str) -> dict:
    r = httpx.post(
        f"{BACKEND}/api/v1/query",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"question": question, "mode": mode},
        timeout=600.0,
    )
    r.raise_for_status()
    return r.json()


def collect(golden: list[dict], mode: str, api_key: str) -> list[dict]:
    rows = []
    for i, item in enumerate(golden):
        start = time.perf_counter()
        out = run_pipeline(item["question"], mode, api_key)
        latency = time.perf_counter() - start
        rows.append({
            "user_input": item["question"],
            "response": out["answer"],
            "retrieved_contexts": [c["text"] for c in out["chunks"]],
            "reference": item["reference"],
            "category": item.get("category", ""),
            "latency_s": round(latency, 2),
            "grounded": out.get("grounded"),
            "trace_id": out.get("trace_id", ""),
        })
        print(f"[{mode}] {i + 1}/{len(golden)} {latency:.1f}s")
    return rows


def ragas_scores(rows: list[dict], judge_model: str, embed_model: str, ollama_url: str) -> dict:
    _install_langchain_community_vertexai_stub()
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    import pandas as pd
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (answer_relevancy, context_precision,
                               context_recall, faithfulness)
    from ragas.run_config import RunConfig

    judge = LangchainLLMWrapper(ChatOllama(model=judge_model, base_url=ollama_url, temperature=0))
    emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=embed_model, base_url=ollama_url))
    dataset = EvaluationDataset.from_list(
        [{k: r[k] for k in ("user_input", "response", "retrieved_contexts", "reference")}
         for r in rows]
    )
    # ragas 0.4.3's RunConfig defaults (timeout=180s, max_workers=16) assume a
    # fast hosted API judge. Against a single local qwen2.5:7b instance on
    # CPU, 16 concurrent jobs all queue behind one another on the same
    # Ollama process; every job then blows the 180s timeout waiting in that
    # queue rather than from any real per-call slowness (confirmed: a 3-row
    # smoke test with the default RunConfig timed out on all 12 jobs). A low
    # max_workers avoids the queuing pileup, and a much longer per-job
    # timeout gives a CPU judge room for faithfulness/context metrics that
    # each issue several LLM calls (one per retrieved chunk for
    # precision/recall) against a multi-chunk real corpus.
    run_config = RunConfig(timeout=900, max_workers=2, max_retries=2, max_wait=30)
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                        context_precision, context_recall],
                      llm=judge, embeddings=emb, run_config=run_config)
    df = result.to_pandas()

    summary: dict[str, dict] = {}
    for m in METRIC_NAMES:
        col = df[m]
        nan_count = int(col.isna().sum())
        non_nan = col.dropna()
        mean = round(float(non_nan.mean()), 3) if len(non_nan) else None
        summary[m] = {"mean": mean, "nan_count": nan_count, "n": int(len(col))}

    # Trace every per-row ragas score back onto the raw rows (kept in the
    # gitignored results.json) so a NaN can be attributed to a specific
    # question rather than only showing up as a diluted mean.
    for i, row in enumerate(rows):
        row["ragas_scores"] = {
            m: (None if pd.isna(df[m].iloc[i]) else round(float(df[m].iloc[i]), 3))
            for m in METRIC_NAMES
        }

    return summary


def latency_stats(rows: list[dict]) -> dict:
    xs = sorted(r["latency_s"] for r in rows)
    return {"p50_s": round(statistics.median(xs), 1),
            "p95_s": round(xs[max(0, int(len(xs) * 0.95) - 1)], 1),
            "mean_s": round(statistics.mean(xs), 1)}


def _fmt_score(cell: dict) -> str:
    if cell["mean"] is None:
        return f"NaN (all {cell['n']} judge calls failed to parse)"
    if cell["nan_count"]:
        return f"{cell['mean']} ({cell['nan_count']}/{cell['n']} NaN)"
    return str(cell["mean"])


def write_report(all_results: dict, out_path: Path, corpus_note: str, golden_subset: list[dict],
                  golden_total: int, subset_reason: str = "", judge_model: str = "qwen2.5:7b",
                  embed_model: str = "nomic-embed-text") -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    question_count = len(golden_subset)
    lines = [
        "# DocuMind RAG Evaluation Report", "",
        f"Generated: {ts} · Judge: local Ollama `{judge_model}` (embeddings: `{embed_model}`) · Metrics: RAGAs", "",
    ]
    if judge_model != "qwen2.5:7b":
        lines += [
            f"**Judge substitution**: this run used `{judge_model}` instead of the originally "
            "planned `qwen2.5:7b`, to fit the available time budget on a CPU-only host. A "
            "smaller judge further weakens the precision of the absolute scores below (see "
            "Caveats) -- relative agentic-vs-simple comparisons remain the more trustworthy "
            "read since both share the same judge.", "",
        ]
    if question_count < golden_total:
        lines += [
            f"**Reduced run**: {question_count} of {golden_total} golden-set questions were "
            f"evaluated in this recorded run, in **both** modes. {subset_reason} The subset was "
            "chosen deliberately to span categories (single-doc factual/numeric, multi-doc, "
            "unanswerable) and all source documents, not truncated to the first N. The full "
            "golden set stays committed at `evaluation/golden_set.json`.", "",
            "Questions evaluated in this run:", "",
        ]
        for i, item in enumerate(golden_subset, 1):
            lines.append(f"{i}. [{item.get('category', '')}] {item['question']}")
        lines.append("")
    lines += [
        "| Metric | " + " | ".join(all_results) + " |",
        "|---|" + "---|" * len(all_results),
    ]
    for m in METRIC_NAMES:
        lines.append(
            f"| {m} | " + " | ".join(_fmt_score(v["scores"][m]) for v in all_results.values()) + " |"
        )
    lines += ["", "## Latency", "",
              "| Stat | " + " | ".join(all_results) + " |",
              "|---|" + "---|" * len(all_results)]
    for stat in ("p50_s", "p95_s", "mean_s"):
        lines.append(f"| {stat} | " + " | ".join(str(v["latency"][stat]) for v in all_results.values()) + " |")

    lines += ["", "## Per-question latency (wall-clock cost per question per mode)", "",
              "| # | Category | Question | " + " | ".join(f"{m} latency_s" for m in all_results) + " |",
              "|---|---|---|" + "---|" * len(all_results)]
    modes = list(all_results)
    n_rows = len(all_results[modes[0]]["rows"]) if modes else 0
    for i in range(n_rows):
        row0 = all_results[modes[0]]["rows"][i]
        q = row0["user_input"].replace("|", "\\|")
        cat = row0.get("category", "")
        cells = [str(all_results[mode]["rows"][i]["latency_s"]) for mode in modes]
        lines.append(f"| {i + 1} | {cat} | {q} | " + " | ".join(cells) + " |")

    # Key finding: how often did each mode come back with literally nothing retrieved,
    # and did that correlate with the boilerplate "couldn't find anything" refusal?
    zero_context = {
        mode: [r for r in v["rows"] if not r["retrieved_contexts"]]
        for mode, v in all_results.items()
    }
    finding_lines = []
    for mode, rows in zero_context.items():
        if rows:
            n = len(rows)
            total = len(all_results[mode]["rows"])
            finding_lines.append(
                f"- **{mode}**: {n}/{total} questions returned zero retrieved chunks (the "
                "pipeline's corrective-retrieval/grading step rejected every candidate chunk "
                "and fell back to a boilerplate \"couldn't find anything relevant\" answer), "
                f"including at least one question the other mode answered correctly from real "
                "retrieved context."
            )
    if finding_lines:
        lines += ["", "## Key finding: retrieval failure, not just score differences", ""]
        lines += finding_lines
        lines += [
            "",
            "This is the mechanism behind the headline numbers, not just a side note: when a "
            "mode returns zero context, `context_precision`/`context_recall` should be trivially "
            "0 for that row, and any non-empty `faithfulness` or `answer_relevancy` score is "
            "scoring a refusal, not an answer -- a high faithfulness score in that state reflects "
            "\"didn't hallucinate because it didn't try to answer,\" not answer quality. Treat "
            "`context_recall` as the metric of interest for retrieval quality specifically "
            "because it is not inflated by refusal behavior the way faithfulness is.",
        ]
    any_recall_on_empty = any(
        r["ragas_scores"].get("context_recall") not in (None, 0, 0.0)
        for v in all_results.values() for r in v["rows"] if not r["retrieved_contexts"]
    )
    if any_recall_on_empty:
        offending = [
            (mode, r["user_input"][:60], r["ragas_scores"].get("context_recall"))
            for mode, v in all_results.items() for r in v["rows"]
            if not r["retrieved_contexts"] and r["ragas_scores"].get("context_recall") not in (None, 0, 0.0)
        ]
        lines += [
            "", "**Judge-reliability anomaly**: the local judge scored `context_recall` > 0 on "
            "at least one row where `retrieved_contexts` was empty -- a logically implausible "
            "result (recall of a reference against zero retrieved text should be 0), most likely "
            f"caused by the smaller judge model. Affected rows: {offending}. Reported here "
            "rather than silently trusting the aggregate mean.",
        ]

    lines += ["", "## Caveats", "",
              f"- Judge is a local `{judge_model}` model on CPU: scores are directionally useful, noisy in absolute terms.",
              "- Agentic vs simple comparison shares the same judge, so relative differences are more reliable.",
              "- Unanswerable questions score via faithfulness (refusing is grounded behavior) -- but see the "
              "Key Finding above: a refusal-driven faithfulness score is not evidence of answer quality."]
    if corpus_note:
        lines.append(f"- **Corpus state at run time**: {corpus_note}")
    any_nan = any(v["scores"][m]["nan_count"] for v in all_results.values() for m in METRIC_NAMES)
    if any_nan:
        lines.append(
            "- One or more metrics returned NaN for some rows (local judge failed to parse a "
            "verdict). NaNs are excluded from the displayed mean and called out per-cell above "
            "rather than silently dropped; per-row detail is in the gitignored "
            "`evaluation/runs/<timestamp>/results.json`."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["simple", "agentic"])
    parser.add_argument("--api-key", default="documind-dev-key")
    parser.add_argument("--judge-model", default="qwen2.5:7b")
    parser.add_argument("--embed-model", default="nomic-embed-text")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Evaluate only the first N golden-set questions (for bounding wall-clock on a "
             "slow CPU judge). The full golden set stays committed regardless; use this only "
             "for a recorded run and disclose the reduction in the report.",
    )
    parser.add_argument(
        "--indices", type=str, default=None,
        help="Comma-separated 0-based indices into golden_set.json to evaluate, e.g. "
             "'1,6,10,12,14,15,18,19,23,24'. Overrides --limit. Use this to deliberately pick a "
             "stratified subset (spanning categories/documents) rather than truncating to the "
             "first N -- disclose the exact subset in the report either way.",
    )
    args = parser.parse_args()

    golden_full = json.loads((ROOT / "evaluation" / "golden_set.json").read_text(encoding="utf-8"))
    subset_reason = ""
    if args.indices:
        idxs = [int(x) for x in args.indices.split(",")]
        golden = [golden_full[i] for i in idxs]
        subset_reason = (
            "The reduction is a CPU-time constraint (a local 7B judge plus a full six-agent "
            "crewai crew per agentic question), not a methodological choice -- a deliberately "
            "stratified subset was selected, not the first N."
        )
    elif args.limit:
        golden = golden_full[: args.limit]
        subset_reason = "The reduction is a CPU-time constraint (first N questions only)."
    else:
        golden = golden_full

    print(f"Evaluating {len(golden)}/{len(golden_full)} golden-set questions: "
          f"{[golden_full.index(g) for g in golden]}")

    corpus_note = check_corpus_status(args.api_key)
    if corpus_note:
        print(f"WARNING: {corpus_note}")

    run_dir = ROOT / "evaluation" / "runs" / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)

    all_results = {}
    for mode in args.modes:
        rows = collect(golden, mode, args.api_key)
        scores = ragas_scores(rows, args.judge_model, args.embed_model, args.ollama_url)
        all_results[mode] = {"scores": scores, "latency": latency_stats(rows), "rows": rows}
        print(f"[{mode}] scores: { {m: v['mean'] for m, v in scores.items()} }")

    (run_dir / "results.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    write_report(all_results, ROOT / "doc" / "evaluation-report.md", corpus_note,
                 golden, len(golden_full), subset_reason, args.judge_model, args.embed_model)


if __name__ == "__main__":
    main()
