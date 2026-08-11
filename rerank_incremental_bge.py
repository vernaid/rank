#!/usr/bin/env python
"""Incrementally score a new Top-20 with BGE, reusing prior query-skill scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import CrossEncoder

from improved_retrieval_experiments import build_fields, read_jsonl, stratified_split, write_json, write_jsonl
from rerank_top20_bge import metrics_from_rows


def run(args: argparse.Namespace) -> None:
    benchmark = Path(args.benchmark_dir)
    results = Path(args.results_dir)
    queries = read_jsonl(benchmark / "queries.jsonl")
    metadata = read_jsonl(benchmark / "query_metadata.jsonl")
    registry = read_jsonl(benchmark / "intermediate" / "skill_registry.jsonl")
    source = read_jsonl(results / args.input_ranking)
    prior = read_jsonl(results / args.prior_reranking)
    query_text = {row["query_id"]: row["query"] for row in queries}
    documents = {row["skill_id"]: build_fields(row)["summary"] for row in registry}

    score_cache: dict[tuple[str, str], float] = {}
    for row in prior:
        for item in row["ranking"]:
            value = item.get("reranker_score")
            if value is not None:
                score_cache[(row["query_id"], item["skill_id"])] = float(value)
    persistent_path = results / "cache_improved" / args.score_cache
    if persistent_path.exists():
        for row in read_jsonl(persistent_path):
            score_cache[(row["query_id"], row["skill_id"])] = float(row["score"])

    missing_keys: list[tuple[str, str]] = []
    pairs: list[tuple[str, str]] = []
    for row in source:
        qid = row["query_id"]
        for item in row["ranking"][: args.depth]:
            key = (qid, item["skill_id"])
            if key not in score_cache:
                missing_keys.append(key)
                pairs.append((query_text[qid], documents[item["skill_id"]]))

    if pairs:
        model = CrossEncoder(args.model, device=args.device, local_files_only=True, max_length=args.max_length)
        values = np.asarray(model.predict(pairs, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True)).reshape(-1)
        for key, value in zip(missing_keys, values):
            score_cache[key] = float(value)
        write_jsonl(persistent_path, [
            {"query_id": qid, "skill_id": sid, "score": score_cache[(qid, sid)]}
            for qid, sid in sorted(score_cache)
        ])

    output_rows: list[dict[str, Any]] = []
    for row in source:
        qid = row["query_id"]
        head = [dict(item) for item in row["ranking"][: args.depth]]
        tail = [dict(item) for item in row["ranking"][args.depth :]]
        for item in head:
            item["retrieval_rank"] = int(item["rank"])
            item["retrieval_score"] = item.get("score")
            item["reranker_score"] = score_cache[(qid, item["skill_id"])]
        head.sort(key=lambda item: (-float(item["reranker_score"]), int(item["retrieval_rank"])))
        ranking = head + tail
        for rank, item in enumerate(ranking, 1):
            item["rank"] = rank
        gold = row["gold_skill_id"]
        gold_rank = next((item["rank"] for item in ranking if item["skill_id"] == gold), None)
        output_rows.append({"query_id": qid, "method": "toprank_tuned_bge_cross_encoder", "gold_skill_id": gold, "gold_rank": gold_rank, "ranking": ranking})

    dev, test = stratified_split(metadata)
    metrics = {
        "model": args.model,
        "depth": args.depth,
        "document": "summary",
        "max_length": args.max_length,
        "cached_before": len(score_cache) - len(missing_keys),
        "new_pairs_scored": len(missing_keys),
        "before": {"dev": metrics_from_rows(source, dev), "test": metrics_from_rows(source, test), "all": metrics_from_rows(source)},
        "after": {"dev": metrics_from_rows(output_rows, dev), "test": metrics_from_rows(output_rows, test), "all": metrics_from_rows(output_rows)},
    }
    write_json(results / args.metrics_output, metrics)
    write_jsonl(results / args.ranking_output, output_rows)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", default="benchmark")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--input-ranking", default="toprank_tuned_hybrid_top500.jsonl")
    parser.add_argument("--prior-reranking", default="optimized_flat_bge_reranker.jsonl")
    parser.add_argument("--score-cache", default="bge_reranker_score_cache.jsonl")
    parser.add_argument("--model", default="BAAI/bge-reranker-base")
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ranking-output", default="toprank_bge_reranker_top500.jsonl")
    parser.add_argument("--metrics-output", default="toprank_bge_reranker_metrics.json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
