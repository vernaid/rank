#!/usr/bin/env python
"""
Category baseline: q -> category -> skill.

Leaf skill scoring intentionally matches flat_retrieval_baseline.py:
- dense: BGE cosine/dot over normalized embeddings
- BM25: same Okapi BM25 formula over the full skill corpus statistics
- hybrid: same RRF over dense and BM25 skill ranks

Only the candidate set changes: after predicting one category, skill ranking is
restricted to skills in that category.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from flat_retrieval_baseline import (
    BM25Index,
    build_skill_doc,
    ensure_dir,
    load_benchmark,
    metrics_for,
    primary_category,
    read_jsonl,
    tokenize,
    write_json,
    write_jsonl,
)


def read_category_docs(data_dir: Path, registry: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    cats = sorted({primary_category(s.get("category")) for s in registry})
    by_cat = defaultdict(list)
    for s in registry:
        by_cat[primary_category(s.get("category"))].append(s)

    docs = []
    for cat in cats:
        slug = cat.lower().replace("&", "").replace("/", "").replace(",", "").replace(" ", "-")
        readme = data_dir / "categories" / slug / "README.md"
        parts = [f"category: {cat}"]
        if readme.exists():
            parts.append(readme.read_text(encoding="utf-8", errors="replace"))
        for skill in by_cat[cat]:
            parts.append(f"{skill.get('name','')} {skill.get('description','')}")
        docs.append("\n".join(parts))
    return cats, docs


def encode(model_name: str, texts: list[str], batch_size: int, device: str, is_query: bool) -> np.ndarray:
    model = SentenceTransformer(model_name, device=device, local_files_only=True)
    if "bge" in model_name.lower() and is_query:
        texts = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32", copy=False)


def top_indices(scores: np.ndarray, candidates: list[int], top_k: int) -> list[int]:
    ranked = sorted(candidates, key=lambda i: (-float(scores[i]), i))
    return ranked[:top_k]


def bm25_score_vector(index: BM25Index, query: str) -> np.ndarray:
    scores = np.zeros(index.n_docs, dtype=np.float32)
    q_terms = Counter(tokenize(query))
    for term, qtf in q_terms.items():
        postings = index.inverted.get(term)
        if not postings:
            continue
        idf = index.idf[term]
        for doc_id, tf in postings:
            denom = tf + index.k1 * (1.0 - index.b + index.b * index.doc_len[doc_id] / max(index.avgdl, 1e-9))
            scores[doc_id] += qtf * idf * (tf * (index.k1 + 1.0)) / denom
    return scores


def bm25_matrix(index: BM25Index, queries: list[dict[str, Any]]) -> np.ndarray:
    return np.vstack([bm25_score_vector(index, q["query"]) for q in queries])


def dense_leaf_rows(
    queries: list[dict[str, Any]],
    q_skill_scores: np.ndarray,
    skill_ids: list[str],
    cat_to_doc_ids: dict[str, list[int]],
    predicted_categories: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    rows = []
    for qi, q in enumerate(queries):
        candidates = cat_to_doc_ids[predicted_categories[qi]]
        idx = top_indices(q_skill_scores[qi], candidates, top_k)
        rows.append(
            {
                "query_id": q["query_id"],
                "predicted_category": predicted_categories[qi],
                "ranking": [
                    {"rank": r, "skill_id": skill_ids[j], "score": float(q_skill_scores[qi, j])}
                    for r, j in enumerate(idx, 1)
                ],
            }
        )
    return rows


def rrf_leaf_rows(
    queries: list[dict[str, Any]],
    dense_scores: np.ndarray,
    bm25_scores: np.ndarray,
    skill_ids: list[str],
    cat_to_doc_ids: dict[str, list[int]],
    predicted_categories: list[str],
    top_k: int,
    rrf_k: int,
) -> list[dict[str, Any]]:
    rows = []
    for qi, q in enumerate(queries):
        candidates = cat_to_doc_ids[predicted_categories[qi]]
        dense_ranked = top_indices(dense_scores[qi], candidates, len(candidates))
        bm25_ranked = top_indices(bm25_scores[qi], candidates, len(candidates))
        scores = defaultdict(float)
        sources = defaultdict(dict)
        for name, ranked in [("dense", dense_ranked), ("bm25", bm25_ranked)]:
            for rank, doc_id in enumerate(ranked, 1):
                sid = skill_ids[doc_id]
                scores[sid] += 1.0 / (rrf_k + rank)
                sources[sid][name] = rank
        ranked_sids = sorted(scores, key=lambda sid: (-scores[sid], sid))[:top_k]
        rows.append(
            {
                "query_id": q["query_id"],
                "predicted_category": predicted_categories[qi],
                "ranking": [
                    {
                        "rank": r,
                        "skill_id": sid,
                        "score": float(scores[sid]),
                        "dense_rank": sources[sid].get("dense"),
                        "bm25_rank": sources[sid].get("bm25"),
                    }
                    for r, sid in enumerate(ranked_sids, 1)
                ],
            }
        )
    return rows


def annotate(rows: list[dict[str, Any]], method: str, qrels: dict[str, str], gold_cat_by_qid: dict[str, str]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        gold = qrels[row["query_id"]]
        gold_rank = None
        for item in row["ranking"]:
            if item["skill_id"] == gold:
                gold_rank = item["rank"]
                break
        out.append(
            {
                "query_id": row["query_id"],
                "method": method,
                "gold_skill_id": gold,
                "gold_category": gold_cat_by_qid[row["query_id"]],
                "predicted_category": row["predicted_category"],
                "category_correct": row["predicted_category"] == gold_cat_by_qid[row["query_id"]],
                "gold_rank": gold_rank,
                "ranking": row["ranking"],
            }
        )
    return out


def category_accuracy(pred: list[str], queries: list[dict[str, Any]], gold_cat_by_qid: dict[str, str]) -> float:
    return round(sum(1 for p, q in zip(pred, queries) if p == gold_cat_by_qid[q["query_id"]]) / max(1, len(queries)), 6)


def run(args: argparse.Namespace) -> None:
    benchmark_dir = Path(args.benchmark_dir)
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    ensure_dir(results_dir)

    queries, qrels, registry = load_benchmark(benchmark_dir)
    metadata = read_jsonl(benchmark_dir / "query_metadata.jsonl")
    gold_cat_by_qid = {m["query_id"]: m["gold_category"] for m in metadata}

    skill_ids = [s["skill_id"] for s in registry]
    skill_docs = [build_skill_doc(s, args.max_skill_chars) for s in registry]
    cat_to_doc_ids = defaultdict(list)
    for i, s in enumerate(registry):
        cat_to_doc_ids[primary_category(s.get("category"))].append(i)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)

    categories, category_docs = read_category_docs(data_dir, registry)
    query_texts = [q["query"] for q in queries]

    skill_emb = encode(args.model, skill_docs, args.batch_size, device, is_query=False)
    query_emb = encode(args.model, query_texts, args.batch_size, device, is_query=True)
    category_emb = encode(args.model, category_docs, args.batch_size, device, is_query=False)

    dense_skill_scores = np.matmul(query_emb, skill_emb.T)
    dense_cat_scores = np.matmul(query_emb, category_emb.T)
    dense_pred = [categories[int(np.argmax(dense_cat_scores[i]))] for i in range(len(queries))]

    skill_bm25 = BM25Index(skill_docs, k1=args.bm25_k1, b=args.bm25_b)
    category_bm25 = BM25Index(category_docs, k1=args.bm25_k1, b=args.bm25_b)
    bm25_skill_scores = bm25_matrix(skill_bm25, queries)
    bm25_cat_scores = bm25_matrix(category_bm25, queries)
    bm25_pred = [categories[int(np.argmax(bm25_cat_scores[i]))] for i in range(len(queries))]

    hybrid_pred = []
    for i in range(len(queries)):
        dense_order = np.argsort(-dense_cat_scores[i])
        bm25_order = np.argsort(-bm25_cat_scores[i])
        scores = defaultdict(float)
        for order in [dense_order, bm25_order]:
            for rank, cat_idx in enumerate(order, 1):
                scores[categories[int(cat_idx)]] += 1.0 / (args.rrf_k + rank)
        hybrid_pred.append(sorted(scores, key=lambda c: (-scores[c], c))[0])

    dense_rows = dense_leaf_rows(queries, dense_skill_scores, skill_ids, cat_to_doc_ids, dense_pred, args.ranking_depth)
    bm25_rows = dense_leaf_rows(queries, bm25_skill_scores, skill_ids, cat_to_doc_ids, bm25_pred, args.ranking_depth)
    hybrid_rows = rrf_leaf_rows(
        queries, dense_skill_scores, bm25_skill_scores, skill_ids, cat_to_doc_ids, hybrid_pred, args.ranking_depth, args.rrf_k
    )

    dense_annotated = annotate(dense_rows, "category_dense_bge", qrels, gold_cat_by_qid)
    bm25_annotated = annotate(bm25_rows, "category_bm25", qrels, gold_cat_by_qid)
    hybrid_annotated = annotate(hybrid_rows, "category_dense_bm25_rrf", qrels, gold_cat_by_qid)

    write_jsonl(results_dir / "category_dense.jsonl", dense_annotated)
    write_jsonl(results_dir / "category_bm25.jsonl", bm25_annotated)
    write_jsonl(results_dir / "category_hybrid.jsonl", hybrid_annotated)

    metrics = {
        "benchmark_dir": str(benchmark_dir),
        "model": args.model,
        "device": device,
        "ranking_depth": args.ranking_depth,
        "topk_evaluated": [1, 5, 10, 20],
        "bm25": {"k1": args.bm25_k1, "b": args.bm25_b},
        "rrf": {"k": args.rrf_k},
        "doc_count": len(registry),
        "category_count": len(categories),
        "query_count": len(queries),
        "metrics": {
            "category_dense_bge": {
                **metrics_for(dense_annotated, qrels),
                "CategoryAcc@1": category_accuracy(dense_pred, queries, gold_cat_by_qid),
            },
            "category_bm25": {
                **metrics_for(bm25_annotated, qrels),
                "CategoryAcc@1": category_accuracy(bm25_pred, queries, gold_cat_by_qid),
            },
            "category_dense_bm25_rrf": {
                **metrics_for(hybrid_annotated, qrels),
                "CategoryAcc@1": category_accuracy(hybrid_pred, queries, gold_cat_by_qid),
            },
        },
    }
    write_json(results_dir / "category_metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", default="benchmark")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ranking-depth", type=int, default=100)
    parser.add_argument("--max-skill-chars", type=int, default=6000)
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--rrf-k", type=int, default=60)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
