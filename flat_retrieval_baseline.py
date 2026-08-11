#!/usr/bin/env python
"""
Flat retrieval baselines for the Agent Skills benchmark.

Outputs full Top-100 rankings for:
- dense BGE retrieval
- BM25
- Dense+BM25 RRF hybrid

The JSONL ranking rows intentionally keep complete ranked candidates because
later diagnostics depend on per-query rankings, not only aggregate metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


TOKEN_RE = re.compile(r"[a-zA-Z0-9_+#.-]+")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def primary_category(category: Any) -> str:
    if isinstance(category, list) and category:
        return str(category[0])
    if isinstance(category, str) and category:
        return category
    return "Uncategorized"


def build_skill_doc(skill: dict[str, Any], max_skill_chars: int) -> str:
    parts = [
        f"name: {skill.get('name', '')}",
        f"slug: {skill.get('slug', '')}",
        f"category: {primary_category(skill.get('category'))}",
        f"description: {skill.get('description', '')}",
    ]
    framework = skill.get("framework")
    if framework:
        parts.append(f"framework: {framework}")
    skill_text = skill.get("skill_text", "")
    if max_skill_chars > 0:
        skill_text = skill_text[:max_skill_chars]
    if skill_text:
        parts.append(f"documentation: {skill_text}")
    return "\n".join(parts)


class BM25Index:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_len: list[int] = []
        self.inverted: dict[str, list[tuple[int, int]]] = defaultdict(list)
        df: Counter[str] = Counter()
        for doc_id, doc in enumerate(docs):
            counts = Counter(tokenize(doc))
            length = sum(counts.values())
            self.doc_len.append(length)
            for term, tf in counts.items():
                self.inverted[term].append((doc_id, tf))
                df[term] += 1
        self.n_docs = len(docs)
        self.avgdl = float(np.mean(self.doc_len)) if self.doc_len else 0.0
        self.idf = {
            term: math.log(1.0 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        q_terms = Counter(tokenize(query))
        if not q_terms:
            return []
        for term, qtf in q_terms.items():
            postings = self.inverted.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            for doc_id, tf in postings:
                denom = tf + self.k1 * (1.0 - self.b + self.b * self.doc_len[doc_id] / max(self.avgdl, 1e-9))
                scores[doc_id] += qtf * idf * (tf * (self.k1 + 1.0)) / denom
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return ranked[:top_k]


def load_benchmark(benchmark_dir: Path) -> tuple[list[dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
    queries = read_jsonl(benchmark_dir / "queries.jsonl")
    qrels_rows = read_jsonl(benchmark_dir / "qrels.jsonl")
    registry = read_jsonl(benchmark_dir / "intermediate" / "skill_registry.jsonl")
    qrels = {row["query_id"]: row["skill_id"] for row in qrels_rows if int(row.get("relevance", 0)) > 0}
    missing = [q["query_id"] for q in queries if q["query_id"] not in qrels]
    if missing:
        raise ValueError(f"Missing qrels for {len(missing)} queries, first={missing[0]}")
    return queries, qrels, registry


def encode_with_bge(
    model_name: str,
    texts: list[str],
    batch_size: int,
    device: str,
    is_query: bool,
    normalize_embeddings: bool = True,
) -> np.ndarray:
    model = SentenceTransformer(model_name, device=device, local_files_only=True)
    if "bge" in model_name.lower() and is_query:
        texts = [f"Represent this sentence for searching relevant passages: {text}" for text in texts]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    return embeddings.astype("float32", copy=False)


def dense_rankings(
    query_embeddings: np.ndarray,
    doc_embeddings: np.ndarray,
    queries: list[dict[str, Any]],
    skill_ids: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    scores = np.matmul(query_embeddings, doc_embeddings.T)
    rows = []
    for i, query in enumerate(queries):
        row_scores = scores[i]
        if top_k >= len(skill_ids):
            idx = np.argsort(-row_scores)
        else:
            idx = np.argpartition(-row_scores, top_k)[:top_k]
            idx = idx[np.argsort(-row_scores[idx])]
        rankings = [
            {"rank": rank, "skill_id": skill_ids[int(j)], "score": float(row_scores[int(j)])}
            for rank, j in enumerate(idx[:top_k], start=1)
        ]
        rows.append({"query_id": query["query_id"], "ranking": rankings})
    return rows


def bm25_rankings(
    bm25: BM25Index,
    queries: list[dict[str, Any]],
    skill_ids: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    rows = []
    for query in queries:
        ranked = bm25.search(query["query"], top_k)
        rankings = [
            {"rank": rank, "skill_id": skill_ids[doc_id], "score": float(score)}
            for rank, (doc_id, score) in enumerate(ranked, start=1)
        ]
        rows.append({"query_id": query["query_id"], "ranking": rankings})
    return rows


def rrf_hybrid(
    dense_rows: list[dict[str, Any]],
    bm25_rows: list[dict[str, Any]],
    top_k: int,
    rrf_k: int,
) -> list[dict[str, Any]]:
    bm25_by_qid = {row["query_id"]: row for row in bm25_rows}
    rows = []
    for dense_row in dense_rows:
        qid = dense_row["query_id"]
        scores: dict[str, float] = defaultdict(float)
        sources: dict[str, dict[str, int]] = defaultdict(dict)
        for source_name, row in [("dense", dense_row), ("bm25", bm25_by_qid[qid])]:
            for item in row["ranking"]:
                sid = item["skill_id"]
                rank = int(item["rank"])
                scores[sid] += 1.0 / (rrf_k + rank)
                sources[sid][source_name] = rank
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]
        rankings = [
            {
                "rank": rank,
                "skill_id": sid,
                "score": float(score),
                "dense_rank": sources[sid].get("dense"),
                "bm25_rank": sources[sid].get("bm25"),
            }
            for rank, (sid, score) in enumerate(ranked, start=1)
        ]
        rows.append({"query_id": qid, "ranking": rankings})
    return rows


def metrics_for(rows: list[dict[str, Any]], qrels: dict[str, str]) -> dict[str, float]:
    n = len(rows)
    hit1 = 0
    recall5 = 0
    recall10 = 0
    recall20 = 0
    mrr = 0.0
    ndcg10 = 0.0
    for row in rows:
        gold = qrels[row["query_id"]]
        ranking = row["ranking"]
        ranks = [item["skill_id"] for item in ranking]
        if ranks and ranks[0] == gold:
            hit1 += 1
        if gold in ranks[:5]:
            recall5 += 1
        if gold in ranks[:10]:
            recall10 += 1
        if gold in ranks[:20]:
            recall20 += 1
        if gold in ranks:
            rank = ranks.index(gold) + 1
            mrr += 1.0 / rank
            if rank <= 10:
                ndcg10 += 1.0 / math.log2(rank + 1)
    denom = max(n, 1)
    return {
        "queries": n,
        "Hit@1": round(hit1 / denom, 6),
        "Recall@5": round(recall5 / denom, 6),
        "Recall@10": round(recall10 / denom, 6),
        "Recall@20": round(recall20 / denom, 6),
        "MRR": round(mrr / denom, 6),
        "NDCG@10": round(ndcg10 / denom, 6),
    }


def annotate_rows(rows: list[dict[str, Any]], method: str, qrels: dict[str, str], metrics: dict[str, float]) -> list[dict[str, Any]]:
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
                "gold_rank": gold_rank,
                "ranking": row["ranking"],
            }
        )
    return out


def run(args: argparse.Namespace) -> None:
    benchmark_dir = Path(args.benchmark_dir)
    results_dir = Path(args.results_dir)
    ensure_dir(results_dir)

    queries, qrels, registry = load_benchmark(benchmark_dir)
    skill_ids = [row["skill_id"] for row in registry]
    docs = [build_skill_doc(row, args.max_skill_chars) for row in registry]
    query_texts = [row["query"] for row in queries]

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    doc_emb = encode_with_bge(args.model, docs, args.batch_size, device, is_query=False)
    query_emb = encode_with_bge(args.model, query_texts, args.batch_size, device, is_query=True)

    dense_rows = dense_rankings(query_emb, doc_emb, queries, skill_ids, args.ranking_depth)
    dense_metrics = metrics_for(dense_rows, qrels)
    write_jsonl(results_dir / "flat_dense.jsonl", annotate_rows(dense_rows, "dense_bge", qrels, dense_metrics))

    bm25 = BM25Index(docs, k1=args.bm25_k1, b=args.bm25_b)
    bm25_rows = bm25_rankings(bm25, queries, skill_ids, args.ranking_depth)
    bm25_metrics = metrics_for(bm25_rows, qrels)
    write_jsonl(results_dir / "flat_bm25.jsonl", annotate_rows(bm25_rows, "bm25", qrels, bm25_metrics))

    hybrid_rows = rrf_hybrid(dense_rows, bm25_rows, args.ranking_depth, args.rrf_k)
    hybrid_metrics = metrics_for(hybrid_rows, qrels)
    write_jsonl(results_dir / "flat_hybrid.jsonl", annotate_rows(hybrid_rows, "dense_bm25_rrf", qrels, hybrid_metrics))

    metrics = {
        "benchmark_dir": str(benchmark_dir),
        "model": args.model,
        "device": device,
        "ranking_depth": args.ranking_depth,
        "topk_evaluated": [1, 5, 10, 20],
        "bm25": {"k1": args.bm25_k1, "b": args.bm25_b},
        "rrf": {"k": args.rrf_k, "inputs": ["dense_bge", "bm25"]},
        "doc_count": len(registry),
        "query_count": len(queries),
        "metrics": {
            "dense_bge": dense_metrics,
            "bm25": bm25_metrics,
            "dense_bm25_rrf": hybrid_metrics,
        },
    }
    write_json(results_dir / "flat_metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", default="benchmark")
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
