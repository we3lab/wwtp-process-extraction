"""
Retrieve wastewater-literature context for each permit from the WaterRAG index
(Zhai et al. 2026, https://github.com/Mudi12138/WaterRAG) and cache it to disk for
step5 --waterrag_context.

This is the only file in the pipeline that needs torch/langchain/faiss, so it runs in
its own conda env. Everything downstream just reads the cached JSON.

We use WaterRAG's retriever and reranker only, not its answer generation: generation
returns prose and is whitelisted to gpt-4o/gpt-4.1, neither of which fits our task or
our API proxy.

Authored with prompting to Claude Opus 4.8
"""

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
import requests
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from helpers.utils import (
    SEP,
    build_txt_jobs,
    leaves,
    TXT_DIR,
    MANUAL_CSV,
    SITE_DATA_RELEVANT_CSV,
    WATERRAG_RETRIEVAL_DIR,
)
from step4_keyword_extraction import search_processes_in_text

# WATERRAG_RETRIEVAL_DIR is deliberately outside output/llm_extraction/: these are retrieved
# literature chunks fed into step5's prompt, not extraction results. The schema-conformant
# items land in output/llm_extraction/ontology-based_<model>-waterrag/.
WATERRAG_DIR = Path.home() / "waterrag"

MAX_QUERY_TERMS = 8
MAX_CHUNKS = 12
MAX_CONTEXT_CHARS = 16000
OVERVIEW_QUERY_CHARS = 1500
RERANK_MAX_TOKENS = 8000
RERANK_MODEL = "gpt-5-mini"
CHUNK_TOP_K = 20
FINAL_TOP_K = 5

# usage accumulator for the monkeypatched reranker call, reset per facility
rerank_usage = {"prompt": 0, "completion": 0}

# sentence-transformers and httpx log one INFO line per HF metadata request, which buries
# the per-facility progress output
for noisy in ("httpx", "sentence_transformers", "transformers", "faiss", "RetrievalSystem",
              "rag_llm_reranker_simplified"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cache WaterRAG literature context per permit for step5 --waterrag_context."
    )
    parser.add_argument("--all_facilities", action="store_true",
                        help="Run the full CA set instead of the manually-read benchmark facilities.")
    return parser.parse_args()


# Their _load_indexes wraps FAISS and BM25 in one try, so a stale BM25 pickle takes the
# whole system down. Load them independently and degrade to vector-only if BM25 fails.
def load_indexes_independently(self):
    self.bm25_retriever = None

    faiss_path = os.path.join(self.index_path, "faiss_index")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    self.faiss_index = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    print("  FAISS index loaded")

    bm25_path = os.path.join(self.index_path, "bm25_retriever.pkl")
    try:
        with open(bm25_path, "rb") as handle:
            self.bm25_retriever = pickle.load(handle)
        print("  BM25 retriever loaded")
    except Exception as exc:
        print(f"  BM25 retriever failed to load ({exc}); falling back to vector-only retrieval.")


# Their _call_api hardcodes max_tokens=2000 and throws away the usage block. gpt-5-mini
# spends most of that on reasoning and returns empty content, so raise the ceiling and
# record tokens on the way past for the cost column in table_1.
def call_api_with_usage(self, messages, api_model):
    response = requests.post(
        self.api_url,
        headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        json={
            "model": api_model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": RERANK_MAX_TOKENS,
            "max_completion_tokens": RERANK_MAX_TOKENS,
        },
        timeout=600,
    )
    response.raise_for_status()
    data = response.json()
    usage = data.get("usage", {})
    rerank_usage["prompt"] += usage.get("prompt_tokens", 0)
    rerank_usage["completion"] += usage.get("completion_tokens", 0)
    return data["choices"][0]["message"]["content"]


def load_waterrag():
    """Import WaterRAG from its clone and return (retrieval_system, reranker)."""
    if not (WATERRAG_DIR / "retrieval_simplified.py").exists():
        raise SystemExit(
            f"WaterRAG not found at {WATERRAG_DIR}. Clone it with:\n"
            "  git lfs install && git clone https://github.com/Mudi12138/WaterRAG.git ~/waterrag"
        )
    sys.path.insert(0, str(WATERRAG_DIR))

    from retrieval_simplified import RetrievalSystem
    from rag_llm_reranker_simplified import LLMReranker

    RetrievalSystem._load_indexes = load_indexes_independently

    index_path = str(WATERRAG_DIR / "0520_256")
    key_path = Path("wwtp_process_extraction/API_key.txt")
    if not key_path.exists():
        raise SystemExit(f"{key_path} not found; needed for the reranker.")
    api_key = key_path.read_text(encoding="utf-8").strip()
    api_url = "https://aiapi-prod.stanford.edu/v1/chat/completions"

    print(f"Loading WaterRAG index from {index_path} (this takes a few minutes)...")
    retrieval = RetrievalSystem(index_path=index_path, openai_api_key=api_key, openai_api_url=api_url)

    LLMReranker._call_api = call_api_with_usage
    reranker = LLMReranker(api_key=api_key, default_model=RERANK_MODEL)
    # __init__ takes no api_url and reads OPENAI_API_URL from the environment, defaulting to
    # api.openai.com — which 401s on a Stanford key and then silently returns unranked order.
    reranker.api_url = api_url
    return retrieval, reranker


def build_queries(description_text):
    """One query per unit process the keyword matcher finds, plus one from the description opening.

    Querying with the whole extract does not work: it is mostly legal boilerplate, and
    bge-large truncates at 512 tokens anyway.
    """
    hits = search_processes_in_text(description_text)
    # drop the 'Unspecified X' catch-alls (priority 1000), which make useless literature queries
    processes = [
        name for name, details, _ in leaves
        if name in hits and details.get("priority") != 1000
    ]

    queries = [f"{name} wastewater treatment" for name in processes[:MAX_QUERY_TERMS]]
    opening = " ".join(description_text.split())[:OVERVIEW_QUERY_CHARS]
    if opening:
        queries.append(opening)
    return queries


def retrieve_context(retrieval, reranker, queries):
    """Run every query, rerank, dedupe across queries, and cap the total context size."""
    per_query = []
    for query in queries:
        candidates, _ = retrieval.retrieve(query, chunk_top_k=CHUNK_TOP_K, final_top_k=CHUNK_TOP_K)
        candidates = reranker.rerank(query, candidates, top_k=FINAL_TOP_K, model=RERANK_MODEL)
        per_query.append((query, candidates))

    # Interleave: take every query's best chunk before any query's second-best. Concatenating
    # query by query instead would spend the whole budget on the first two processes, so a
    # plant with UV disinfection would never see any UV literature.
    seen = set()
    chunks = []
    total = 0
    for rank in range(FINAL_TOP_K):
        for query, candidates in per_query:
            if rank >= len(candidates):
                continue
            doc = candidates[rank]
            key = (doc.metadata.get("document_id"), doc.metadata.get("chunk_id"))
            if key in seen:
                continue
            if len(chunks) >= MAX_CHUNKS or total + len(doc.page_content) > MAX_CONTEXT_CHARS:
                return chunks
            seen.add(key)
            total += len(doc.page_content)
            chunks.append({
                "document_id": doc.metadata.get("document_id"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "citation": doc.metadata.get("citation_info"),
                "query": query,
                "text": doc.page_content,
            })
    return chunks


def main():
    args = parse_args()
    facilities_info = SITE_DATA_RELEVANT_CSV if args.all_facilities else MANUAL_CSV
    jobs = build_txt_jobs(facilities_info)
    if not jobs:
        raise SystemExit(f"No facilities found. Check {facilities_info} and {TXT_DIR}.")

    retrieval, reranker = load_waterrag()

    for _, txt_path, facility_name, place_id in jobs:
        description_text = txt_path.read_text(encoding="utf-8").split(SEP, 1)[0]
        if not description_text.strip():
            print(f"{facility_name}: empty description section, skipping.")
            continue

        output_path = WATERRAG_RETRIEVAL_DIR / f"{txt_path.stem}_{place_id}.json"
        if output_path.exists():
            print(f"{facility_name}: cached, skipping.")
            continue

        print(f"\nProcessing {txt_path.name} for {facility_name}...")
        queries = build_queries(description_text)
        print(f"  {len(queries)} queries: {[q[:40] for q in queries]}")

        rerank_usage["prompt"] = rerank_usage["completion"] = 0
        chunks = retrieve_context(retrieval, reranker, queries)
        print(f"  kept {len(chunks)} chunks, rerank tokens "
              f"prompt={rerank_usage['prompt']} completion={rerank_usage['completion']}")

        # LLMReranker.rerank swallows per-batch API errors and returns the unranked order, so a
        # bad key or URL yields plausible-looking output that is silently retrieval-only. Zero
        # tokens after a full facility means every batch failed; stop rather than cache it.
        if rerank_usage["prompt"] == 0:
            raise SystemExit(
                "Reranking produced no tokens — every batch failed (check the API key and that "
                f"{RERANK_MODEL} is served by the proxy)."
            )

        output_path.write_text(json.dumps({
            "facility_name": facility_name,
            "place_id": place_id,
            "queries": queries,
            "chunks": chunks,
            "rerank_model": RERANK_MODEL,
            "rerank_prompt_token": rerank_usage["prompt"],
            "rerank_completion_token": rerank_usage["completion"],
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    usage_rows = []
    for context_path in sorted(WATERRAG_RETRIEVAL_DIR.glob("*.json")):
        context = json.loads(context_path.read_text(encoding="utf-8"))
        usage_rows.append({
            "facility_name": context["facility_name"],
            "place_id": context["place_id"],
            "context_file": context_path.name,
            "n_queries": len(context["queries"]),
            "n_chunks": len(context["chunks"]),
            "rerank_model": context["rerank_model"],
            "prompt_token": context["rerank_prompt_token"],
            "completion_token": context["rerank_completion_token"],
        })
    usage_path = WATERRAG_RETRIEVAL_DIR / "token_usage_summary.csv"
    pd.DataFrame(usage_rows).to_csv(usage_path, index=False)
    print(f"\nRerank token usage: {usage_path}")


if __name__ == "__main__":
    main()
