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
import sys
from pathlib import Path

import pandas as pd

# sentence-transformers and httpx log one INFO line per HF metadata request, which buries
# the per-facility progress output
for noisy in ("httpx", "sentence_transformers", "transformers", "faiss", "RetrievalSystem",
              "rag_llm_reranker_simplified"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

from helpers.utils import SEP, build_txt_jobs, extract_leaves
from step4_keyword_extraction import search_processes_in_text

TXT_DIR = "wwtp_process_extraction/output/permits/text"
FACILITIES_INFO_PATH = "wwtp_process_extraction/data/unit_processes_by_facility_manual.csv"
FULL_CA_PATH = "wwtp_process_extraction/output/site_data_relevant.csv"
UNITPROCESS_KEYWORDS_JSON = "wwtp_process_extraction/data/unitprocess_keywords.json"
# Deliberately outside output/llm_extraction/: these are retrieved literature chunks fed
# into step5's prompt, not extraction results. The schema-conformant items land in
# output/llm_extraction/ontology-based_<model>-waterrag/.
OUTPUT_DIR = Path("wwtp_process_extraction/output/waterrag_retrieval")
WATERRAG_DIR = os.getenv("WATERRAG_DIR", str(Path.home() / "waterrag"))

MAX_QUERY_TERMS = 8
MAX_CHUNKS = 12
MAX_CONTEXT_CHARS = 16000
OVERVIEW_QUERY_CHARS = 1500
RERANK_MAX_TOKENS = 8000

# usage accumulator for the monkeypatched reranker call, reset per facility
_rerank_usage = {"prompt": 0, "completion": 0}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cache WaterRAG literature context per permit for step5 --waterrag_context."
    )
    parser.add_argument("--waterrag_dir", default=WATERRAG_DIR,
                        help=f"Clone of the WaterRAG repo, containing the index (default: {WATERRAG_DIR}).")
    parser.add_argument("--txt_folder", default=TXT_DIR)
    parser.add_argument("--all_facilities", action="store_true",
                        help="Run the full CA set instead of the manually-read benchmark facilities.")
    parser.add_argument("--max_facilities", type=int, default=None)
    parser.add_argument("--rerank_model", default="gpt-5-mini",
                        help="Model for WaterRAG's LLM reranker, served by the Stanford proxy (default: gpt-5-mini).")
    parser.add_argument("--no_rerank", action="store_true",
                        help="Skip the LLM reranker; hybrid FAISS+BM25 retrieval only. Hits no API.")
    parser.add_argument("--chunk_top_k", type=int, default=20, help="Candidates retrieved per query.")
    parser.add_argument("--final_top_k", type=int, default=5, help="Chunks kept per query after reranking.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-retrieve facilities that already have a cached context file.")
    return parser.parse_args()


def load_waterrag(waterrag_dir, rerank_model, no_rerank):
    """Import WaterRAG from its clone and return (retrieval_system, reranker_or_None)."""
    waterrag_path = Path(waterrag_dir)
    if not (waterrag_path / "retrieval_simplified.py").exists():
        raise SystemExit(
            f"WaterRAG not found at {waterrag_path}. Clone it with:\n"
            "  git lfs install && git clone https://github.com/Mudi12138/WaterRAG.git ~/waterrag"
        )
    sys.path.insert(0, str(waterrag_path))

    from retrieval_simplified import RetrievalSystem
    from rag_llm_reranker_simplified import LLMReranker

    # Their _load_indexes wraps FAISS and BM25 in one try, so a stale BM25 pickle takes the
    # whole system down. Load them independently and degrade to vector-only if BM25 fails.
    def load_indexes_independently(self):
        import pickle
        import torch
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings

        self.faiss_index = None
        self.bm25_retriever = None

        faiss_path = os.path.join(self.index_path, "faiss_index")
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            # the T400 has 2 GB VRAM, not enough for bge-large; queries are short and few
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

    RetrievalSystem._load_indexes = load_indexes_independently

    index_path = str(waterrag_path / "0520_256")
    key_path = Path("wwtp_process_extraction/API_key.txt")
    # retrieval is entirely local, so --no_rerank needs no key at all
    api_key = key_path.read_text(encoding="utf-8").strip() if key_path.exists() else None
    if api_key is None and not no_rerank:
        raise SystemExit(f"{key_path} not found; needed for the reranker. Use --no_rerank to skip it.")
    api_url = "https://aiapi-prod.stanford.edu/v1/chat/completions"

    print(f"Loading WaterRAG index from {index_path} (this takes a few minutes)...")
    retrieval = RetrievalSystem(index_path=index_path, openai_api_key=api_key, openai_api_url=api_url)

    if no_rerank:
        return retrieval, None

    # Their _call_api hardcodes max_tokens=2000 and throws away the usage block. gpt-5-mini
    # spends most of that on reasoning and returns empty content, so raise the ceiling and
    # record tokens on the way past for the cost column in table_1.
    def call_api_with_usage(self, messages, api_model):
        import requests

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
        _rerank_usage["prompt"] += usage.get("prompt_tokens", 0)
        _rerank_usage["completion"] += usage.get("completion_tokens", 0)
        return data["choices"][0]["message"]["content"]

    LLMReranker._call_api = call_api_with_usage
    reranker = LLMReranker(api_key=api_key, default_model=rerank_model)
    # __init__ takes no api_url and reads OPENAI_API_URL from the environment, defaulting to
    # api.openai.com — which 401s on a Stanford key and then silently returns unranked order.
    reranker.api_url = api_url
    return retrieval, reranker


def build_queries(description_text, keywords):
    """One query per unit process the keyword matcher finds, plus one from the description opening.

    Querying with the whole extract does not work: it is mostly legal boilerplate, and
    bge-large truncates at 512 tokens anyway.
    """
    hits = {}
    search_processes_in_text(description_text, keywords, hits)
    # search_processes_in_text also flags parent categories; keep leaves, and drop the
    # 'Unspecified X' catch-alls (priority 1000), which make useless literature queries
    leaves = [
        name for name, details, _ in extract_leaves(keywords)
        if hits.get(name) == 1 and details.get("priority") != 1000
    ]

    queries = [f"{name} wastewater treatment" for name in leaves[:MAX_QUERY_TERMS]]
    opening = " ".join(description_text.split())[:OVERVIEW_QUERY_CHARS].strip()
    if opening:
        queries.append(opening)
    return queries


def retrieve_context(retrieval, reranker, queries, args):
    """Run every query, rerank, dedupe across queries, and cap the total context size."""
    per_query = []
    for query in queries:
        candidates, _ = retrieval.retrieve(query, chunk_top_k=args.chunk_top_k,
                                           final_top_k=args.chunk_top_k)
        if reranker:
            candidates = reranker.rerank(query, candidates, top_k=args.final_top_k,
                                         model=args.rerank_model)
        else:
            candidates = candidates[:args.final_top_k]
        per_query.append((query, candidates))

    # Interleave: take every query's best chunk before any query's second-best. Concatenating
    # query by query instead would spend the whole budget on the first two processes, so a
    # plant with UV disinfection would never see any UV literature.
    seen = set()
    chunks = []
    total = 0
    for rank in range(args.final_top_k):
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
    facilities_info = FULL_CA_PATH if args.all_facilities else FACILITIES_INFO_PATH
    jobs = build_txt_jobs(args.txt_folder, facilities_info)
    if args.max_facilities is not None:
        jobs = jobs[:args.max_facilities]
    if not jobs:
        raise SystemExit(f"No facilities found. Check {facilities_info} and --txt_folder.")

    keywords = json.loads(Path(UNITPROCESS_KEYWORDS_JSON).read_text(encoding="utf-8"))
    facilities_source_df = pd.read_csv(facilities_info, dtype=str).fillna("")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    retrieval, reranker = load_waterrag(args.waterrag_dir, args.rerank_model, args.no_rerank)

    usage_rows = []
    for row_idx, txt_path, txt_file, facility_name in jobs:
        txt_path = Path(txt_path)
        description_text = txt_path.read_text(encoding="utf-8").split(SEP, 1)[0]
        if not description_text.strip():
            print(f"{facility_name}: empty description section, skipping.")
            continue

        place_id = str(facilities_source_df.iloc[row_idx].get("Place ID", "")).strip()
        output_path = OUTPUT_DIR / f"{txt_path.stem}_{place_id}.json"
        if output_path.exists() and not args.overwrite:
            print(f"{facility_name}: cached, skipping.")
            continue

        print(f"\nProcessing {txt_file} for {facility_name}...")
        queries = build_queries(description_text, keywords)
        print(f"  {len(queries)} queries: {[q[:40] for q in queries]}")

        _rerank_usage["prompt"] = _rerank_usage["completion"] = 0
        chunks = retrieve_context(retrieval, reranker, queries, args)
        print(f"  kept {len(chunks)} chunks, rerank tokens "
              f"prompt={_rerank_usage['prompt']} completion={_rerank_usage['completion']}")

        # LLMReranker.rerank swallows per-batch API errors and returns the unranked order, so a
        # bad key or URL yields plausible-looking output that is silently retrieval-only. Zero
        # tokens after a full facility means every batch failed; stop rather than cache it.
        if reranker and _rerank_usage["prompt"] == 0:
            raise SystemExit(
                "Reranking produced no tokens — every batch failed (check the API key and that "
                f"{args.rerank_model} is served by the proxy). Re-run with --no_rerank to "
                "deliberately skip reranking."
            )

        output_path.write_text(json.dumps({
            "facility_name": facility_name,
            "place_id": place_id,
            "queries": queries,
            "chunks": chunks,
            "rerank_model": None if args.no_rerank else args.rerank_model,
            "rerank_prompt_token": _rerank_usage["prompt"],
            "rerank_completion_token": _rerank_usage["completion"],
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        usage_rows.append({
            "facility_name": facility_name,
            "place_id": place_id,
            "n_queries": len(queries),
            "n_chunks": len(chunks),
            "rerank_model": None if args.no_rerank else args.rerank_model,
            "prompt_token": _rerank_usage["prompt"],
            "completion_token": _rerank_usage["completion"],
        })

    if usage_rows:
        usage_path = OUTPUT_DIR / "token_usage_summary.csv"
        combined = pd.DataFrame(usage_rows)
        if usage_path.exists():
            combined = pd.concat([pd.read_csv(usage_path), combined], ignore_index=True)
        combined.drop_duplicates(subset="facility_name", keep="last").to_csv(usage_path, index=False)
        print(f"\nRerank token usage: {usage_path}")


if __name__ == "__main__":
    main()
