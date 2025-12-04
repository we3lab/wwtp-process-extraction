import sqlite3
import sys
import os
import warnings
import argparse
import json
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core._api import LangChainDeprecationWarning
from ollama import chat
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import time

# Ensure sqlite3 is new enough for chromadb if necessary (same check as loader)
if (sqlite3.sqlite_version_info[0] < 3) or (
    (sqlite3.sqlite_version_info[0] == 3) and (sqlite3.sqlite_version_info[1] < 35)
):
    print("Upgrading sqlite3 version from " + sqlite3.sqlite_version)
    import pysqlite3  # noqa: F401

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    import sqlite3

    print("New sqlite3 version is " + sqlite3.sqlite_version)
    import chromadb  # noqa: F401


os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

##################### Load Reference Processes #####################
REFERENCE_PROCESSES = json.load(open("data/treatment_processes.json"))

# Extract all generic names from the list
VALID_PROCESS_NAMES = tuple([proc["generic_name"] for proc in REFERENCE_PROCESSES["treatment_processes"]])
VALID_CATEGORIES = tuple(set([proc["category"] for proc in REFERENCE_PROCESSES["treatment_processes"] if "category" in proc]))
# Create the Literal type with all valid process names
ProcessNameLiteral = Literal[VALID_PROCESS_NAMES]
CategoryLiteral = Literal[VALID_CATEGORIES]

class WWTPProcess(BaseModel):
    process_name: ProcessNameLiteral = Field(description="EXACT generic_name from the list - MUST be one of the predefined values")
    category: CategoryLiteral = Field(description="Category of the treatment process")
    confidence: float = Field(ge=0, le=1, description="Confidence score")
    match_sentence: str = Field(description="Sentence in the context that led to the match")
    alternative_name_used: Optional[str] = Field(default=None, description="If matched via alternative name, which one")

class UnknownProcess(BaseModel):
    process_mentioned: str = Field(description="Process name as mentioned in the context")
    confidence: float = Field(ge=0, le=1, description="Confidence that this is a real process")
    match_sentence: str = Field(description="Sentence where this process was mentioned")
    reason_not_matched: str = Field(description="Why this couldn't be matched to reference processes")

class WWTPAnalysis(BaseModel):
    facility_name: Optional[str] = None
    processes: List[WWTPProcess] = Field(description="Processes that match the reference JSON")
    unknown_processes: List[UnknownProcess] = Field(default=[], description="Processes mentioned but not in reference JSON")
    design_capacity: Optional[str] = None
    # timing info: seconds spent waiting for LLM responses (sum of attempts)
    llm_time_seconds: Optional[float] = Field(default=None, description="Seconds spent in LLM calls (sum of attempts)")
    llm_attempts: int = Field(default=0, description="Number of LLM attempts made (formatted, fallback, ...)")

##################### Define the main RAG query function #####################

import llm_rag_loader as loader

def create_system_message():
    """Create a compact, human-readable system message listing allowed processes.

    We provide a short curated list (one line per process) with the generic
    name and alternative names to keep the prompt compact and focused.
    """
    lines = []
    for proc in REFERENCE_PROCESSES.get("treatment_processes", []):
        generic = proc.get("generic_name", "").strip()
        alts = proc.get("alternative_names") or []
        if isinstance(alts, str):
            # support comma-separated string variants
            alts = [a.strip() for a in alts.split(",") if a.strip()]
        alt_part = f" ({', '.join(alts)})" if alts else ""
        category = proc.get("category")
        cat_part = f" — {category}" if category else ""
        lines.append(f"{generic}{alt_part}{cat_part}")

    compact_list = "\n".join(lines)

    return f"""You are an expert in wastewater treatment process identification.

Below is a compact, curated list of allowed WWTP processes (one per line):

{compact_list}

When matching, use only the exact generic names shown above as canonical values.
Ignore irrelevant permit inforamtion (especially all sort of tables). Think step by step.
"""

USER_MESSAGE = """Permit of the wastewater treatment facility:

{context}

---

Question: {question}

Reasoning as follows:

1. For each treatment process in the reference list above, see if it is mentioned in the facility permit, by looking at both the generic name and any alternative names shown.

2. Determine if this mentioned process is currently implemented in the facility described in the context, or if it is a planned change.

3. If you find a process mentioned in the context that CANNOT be matched to any reference process:
    - Add it to the "unknown_processes" list
    - Provide a reason why it couldn't be matched (e.g., "Not in reference", "Ambiguous name", etc.)

4. Only include processes that are explicitly mentioned in the context

5. Provide confidence scores based on how clearly each process is mentioned

- IMPORTANT:
- Use ONLY the exact generic name from the reference list above as the canonical value
- If a process is mentioned but not in the reference, add it to unknown_processes
- Extract the facility name and design capacity if mentioned"""


def query_rag(query_text: str, k=10, verbose=False):

    # Prepare the DB.
    embedding_function = loader.get_embedding_function()
    db = Chroma(persist_directory=loader.CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=k)
    print(f"Retrieved {len(results)} relevant documents from the database.")
    # If the DB was initialized with --no-split, a flag file is written to
    # CHROMA_PATH/no_split; in that case prefer using the single top
    # document's full text as the context instead of concatenating many
    # small chunks.
    no_split_flag = os.path.join(loader.CHROMA_PATH, "no_split")
    if os.path.exists(no_split_flag) and results:
        top_doc, _score = results[0]
        context_text = top_doc.page_content
    else:
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Prepare messages once
    messages = [
        {'role': 'system', 'content': create_system_message()},
        {'role': 'user', 'content': USER_MESSAGE.format(context=context_text, question=query_text)},
    ]
    if verbose:
        print(f"User message: {messages[1]['content']}\n")
        print(f"system message: {messages[0]['content']}\n")

    # Call Ollama with structured output first; if parsing fails fall back to a
    # plain chat call and attempt best-effort JSON parsing. No hard process-level
    # timeout is enforced here (calls rely on the Ollama client's own options).
    try:
        t0 = time.time()
        response = chat(
            model='mistral:7b',
            messages=messages,
            #format=WWTPAnalysis.model_json_schema(),
            options={'temperature': 0},
        )
        elapsed = time.time() - t0

        content = getattr(response, 'message', None)
        content_str = content.content if content is not None else str(response)

        if not content_str or not content_str.strip():
            raise ValueError("Empty response from formatted call")
        print("Formatted LLM RESPONSE:\n", content_str)
        try:
            wwtp_analysis = WWTPAnalysis.model_validate_json(content_str)
            # record timing
            wwtp_analysis.llm_time_seconds = elapsed
            wwtp_analysis.llm_attempts = 1
        except Exception as e_parse:
            # Parsing failed: fall back to a plain chat call and dump raw text
            print("Failed to parse formatted response:", e_parse)
            print("Falling back to plain chat (no format) to inspect raw text.")
            t1 = time.time()
            fallback = chat(model='mistral:7b', messages=messages, options={'temperature': 0})
            elapsed += time.time() - t1
            raw = getattr(fallback, 'message', None)
            raw_text = raw.content if raw is not None else str(fallback)
            print("RAW LLM RESPONSE:\n", raw_text)
            # Attempt best-effort JSON extraction, otherwise return empty analysis
            try:
                parsed = json.loads(raw_text)
                wwtp_analysis = WWTPAnalysis.model_validate(parsed)
                wwtp_analysis.llm_time_seconds = elapsed
                wwtp_analysis.llm_attempts = 2
            except Exception as e2:
                print("Could not parse fallback response as JSON:", e2)
                wwtp_analysis = WWTPAnalysis(processes=[], unknown_processes=[], llm_time_seconds=elapsed, llm_attempts=2)

    except Exception as e:
        print("Error invoking LLM (formatted):", e)
        # Try a simple plain chat to capture any raw output for debugging
        try:
            t2 = time.time()
            fallback = chat(model='mistral:7b', messages=messages, options={'temperature': 0})
            elapsed2 = time.time() - t2
            raw = getattr(fallback, 'message', None)
            raw_text = raw.content if raw is not None else str(fallback)
            print("Fallback RAW LLM RESPONSE:\n", raw_text)
            try:
                parsed = json.loads(raw_text)
                wwtp_analysis = WWTPAnalysis.model_validate(parsed)
                wwtp_analysis.llm_time_seconds = elapsed2
                wwtp_analysis.llm_attempts = 1
            except Exception as e3:
                print("Could not parse fallback raw response as JSON:", e3)
                wwtp_analysis = WWTPAnalysis(processes=[], unknown_processes=[], llm_time_seconds=elapsed2, llm_attempts=1)
        except Exception as e2:
            print("Final error invoking LLM:", e2)
            return None, []

    if verbose:
        print(f"\n{'='*60}")
        print(f"Facility: {wwtp_analysis.facility_name or 'Not specified'}")
        print(f"Design Capacity: {wwtp_analysis.design_capacity or 'Not specified'}")
        print(f"\n{'='*60}")
        print(f"MATCHED PROCESSES ({len(wwtp_analysis.processes)}):")
        print(f"{'='*60}\n")
        
        for i, proc in enumerate(wwtp_analysis.processes, 1):
            print(f"{i}. {proc.process_name}")
            print(f"   Category: {proc.category}")
            print(f"   Confidence: {proc.confidence:.2f}")
            if proc.alternative_name_used:
                print(f"   Matched via: {proc.alternative_name_used}")
            print(f"   Context: {proc.match_sentence[:100]}...")
            print()
        
        if wwtp_analysis.unknown_processes:
            print(f"\n{'='*60}")
            print(f"UNKNOWN PROCESSES ({len(wwtp_analysis.unknown_processes)}):")
            print(f"{'='*60}\n")
            
            for i, unk in enumerate(wwtp_analysis.unknown_processes, 1):
                print(f"{i}. {unk.process_mentioned}")
                print(f"   Confidence: {unk.confidence:.2f}")
                print(f"   Reason: {unk.reason_not_matched}")
                print(f"   Context: {unk.match_sentence[:100]}...")
                print()
    
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    
    return wwtp_analysis, sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--initialize", action="store_true", help="Initialize the database.")
    parser.add_argument("--no-split", action="store_true", help="When initializing, do not split documents into chunks.")
    parser.add_argument("--pdf", type=str, help="(Optional) PDF filename or path relative to repo root.")
    parser.add_argument("--query", action="store_true", help="Query the database.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output.")
    parser.add_argument("arg", type=str, help="Positional argument: treated as PDF filename when used with --initialize, or as query text when used with --query.", nargs="?")
    args = parser.parse_args()
    
    if args.reset:
        print("Clearing Database")
        loader.clear_database()
        
    if args.initialize:
        # Determine target: priority --pdf, then positional pdf_name, then
        # if a single positional was provided as query_text we accept that as
        # the filename when initializing. Otherwise default to folder.
        if args.pdf:
            target = args.pdf
        elif args.arg:
            # positional arg used as pdf filename when initializing
            target = os.path.join("npdes_permits", args.arg)
        else:
            target = "npdes_permits"

        documents = loader.load_documents(target)
        if args.no_split:
            loader.add_to_chroma(documents)
            try:
                os.makedirs(loader.CHROMA_PATH, exist_ok=True)
                open(os.path.join(loader.CHROMA_PATH, "no_split"), "w").close()
            except Exception:
                pass
        else:
            chunks = loader.split_documents(documents)
            loader.add_to_chroma(chunks)
            try:
                no_split_flag = os.path.join(loader.CHROMA_PATH, "no_split")
                if os.path.exists(no_split_flag):
                    os.remove(no_split_flag)
            except Exception:
                pass
                
    if args.query:
        # positional arg used as query text when querying
        query_text = args.arg
        analysis, sources = query_rag(query_text, verbose=args.verbose)
        
        print("\n" + "="*60)
        print("JSON OUTPUT")
        print("="*60)
        print(analysis.model_dump_json(indent=2))
        
        if not args.verbose:
            print(f"\nFound {len(analysis.processes)} matched processes")
            print(f"Found {len(analysis.unknown_processes)} unknown processes")
            print(f"\nUse --verbose or -v for detailed output")


if __name__ == "__main__":
    main()