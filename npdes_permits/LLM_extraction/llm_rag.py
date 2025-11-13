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
import multiprocessing
import traceback
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
VALID_CATEGORY_NAMES = tuple([proc["category"] for proc in REFERENCE_PROCESSES["treatment_processes"]])

# Create the Literal type with all valid process names
ProcessNameLiteral = Literal[VALID_PROCESS_NAMES]
CategoryLiteral = Literal[VALID_CATEGORY_NAMES]


class WWTPProcess(BaseModel):
    process_name: ProcessNameLiteral = Field(description="EXACT generic_name from the JSON - MUST be one of the predefined values")
    category: CategoryLiteral = Field(description="Functional category from the reference JSON")
    subcategory: Optional[str] = Field(default=None, description="Subcategory if applicable")
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

##################### Define the main RAG query function #####################

import llm_rag_loader as loader

def create_system_message():
    """Create system message with the actual JSON content"""
    reference_json_str = json.dumps(REFERENCE_PROCESSES, indent=2)
    
    return f"""You are an expert in wastewater treatment process identification.

You have access to a comprehensive reference of valid WWTP processes you must search for in the provided context:

{reference_json_str}

Feel free to ingnore irrelevant information given in the wastewater treament facility permit.

You will think step by step.
"""

USER_MESSAGE = """Permit of the wastewater treatment facility:

{context}

---

Question: {question}

Reasoning as follows:

1. For each treatment process of the reference json, see if it is mentioned in the facility permit, by looking at both the "generic_name" and the "alternative_names" list.

2. Determine if this mentioned process is currently implemented in the facility described in the context, or if it is a planned change.

3. Use the category from the JSON structure (e.g., "preliminary_treatment" → category: "Preliminary Treatment")

4. If you find a process mentioned in the context that CANNOT be matched to any reference process:
    - Add it to the "unknown_processes" list
    - Provide a reason why it couldn't be matched (e.g., "Not in reference", "Ambiguous name", etc.)

5. Only include processes that are explicitly mentioned in the context

6. Provide confidence scores based on how clearly each process is mentioned

IMPORTANT: 
- Use ONLY the exact generic_name from the reference JSON
- If a process is mentioned but not in the reference, add it to unknown_processes
- Extract the facility name and design capacity if mentioned"""

def query_rag(query_text: str, k=25, verbose=False):

    # Prepare the DB.
    embedding_function = loader.get_embedding_function()
    db = Chroma(persist_directory=loader.CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=k)

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

    # Helper: run an Ollama chat call in a subprocess to enforce a hard timeout.
    def _llm_worker(messages_obj, format_schema, out_q):
        try:
            if format_schema:
                resp = chat(model='mistral:7b', messages=messages_obj, format=format_schema, options={'temperature': 0})
            else:
                resp = chat(model='mistral:7b', messages=messages_obj, options={'temperature': 0})
            content = getattr(resp, 'message', None)
            content_str = content.content if content is not None else str(resp)
            out_q.put({'ok': True, 'content': content_str})
        except Exception:
            out_q.put({'ok': False, 'error': traceback.format_exc()})

    def run_llm_with_timeout(messages_obj, format_schema=None, timeout=30):
        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=_llm_worker, args=(messages_obj, format_schema, q))
        p.start()
        p.join(timeout)
        if p.is_alive():
            # Hard kill the process if it exceeded the timeout
            p.terminate()
            p.join()
            raise TimeoutError(f"LLM call exceeded timeout of {timeout}s and was terminated")
        try:
            result = q.get_nowait()
        except Exception:
            raise RuntimeError("No result returned from LLM worker")
        if not result.get('ok'):
            raise RuntimeError(result.get('error', 'Unknown error in LLM worker'))
        return result['content']

    # Call Ollama with structured output using the hard timeout helper. If
    # the formatted/schema call fails (timeout/parse issues), fall back to
    # a plain chat call (also with a timeout) and attempt best-effort parsing.
    try:
        content_str = run_llm_with_timeout(messages, format_schema=WWTPAnalysis.model_json_schema(), timeout=30)

        if not content_str or not content_str.strip():
            raise ValueError("Empty response from formatted call")

        try:
            wwtp_analysis = WWTPAnalysis.model_validate_json(content_str)
        except Exception as e_parse:
            # Parsing failed: fall back to plain chat call to inspect raw text
            print("Failed to parse formatted response:", e_parse)
            print("Falling back to plain chat (no format) to inspect raw text.")
            raw_text = run_llm_with_timeout(messages, format_schema=None, timeout=30)
            print("RAW LLM RESPONSE:\n", raw_text)
            try:
                parsed = json.loads(raw_text)
                wwtp_analysis = WWTPAnalysis.model_validate(parsed)
            except Exception as e2:
                print("Could not parse fallback response as JSON:", e2)
                wwtp_analysis = WWTPAnalysis(processes=[], unknown_processes=[])

    except Exception as e:
        print("Error invoking LLM (formatted or fallback):", e)
        # Try a final plain chat attempt to capture any raw output for debugging
        try:
            raw_text = run_llm_with_timeout(messages, format_schema=None, timeout=10)
            print("Fallback RAW LLM RESPONSE:\n", raw_text)
            try:
                parsed = json.loads(raw_text)
                wwtp_analysis = WWTPAnalysis.model_validate(parsed)
            except Exception as e3:
                print("Could not parse fallback raw response as JSON:", e3)
                wwtp_analysis = WWTPAnalysis(processes=[], unknown_processes=[])
        except Exception as e2:
            print("Final error invoking LLM (no more fallbacks):", e2)
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
            if proc.subcategory:
                print(f"   Subcategory: {proc.subcategory}")
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