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
import re

# Ensure sqlite3 is new enough for chromadb if necessary
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

##################### MODELS #####################

class WWTPProcess(BaseModel):
    process_name: str = Field(description="EXACT generic_name from the reference list")
    category: str = Field(description="Category of the treatment process")
    implementation_status: Literal["implemented", "planned", "unclear"] = Field(
        description="Whether process is currently operational, planned for future, or unclear"
    )
    confidence: float = Field(ge=0, le=1, description="Confidence score (0-1)")
    evidence: str = Field(description="Key sentence(s) or description from permit that support this finding")
    name_variations_found: Optional[List[str]] = Field(
        default=None, 
        description="Any alternative names or variations found in the text"
    )

class WWTPAnalysis(BaseModel):
    facility_name: Optional[str] = None
    processes: List[WWTPProcess] = Field(description="Processes found in the facility")
    design_capacity: Optional[str] = None
    llm_time_seconds: Optional[float] = Field(default=None, description="Seconds spent in LLM calls")
    llm_attempts: int = Field(default=0, description="Number of LLM attempts made")

##################### CONTEXT PREPROCESSING #####################

def preprocess_permit_text(text: str) -> str:
    """Remove obvious data tables but keep treatment descriptions"""
    
    lines = text.split('\n')
    filtered_lines = []
    
    in_table = False
    
    for line in lines:
        stripped = line.strip()
        
        if not stripped:
            filtered_lines.append(line)
            continue
        
        # Detect obvious table patterns
        digit_count = sum(c.isdigit() for c in line)
        tab_count = line.count('\t')
        pipe_count = line.count('|')
        word_count = len(stripped.split())
        
        # Skip lines that are clearly tables
        if tab_count > 4 or pipe_count > 2:
            in_table = True
            continue
        
        # Skip data-heavy rows
        if word_count > 5 and digit_count > 20 and line.count('  ') > 3:
            in_table = True
            continue
        
        # Exit table mode when we see treatment keywords
        if in_table:
            lower = stripped.lower()
            treatment_keywords = ['treatment', 'process', 'system', 'consists of', 'includes']
            if any(kw in lower for kw in treatment_keywords):
                in_table = False
        
        if not in_table:
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)

##################### PROCESS REFERENCE BUILDER #####################

def build_process_reference_text():
    """Build a comprehensive reference text describing all processes for the LLM"""
    
    process_descriptions = []
    
    for proc in REFERENCE_PROCESSES["treatment_processes"]:
        generic_name = proc.get("generic_name", "")
        category = proc.get("category", "")
        subcategory = proc.get("subcategory", "")
        
        # Get alternative names
        alt_names = proc.get("alternative_names", [])
        if isinstance(alt_names, str):
            alt_names = [a.strip() for a in alt_names.split(",") if a.strip()]
        
        # Build description
        desc_parts = [f"- **{generic_name}** (Category: {category}"]
        if subcategory:
            desc_parts.append(f", Subcategory: {subcategory}")
        desc_parts.append(")")
        
        if alt_names:
            desc_parts.append(f"\n  Alternative names: {', '.join(alt_names)}")
        
        process_descriptions.append(''.join(desc_parts))
    
    return "\n".join(process_descriptions)

##################### MAIN RAG QUERY FUNCTION #####################

import llm_rag_loader as loader

def query_rag(query_text: str, k=5, verbose=False):
    """
    Main query function using LLM understanding to identify processes
    
    Args:
        query_text: Query to search for
        k: Number of documents to retrieve
        verbose: Show detailed output
    """

    # Prepare the DB
    embedding_function = loader.get_embedding_function()
    db = Chroma(persist_directory=loader.CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB
    enhanced_query = f"treatment processes systems operations: {query_text}"
    results = db.similarity_search_with_score(enhanced_query, k=k)
    print(f"Retrieved {len(results)} relevant documents from the database.")
    
    # Check for no-split flag
    no_split_flag = os.path.join(loader.CHROMA_PATH, "no_split")
    if os.path.exists(no_split_flag) and results:
        top_doc, _score = results[0]
        context_text = top_doc.page_content
    else:
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    # Preprocess to remove tables
    print("Preprocessing context...")
    original_length = len(context_text)
    context_text = preprocess_permit_text(context_text)
    removed_percentage = (1 - len(context_text) / original_length) * 100
    print(f"Removed {removed_percentage:.1f}% of text (data tables)")
    
    if verbose:
        print(f"\nContext length: {len(context_text)} characters")
        print(f"Context preview:\n{context_text[:600]}...\n")

    # Build reference for LLM
    process_reference = build_process_reference_text()
    
    # Create comprehensive system message
    system_message = f"""You are an expert in wastewater treatment process identification.

You will be given:
1. A comprehensive reference list of wastewater treatment processes
2. A facility permit describing the treatment operations

Your task is to systematically go through EACH process in the reference list and determine:
- Is this process mentioned or implemented at the facility?
- Is it currently operational, planned for future, or unclear?
- What evidence supports this determination?

REFERENCE PROCESSES:
{process_reference}

CRITICAL INSTRUCTIONS:
1. Go through EVERY process in the reference list above
2. For each process, analyze the permit text to see if that process is part of the facility treatment processes (carefully separate primary from secondary and tertiary processes)
3. Use your understanding of wastewater treatment to identify processes even if:
   - The exact name isn't used
   - Only technical descriptions are given
   - Alternative terminology is used
   - The process is implied by equipment or operations described

4. Distinguish between:
   - IMPLEMENTED: Currently operational ("consists of", "includes", "treats", "uses")
   - PLANNED: Future implementation ("will be", "proposed", "planned", "under construction")
   - UNCLEAR: Mentioned but status uncertain
   - UNDER OTHER FACILITY: If the process is described but clearly applies to a different facility or part of the system not covered by this permit

5. For each identified process:
   - Use ONLY the exact generic_name from the reference list
   - Provide confidence score (0-1) based on how clearly the process is described
   - Quote the key evidence from the permit
   - Note any alternative names found in the text

6. DO NOT include processes that are clearly not present (confidence < 0.5)
7. Be thorough - many facilities use 5-15+ processes

IMPORTANT: Think like an expert who understands that:
- "primary sedimentation" = sedimentation process in primary treatment
- "activated sludge biological treatment" = activated sludge process
- "inert media filtration" = filtration process
- "UV disinfection" = UV disinfection process
- Equipment descriptions often imply processes (e.g., "clarifiers" → sedimentation)"""

    user_message = f"""FACILITY PERMIT:

{context_text}

---

TASK: Analyze this facility permit and identify which processes from the reference list are present.

Go through each process systematically:
1. Check if the process (or its alternatives) is mentioned by name
2. Check if equipment/operations that indicate this process are described
3. Determine implementation status (implemented/planned/unclear)
4. Assess confidence and gather evidence

Extract if mentioned:
- Facility name
- Design capacity

Respond with a complete analysis of all processes found."""

    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_message},
    ]

    if verbose:
        print(f"\nPrompt statistics:")
        print(f"  System message: {len(system_message)} chars")
        print(f"  Reference processes: {len(process_reference)} chars")
        print(f"  User message: {len(user_message)} chars")
        print(f"  Total context: {len(context_text)} chars\n")

    # Call LLM with structured output
    print("Calling LLM for process identification...")
    
    try:
        t0 = time.time()
        response = chat(
            model='mistral:7b',
            messages=messages,
            format=WWTPAnalysis.model_json_schema(),
            options={'temperature': 0, 'num_ctx': 16384}  # Larger context window
        )
        elapsed = time.time() - t0
        print(f"LLM call completed in {elapsed:.2f} seconds")

        content = getattr(response, 'message', None)
        content_str = content.content if content is not None else str(response)

        if not content_str or not content_str.strip():
            raise ValueError("Empty response from LLM")

        try:
            wwtp_analysis = WWTPAnalysis.model_validate_json(content_str)
            wwtp_analysis.llm_time_seconds = elapsed
            wwtp_analysis.llm_attempts = 1
            print(f"Successfully extracted {len(wwtp_analysis.processes)} processes")
            
        except Exception as e_parse:
            print(f"Failed to parse formatted response: {e_parse}")
            print("Attempting fallback...")
            
            # Fallback to plain chat
            t1 = time.time()
            fallback = chat(model='mistral:7b', messages=messages, options={'temperature': 0, 'num_ctx': 16384})
            elapsed += time.time() - t1
            
            raw = getattr(fallback, 'message', None)
            raw_text = raw.content if raw is not None else str(fallback)
            
            if verbose:
                print("RAW RESPONSE:\n", raw_text[:1000], "...\n")
            
            # Try to parse JSON
            try:
                parsed = json.loads(raw_text)
                wwtp_analysis = WWTPAnalysis.model_validate(parsed)
                wwtp_analysis.llm_time_seconds = elapsed
                wwtp_analysis.llm_attempts = 2
            except:
                # Try extracting from markdown code block
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(1))
                    wwtp_analysis = WWTPAnalysis.model_validate(parsed)
                    wwtp_analysis.llm_time_seconds = elapsed
                    wwtp_analysis.llm_attempts = 2
                else:
                    print("Could not parse response as JSON")
                    wwtp_analysis = WWTPAnalysis(
                        processes=[], 
                        llm_time_seconds=elapsed, 
                        llm_attempts=2
                    )

    except Exception as e:
        print(f"Error invoking LLM: {e}")
        # Final fallback attempt
        try:
            t2 = time.time()
            fallback = chat(model='mistral:7b', messages=messages, options={'temperature': 0})
            elapsed2 = time.time() - t2
            
            raw = getattr(fallback, 'message', None)
            raw_text = raw.content if raw is not None else str(fallback)
            
            print("Fallback RAW RESPONSE:\n", raw_text[:500], "...\n")
            
            try:
                parsed = json.loads(raw_text)
                wwtp_analysis = WWTPAnalysis.model_validate(parsed)
                wwtp_analysis.llm_time_seconds = elapsed2
                wwtp_analysis.llm_attempts = 1
            except:
                wwtp_analysis = WWTPAnalysis(
                    processes=[], 
                    llm_time_seconds=elapsed2, 
                    llm_attempts=1
                )
        except Exception as e2:
            print(f"Final fallback failed: {e2}")
            return None, []

    # Verbose output
    if verbose:
        print(f"\n{'='*70}")
        print(f"ANALYSIS RESULTS")
        print(f"{'='*70}")
        print(f"Facility: {wwtp_analysis.facility_name or 'Not specified'}")
        print(f"Design Capacity: {wwtp_analysis.design_capacity or 'Not specified'}")
        print(f"Total Time: {wwtp_analysis.llm_time_seconds:.2f} seconds")
        print(f"LLM Attempts: {wwtp_analysis.llm_attempts}")
        print(f"\n{'='*70}")
        print(f"PROCESSES IDENTIFIED ({len(wwtp_analysis.processes)}):")
        print(f"{'='*70}\n")
        
        for i, proc in enumerate(wwtp_analysis.processes, 1):
            print(f"{i}. {proc.process_name}")
            print(f"   Category: {proc.category}")
            print(f"   Status: {proc.implementation_status.upper()}")
            print(f"   Confidence: {proc.confidence:.2f}")
            if proc.name_variations_found:
                print(f"   Variations found: {', '.join(proc.name_variations_found)}")
            print(f"   Evidence: {proc.evidence[:200]}...")
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
        if args.pdf:
            target = args.pdf
        elif args.arg:
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
        query_text = args.arg
        
        print("\n" + "="*70)
        print("INTELLIGENT PROCESS IDENTIFICATION")
        print("="*70 + "\n")
        
        analysis, sources = query_rag(query_text, verbose=args.verbose)
        
        if analysis is None:
            print("Failed to generate analysis")
            return
        
        print("\n" + "="*70)
        print("JSON OUTPUT")
        print("="*70)
        json_str = analysis.model_dump_json(indent=2)
        print(json_str)

        # Save JSON to output/llm_results
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "llm_results")
        os.makedirs(out_dir, exist_ok=True)

        # Build a safe filename
        timestamp = time.strftime("%Y%m%dT%H%M%S")
        facility_raw = (analysis.facility_name or "unknown_facility")
        facility_safe = re.sub(r'[^A-Za-z0-9._-]+', '_', facility_raw).strip('_')[:100]
        filename = f"llm_result_{facility_safe}_{timestamp}.json"
        out_path = os.path.join(out_dir, filename)

        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json_str)

        print(f"\nSaved JSON to: {out_path}")
        
        if not args.verbose:
            print(f"\nFound {len(analysis.processes)} processes")
            
            # Summary by implementation status
            implemented = [p for p in analysis.processes if p.implementation_status == "implemented"]
            planned = [p for p in analysis.processes if p.implementation_status == "planned"]
            unclear = [p for p in analysis.processes if p.implementation_status == "unclear"]
            
            print(f"  - Implemented: {len(implemented)}")
            print(f"  - Planned: {len(planned)}")
            print(f"  - Unclear: {len(unclear)}")
            
            print(f"Execution time: {analysis.llm_time_seconds:.2f} seconds")
            print(f"\nUse --verbose or -v for detailed output")


if __name__ == "__main__":
    main()