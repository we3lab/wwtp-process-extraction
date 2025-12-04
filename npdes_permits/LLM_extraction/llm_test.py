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
from typing import List, Optional
import time
from rapidfuzz import fuzz, process as fuzzy_process

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

##################### UNCONSTRAINED MODELS (Stage 1) #####################

class WWTPProcessUnconstrained(BaseModel):
    """Unconstrained extraction - allows any process name"""
    process_name: str = Field(description="Name of the treatment process as mentioned in text")
    confidence: float = Field(ge=0, le=1, description="Confidence score")
    match_sentence: str = Field(description="Exact sentence from the permit that mentions this process")
    category: Optional[str] = Field(default=None, description="Category if identifiable")

class WWTPAnalysisUnconstrained(BaseModel):
    """Unconstrained analysis output"""
    facility_name: Optional[str] = None
    processes: List[WWTPProcessUnconstrained] = Field(description="All processes explicitly mentioned in the permit text")
    design_capacity: Optional[str] = None

##################### CONSTRAINED MODELS (Final Output) #####################

class WWTPProcess(BaseModel):
    process_name: str = Field(description="Canonical generic_name from reference")
    category: str = Field(description="Category of the treatment process")
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
    llm_time_seconds: Optional[float] = Field(default=None, description="Seconds spent in LLM calls")
    llm_attempts: int = Field(default=0, description="Number of LLM attempts made")

##################### CONTEXT PREPROCESSING #####################

def preprocess_permit_text(text: str) -> str:
    """Remove obvious tables but keep treatment descriptions intact"""
    
    lines = text.split('\n')
    filtered_lines = []
    
    in_table = False
    consecutive_blank = 0
    
    for line in lines:
        stripped = line.strip()
        
        # Track blank lines to detect table boundaries
        if not stripped:
            consecutive_blank += 1
            filtered_lines.append(line)
            continue
        else:
            consecutive_blank = 0
        
        # VERY aggressive table detection - only remove obvious data tables
        # Count digits, tabs, and pipe characters
        digit_count = sum(c.isdigit() for c in line)
        tab_count = line.count('\t')
        pipe_count = line.count('|')
        word_count = len(stripped.split())
        
        # Pattern 1: Lines with excessive tabs (clearly formatted tables)
        if tab_count > 4:
            in_table = True
            continue
        
        # Pattern 2: Lines with pipes (markdown/ascii tables)
        if pipe_count > 2:
            in_table = True
            continue
        
        # Pattern 3: Lines that are mostly numbers with many spaces
        # e.g., "123  456  789  012" but NOT "design capacity of 15.0 MGD"
        if word_count > 5 and digit_count > 20 and line.count('  ') > 3:
            in_table = True
            continue
        
        # Pattern 4: Repetitive structure (units, parameters, values)
        # Skip lines that look like "Parameter  Unit  Value  Limit"
        lower = stripped.lower()
        if any(pattern in lower for pattern in ['unit  ', 'units  ', 'mg/l  ', 'µg/l  ']):
            if word_count > 4:
                in_table = True
                continue
        
        # Exit table mode after 2+ blank lines or when we see treatment keywords
        if in_table:
            treatment_keywords = ['treatment', 'process', 'system', 'consists of', 'includes']
            if any(kw in lower for kw in treatment_keywords):
                in_table = False
        
        if not in_table:
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)

##################### FUZZY MATCHING (Stage 2) #####################

def build_reference_lookup():
    """Build lookup dictionary for fast matching"""
    name_to_process = {}
    
    for proc in REFERENCE_PROCESSES["treatment_processes"]:
        generic = proc["generic_name"].lower().strip()
        
        # Store reference to original process
        name_to_process[generic] = proc
        
        # Add alternative names
        alt_names = proc.get("alternative_names", [])
        if isinstance(alt_names, str):
            alt_names = [a.strip() for a in alt_names.split(",") if a.strip()]
        
        for alt in alt_names:
            alt_lower = alt.lower().strip()
            if alt_lower:
                name_to_process[alt_lower] = proc
    
    return name_to_process

def match_to_reference(extracted_processes: List[WWTPProcessUnconstrained], 
                       reference_lookup: dict,
                       threshold: int = 80):
    """Match extracted process names to canonical reference using fuzzy matching"""
    
    matched = []
    unknown = []
    
    all_ref_names = list(reference_lookup.keys())
    
    for extracted in extracted_processes:
        name_lower = extracted.process_name.lower().strip()
        
        # Exact match first
        if name_lower in reference_lookup:
            ref_proc = reference_lookup[name_lower]
            matched.append(WWTPProcess(
                process_name=ref_proc["generic_name"],
                category=ref_proc.get("category", "Unknown"),
                confidence=extracted.confidence,
                match_sentence=extracted.match_sentence,
                alternative_name_used=extracted.process_name if name_lower != ref_proc["generic_name"].lower() else None
            ))
        else:
            # Fuzzy match
            best_match = fuzzy_process.extractOne(
                name_lower,
                all_ref_names,
                scorer=fuzz.ratio
            )
            
            if best_match and best_match[1] >= threshold:
                ref_proc = reference_lookup[best_match[0]]
                matched.append(WWTPProcess(
                    process_name=ref_proc["generic_name"],
                    category=ref_proc.get("category", "Unknown"),
                    confidence=extracted.confidence * (best_match[1] / 100),
                    match_sentence=extracted.match_sentence,
                    alternative_name_used=extracted.process_name
                ))
            else:
                # No good match found
                unknown.append(UnknownProcess(
                    process_mentioned=extracted.process_name,
                    confidence=extracted.confidence,
                    match_sentence=extracted.match_sentence,
                    reason_not_matched=f"No match in reference (best: {best_match[0] if best_match else 'none'}, score: {best_match[1] if best_match else 0})"
                ))
    
    return matched, unknown

##################### MAIN RAG QUERY FUNCTION #####################

import llm_rag_loader as loader

def query_rag(query_text: str, k=5, verbose=False, use_old_method=False):
    """
    Main query function with two-stage pipeline optimization
    
    Args:
        query_text: Query to search for
        k: Number of documents to retrieve
        verbose: Show detailed output
        use_old_method: If True, use old constrained method (for comparison)
    """

    # Prepare the DB
    embedding_function = loader.get_embedding_function()
    db = Chroma(persist_directory=loader.CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB with enhanced query
    enhanced_query = f"treatment processes systems operations: {query_text}"
    results = db.similarity_search_with_score(enhanced_query, k=k)
    print(f"Retrieved {len(results)} relevant documents from the database.")
    
    # Check for no-split flag
    no_split_flag = os.path.join(loader.CHROMA_PATH, "no_split")
    if os.path.exists(no_split_flag) and results:
        top_doc, _score = results[0]
        context_text = top_doc.page_content
    else:
        # Concatenate retrieved documents
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    # Light preprocessing: Remove only obvious tables
    print("Preprocessing context to remove data tables...")
    original_length = len(context_text)
    context_text = preprocess_permit_text(context_text)
    removed_percentage = (1 - len(context_text) / original_length) * 100
    print(f"Removed {removed_percentage:.1f}% of text (tables/formatting)")
    
    if verbose:
        print(f"\nPreprocessed context length: {len(context_text)} characters")
        print(f"Context preview:\n{context_text[:800]}...\n")

    # TWO-STAGE PIPELINE
    
    # STAGE 1: Fast extraction with unconstrained schema
    print("\nStage 1: Extracting processes (unconstrained)...")
    
    system_message = """You are an expert in wastewater treatment process identification.

Your task is to extract ONLY the treatment processes that are EXPLICITLY MENTIONED in the facility permit text provided.

CRITICAL RULES:
1. Extract ONLY processes that are directly stated in the text
2. DO NOT infer, assume, or generate processes not explicitly mentioned
3. DO NOT create generic descriptions - use the exact terminology from the text
4. For each process, you MUST quote the exact sentence where it appears
5. If a sentence mentions multiple processes, extract each one separately
6. Ignore administrative information, monitoring requirements, and compliance data
7. Focus on sections describing treatment operations and equipment

Examples of what TO extract:
- "activated sludge biological treatment" → Extract "activated sludge"
- "primary sedimentation" → Extract "primary sedimentation"
- "UV disinfection" → Extract "UV disinfection"

Examples of what NOT to extract:
- If text says "primary sedimentation", do NOT extract "primary treatment" 
- If text says "activated sludge", do NOT extract "biological treatment" unless separately stated
- Do NOT extract processes from your knowledge - only from the text provided"""

    user_message = f"""Facility permit text:

{context_text}

---

Extract all wastewater treatment processes that are explicitly mentioned in the text above.

For each process found:
1. Use the exact name/terminology from the text
2. Provide the complete sentence where it appears (copy it exactly)
3. Give a confidence score (1.0 if explicitly named, lower if ambiguous)
4. Identify category if clear (primary, secondary, tertiary, disinfection, etc.)

Also extract if mentioned:
- Facility name
- Design capacity (e.g., "15.0 MGD")

IMPORTANT: Only extract what is explicitly written in the text. Do not infer or generate."""

    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_message},
    ]

    if verbose:
        print(f"\nPrompt lengths:")
        print(f"  System: {len(system_message)} chars")
        print(f"  User: {len(user_message)} chars")
        print(f"  Total context: {len(context_text)} chars\n")

    try:
        t0 = time.time()
        response = chat(
            model='mistral:7b',
            messages=messages,
            format=WWTPAnalysisUnconstrained.model_json_schema(),
            options={'temperature': 0, 'num_ctx': 8192}
        )
        stage1_time = time.time() - t0
        print(f"Stage 1 completed in {stage1_time:.2f} seconds")

        content = getattr(response, 'message', None)
        content_str = content.content if content is not None else str(response)

        if not content_str or not content_str.strip():
            raise ValueError("Empty response from Stage 1")

        unconstrained_analysis = WWTPAnalysisUnconstrained.model_validate_json(content_str)
        print(f"Stage 1: Extracted {len(unconstrained_analysis.processes)} processes")
        
        if verbose:
            print("\nUnconstrained extractions:")
            for i, proc in enumerate(unconstrained_analysis.processes, 1):
                print(f"{i}. {proc.process_name} (conf: {proc.confidence:.2f})")
                print(f"   Sentence: {proc.match_sentence[:100]}...")
        
    except Exception as e:
        print(f"Error in Stage 1: {e}")
        # Try fallback
        try:
            print("Attempting fallback to plain chat...")
            t1 = time.time()
            fallback = chat(model='mistral:7b', messages=messages, options={'temperature': 0})
            stage1_time = time.time() - t1
            raw = getattr(fallback, 'message', None)
            raw_text = raw.content if raw is not None else str(fallback)
            
            if verbose:
                print("RAW RESPONSE:\n", raw_text[:1000], "...\n")
            
            # Try to parse as JSON
            try:
                parsed = json.loads(raw_text)
                unconstrained_analysis = WWTPAnalysisUnconstrained.model_validate(parsed)
            except:
                # If JSON parsing fails, try to extract JSON from markdown code blocks
                import re
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(1))
                    unconstrained_analysis = WWTPAnalysisUnconstrained.model_validate(parsed)
                else:
                    raise
                    
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            return WWTPAnalysis(processes=[], unknown_processes=[], llm_time_seconds=stage1_time, llm_attempts=1), []

    # STAGE 2: Match to reference processes
    print("\nStage 2: Matching to reference processes...")
    t2 = time.time()
    
    reference_lookup = build_reference_lookup()
    matched_processes, unknown_processes = match_to_reference(
        unconstrained_analysis.processes,
        reference_lookup,
        threshold=80  # Lowered from 85 to catch more variations
    )
    
    stage2_time = time.time() - t2
    print(f"Stage 2 completed in {stage2_time:.2f} seconds")
    print(f"Matched: {len(matched_processes)} processes")
    print(f"Unknown: {len(unknown_processes)} processes")

    # Build final analysis
    wwtp_analysis = WWTPAnalysis(
        facility_name=unconstrained_analysis.facility_name,
        processes=matched_processes,
        unknown_processes=unknown_processes,
        design_capacity=unconstrained_analysis.design_capacity,
        llm_time_seconds=stage1_time + stage2_time,
        llm_attempts=1
    )

    # Verbose output
    if verbose:
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Facility: {wwtp_analysis.facility_name or 'Not specified'}")
        print(f"Design Capacity: {wwtp_analysis.design_capacity or 'Not specified'}")
        print(f"Total Time: {wwtp_analysis.llm_time_seconds:.2f} seconds")
        print(f"\n{'='*60}")
        print(f"MATCHED PROCESSES ({len(wwtp_analysis.processes)}):")
        print(f"{'='*60}\n")
        
        for i, proc in enumerate(wwtp_analysis.processes, 1):
            print(f"{i}. {proc.process_name}")
            print(f"   Category: {proc.category}")
            print(f"   Confidence: {proc.confidence:.2f}")
            if proc.alternative_name_used:
                print(f"   Matched via: '{proc.alternative_name_used}'")
            print(f"   Context: {proc.match_sentence[:150]}...")
            print()
        
        if wwtp_analysis.unknown_processes:
            print(f"\n{'='*60}")
            print(f"UNKNOWN PROCESSES ({len(wwtp_analysis.unknown_processes)}):")
            print(f"{'='*60}\n")
            
            for i, unk in enumerate(wwtp_analysis.unknown_processes, 1):
                print(f"{i}. {unk.process_mentioned}")
                print(f"   Confidence: {unk.confidence:.2f}")
                print(f"   Reason: {unk.reason_not_matched}")
                print(f"   Context: {unk.match_sentence[:150]}...")
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
        
        print("\n" + "="*60)
        print("OPTIMIZED TWO-STAGE PIPELINE")
        print("="*60 + "\n")
        
        analysis, sources = query_rag(query_text, verbose=args.verbose)
        
        print("\n" + "="*60)
        print("JSON OUTPUT")
        print("="*60)
        print(analysis.model_dump_json(indent=2))
        
        if not args.verbose:
            print(f"\nFound {len(analysis.processes)} matched processes")
            print(f"Found {len(analysis.unknown_processes)} unknown processes")
            print(f"Execution time: {analysis.llm_time_seconds:.2f} seconds")
            print(f"\nUse --verbose or -v for detailed output")


if __name__ == "__main__":
    main()