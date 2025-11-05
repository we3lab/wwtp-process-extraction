"""
Comparison: Original vs RAG approach for treatment process extraction

This script demonstrates the key differences between the two approaches.
"""

print("""
================================================================================
                    TREATMENT PROCESS EXTRACTION
            Original Approach vs RAG (ChromaDB) Approach
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                         ORIGINAL APPROACH (FAISS)                           │
└─────────────────────────────────────────────────────────────────────────────┘

Workflow for EACH permit:
  1. Read PDF → Extract text
  2. Chunk text (sliding window, 1000 chars)
  3. Score all chunks with TF-IDF vs training data
  4. Select top 5 chunks
  5. Find 2 similar permits (TF-IDF across example database)
  6. Build prompt with chunks + examples
  7. Query LLM (Ollama)
  8. Parse JSON response

Pros:
  ✓ Self-contained (no external database)
  ✓ Simple to understand
  ✓ Works with LangChain or basic TF-IDF

Cons:
  ✗ Processes ENTIRE PDF every time
  ✗ No memory between runs
  ✗ Can't learn from all permits
  ✗ Slow for large datasets (sequential processing)
  ✗ TF-IDF is keyword-based (not semantic)

Best for:
  • Small datasets (< 20 permits)
  • One-off analyses
  • When you want simple, portable code

Usage:
  python run_extraction.py \\
      --pdf permit.pdf \\
      --facility "City WWTP" \\
      --npdes CA0012345


┌─────────────────────────────────────────────────────────────────────────────┐
│                      NEW RAG APPROACH (ChromaDB)                            │
└─────────────────────────────────────────────────────────────────────────────┘

Workflow (TWO PHASES):

PHASE 1: INDEX (run once)
  1. Read ALL PDFs → Extract text
  2. Chunk text (semantic boundaries)
  3. Generate embeddings (nomic-embed-text)
  4. Store in ChromaDB with metadata
  → Database persists to disk

PHASE 2: QUERY (per facility)
  1. Build semantic query
  2. Vector search across ALL permits in database
  3. Retrieve top 5 most relevant chunks (from any permit!)
  4. Find 2 similar permits from examples
  5. Build prompt with retrieved chunks + examples
  6. Query LLM (Ollama)
  7. Parse JSON response

Pros:
  ✓ Index once, query many times
  ✓ Semantic search (understands meaning, not just keywords)
  ✓ Cross-document learning (finds relevant sections from ALL permits)
  ✓ Persistent database (fast subsequent queries)
  ✓ Incremental updates (add new permits without full rebuild)
  ✓ Metadata filtering (search by facility, region, etc.)
  ✓ Scales to 1000s of permits

Cons:
  ✗ More dependencies (ChromaDB)
  ✗ Requires index phase before querying
  ✗ Database takes disk space (~1-2MB per PDF)

Best for:
  • Large datasets (100+ permits)
  • Production systems
  • When you'll query multiple times
  • When you want cross-document context

Usage:
  # Index once
  python treatment_process_extractor_rag.py \\
      --index permits/ \\
      --metadata metadata.csv
  
  # Query many times (fast!)
  python treatment_process_extractor_rag.py \\
      --extract facilities.csv \\
      --output results.csv


┌─────────────────────────────────────────────────────────────────────────────┐
│                        CONCRETE EXAMPLE                                     │
└─────────────────────────────────────────────────────────────────────────────┘

Scenario: Extract processes from 100 permits

ORIGINAL APPROACH:
  • Time per permit: ~30 seconds
  • Total time: 100 * 30s = 50 minutes
  • Re-run all 100?: Another 50 minutes
  
  Process:
    For each of 100 permits:
      → Read PDF (5s)
      → Chunk text (2s)
      → Score chunks with TF-IDF (3s)
      → Find similar permits (5s)
      → Query LLM (15s)
      → Total: 30s
    
    If you want to re-process 1 permit with better prompt:
      → Must re-run that permit (30s)

RAG APPROACH:
  • Index time (one-time): 100 * 10s = 17 minutes
  • Query time per permit: ~15 seconds
  • Total first run: 17 + (100 * 15s) = 42 minutes
  • Re-run all 100?: 100 * 15s = 25 minutes (60% faster!)
  • Re-run 1 permit: 15 seconds (instant!)
  
  Process:
    ONCE:
      → Index all 100 permits (17 min)
      → Database saved to disk
    
    THEN (unlimited times):
      For each permit:
        → Vector search in database (1s)
        → Query LLM (15s)
        → Total: 16s
      
      If you improve your prompt:
        → Re-query all 100 permits: 25 minutes
        → Database already has all text indexed!


┌─────────────────────────────────────────────────────────────────────────────┐
│                      SEMANTIC SEARCH EXAMPLE                                │
└─────────────────────────────────────────────────────────────────────────────┘

Query: "What disinfection methods are used?"

ORIGINAL (TF-IDF):
  Finds chunks with keywords:
    ✓ "disinfection" ← direct match
    ✓ "chlorination" ← might match if in training data
    ✗ "UV treatment" ← missed (different words)
    ✗ "sodium hypochlorite" ← missed (technical term)
  
  Result: Finds 3/10 relevant sections

RAG (Semantic Embeddings):
  Finds chunks with similar MEANING:
    ✓ "disinfection" ← direct match
    ✓ "chlorination" ← semantically related
    ✓ "UV treatment" ← understands it's disinfection
    ✓ "sodium hypochlorite" ← knows it's a disinfectant
    ✓ "ultraviolet irradiation" ← semantic match
    ✓ "chlorine contact chamber" ← context-aware
  
  Result: Finds 9/10 relevant sections


┌─────────────────────────────────────────────────────────────────────────────┐
│                          RECOMMENDATION                                     │
└─────────────────────────────────────────────────────────────────────────────┘

Use ORIGINAL if:
  • < 20 permits
  • One-time analysis
  • Simple deployment (fewer dependencies)
  • Quick prototyping

Use RAG if:
  • 50+ permits
  • Will query multiple times
  • Want best accuracy (semantic search)
  • Want cross-document learning
  • Production system

For your NPDES permits project:
  → I recommend RAG! 
  
  You likely have 100+ permits and will:
    • Re-run with improved prompts
    • Add new permits over time
    • Query specific facilities
    • Need best accuracy
  
  The upfront indexing cost pays off quickly!


┌─────────────────────────────────────────────────────────────────────────────┐
│                          GETTING STARTED                                    │
└─────────────────────────────────────────────────────────────────────────────┘

Try RAG in 3 steps:

1. Install dependencies:
   pip install ollama langchain langchain-community chromadb pysqlite3-binary

2. Index your permits:
   python npdes_permits/LLM_extraction/treatment_process_extractor_rag.py \\
       --index output/2025-10-8-test1/NPDES \\
       --metadata permits_metadata.csv

3. Extract processes:
   python npdes_permits/LLM_extraction/treatment_process_extractor_rag.py \\
       --extract facilities_to_extract.csv \\
       --output results_rag.csv

Then compare results with original approach!

================================================================================
                    See RAG_QUICKSTART.md for full guide
================================================================================
""")
