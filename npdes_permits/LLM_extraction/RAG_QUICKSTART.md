# Treatment Process Extraction with ChromaDB RAG

**Improved approach using Retrieval Augmented Generation (RAG)**

This system uses ChromaDB for persistent vector storage and semantic search to find the most relevant sections of permits before sending to the LLM.

## Why RAG is Better

### Previous Approach (FAISS)
- ✗ In-memory only (rebuilt each time)
- ✗ Sequential processing (chunk entire PDF)
- ✗ Limited to single document context

### New Approach (ChromaDB RAG)
- ✓ **Persistent database** - index once, query many times
- ✓ **Semantic search** - find relevant sections across ALL permits
- ✓ **Cross-document learning** - can retrieve examples from other permits
- ✓ **Incremental updates** - add new permits without reindexing everything
- ✓ **Metadata filtering** - search by facility, region, etc.

## Architecture

```
┌─────────────────┐
│ Index Phase     │
│ (Run once)      │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  ChromaDB Vector Store                  │
│  • All permit PDFs chunked              │
│  • Semantic embeddings (nomic-embed)    │
│  • Metadata (facility, NPDES, etc.)     │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ Query Phase     │
│ (Per facility)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  1. Semantic Search                     │
│     → Retrieve top K relevant chunks    │
│  2. Few-Shot Examples                   │
│     → Get ground truth examples         │
│  3. LLM Extraction                      │
│     → Query Ollama with context         │
│  4. Structured Output                   │
│     → Parse JSON results                │
└─────────────────────────────────────────┘
```

## Setup

### 1. Install Dependencies

```bash
# Core
pip install pandas numpy PyPDF2 pdfminer.six

# Ollama
pip install ollama

# LangChain + ChromaDB
pip install langchain langchain-community chromadb

# If you get sqlite3 version errors:
pip install pysqlite3-binary
```

### 2. Install and Start Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull models
ollama pull mistral:7b           # LLM for extraction
ollama pull nomic-embed-text     # Embeddings for RAG

# Start Ollama (keep running in background)
ollama serve
```

## Quick Start Guide

### Step 1: Index Your PDFs

First, create a CSV with your permit metadata (`permits_metadata.csv`):

```csv
pdf_path,facility_name,npdes_no
output/2025-10-8-test1/NPDES/permit1.pdf,City WWTP,CA0012345
output/2025-10-8-test1/NPDES/permit2.pdf,County Plant,CA0067890
```

Then index all PDFs into ChromaDB (only need to do this once):

```bash
python treatment_process_extractor_rag.py \
    --index output/2025-10-8-test1/NPDES \
    --metadata permits_metadata.csv
```

Output:
```
Vector store initialized with nomic-embed-text embeddings

Indexing 15 PDFs from output/2025-10-8-test1/NPDES
[1/15] City WWTP (CA0012345)
  Added 87 chunks to database
[2/15] County Plant (CA0067890)
  Added 92 chunks to database
...
✓ Indexed 1,234 total chunks from 15 PDFs
```

### Step 2: Extract Treatment Processes

Create a CSV with facilities to process (`facilities_to_extract.csv`):

```csv
facility_name,npdes_no
City WWTP,CA0012345
County Plant,CA0067890
```

Run extraction:

```bash
python treatment_process_extractor_rag.py \
    --extract facilities_to_extract.csv \
    --output results.csv \
    --processes-json ../data/treatment_processes.json \
    --examples ../data/example_permits.csv
```

Output:
```
Using Ollama with model: mistral:7b
✓ Ollama is running
Loaded 5 example permits for few-shot learning
================================================================================

Extracting processes: City WWTP (CA0012345)
Step 1: Retrieving relevant chunks from vector store...
  Retrieved 5 relevant chunks
Step 2: Finding similar example permits...
  Using 2 example permits
Step 3: Building prompt and querying LLM...
  Found 8 processes

...

✓ Results saved to: results.csv
```

### Step 3: Review Results

The output CSV contains:

| facility_name | npdes_no | n_processes | process_list | confidence_scores | evidence_snippets |
|---------------|----------|-------------|--------------|-------------------|-------------------|
| City WWTP | CA0012345 | 8 | preliminary_screening; activated_sludge; ... | 0.95; 0.92; ... | "bar screens and grit removal..."; "aeration basins..." |

## Advanced Usage

### Update Database with New Permits

Add new permits without clearing existing database:

```bash
# Index new PDFs (they'll be added to existing database)
python treatment_process_extractor_rag.py \
    --index output/2025-11-1/NPDES \
    --metadata new_permits.csv
```

ChromaDB automatically:
- ✓ Skips duplicates (checks by NPDES:chunk_id)
- ✓ Adds only new chunks
- ✓ Persists to disk

### Clear and Rebuild Database

```bash
# Clear database
python treatment_process_extractor_rag.py --reset

# Re-index everything
python treatment_process_extractor_rag.py \
    --index output/all_permits \
    --metadata all_metadata.csv
```

### Query Database Directly (Python)

```python
from treatment_process_extractor_rag import PermitVectorStore

# Initialize
vector_store = PermitVectorStore()

# Query across all permits
results = vector_store.query(
    query_text="What disinfection methods are used?",
    k=10  # Top 10 most relevant chunks
)

for chunk in results:
    print(f"Facility: {chunk['facility_name']}")
    print(f"Score: {chunk['score']:.3f}")
    print(f"Text: {chunk['text'][:200]}...\n")
```

### Filter by Facility

```python
# Query only specific facility
results = vector_store.query(
    query_text="secondary treatment processes",
    k=5,
    filter_facility="City WWTP"
)
```

### Custom Embedding Model

```bash
# Use different Ollama embedding model
ollama pull mxbai-embed-large

python treatment_process_extractor_rag.py \
    --embedding-model mxbai-embed-large \
    --index permits/ \
    --metadata metadata.csv
```

### Custom LLM Model

```bash
# Use different Ollama LLM
ollama pull llama3:8b

python treatment_process_extractor_rag.py \
    --extract facilities.csv \
    --llm ollama \
    --model llama3:8b \
    --output results.csv
```

## Python API

```python
from treatment_process_extractor_rag import PermitVectorStore, TreatmentProcessRAG

# Initialize vector store
vector_store = PermitVectorStore(
    chroma_path="chroma_permits",
    embedding_model="nomic-embed-text"
)

# Index PDFs (one time)
vector_store.index_pdf(
    pdf_path="permit.pdf",
    facility_name="Example WWTP",
    npdes_no="CA0012345"
)

# Initialize RAG pipeline
rag = TreatmentProcessRAG(
    process_json_path="data/treatment_processes.json",
    vector_store=vector_store,
    example_database_path="data/example_permits.csv",
    llm_provider="ollama",
    llm_model="mistral:7b"
)

# Extract processes
result = rag.extract_processes(
    facility_name="Example WWTP",
    npdes_no="CA0012345",
    query_text="What treatment processes are used?"  # Optional
)

# Check results
print(f"Found {len(result['processes'])} processes:")
for proc in result['processes']:
    print(f"  - {proc['generic_name']}: {proc['confidence']:.2f}")
    print(f"    Evidence: {proc['evidence'][:100]}...")
```

## Comparison: Original vs RAG

| Feature | Original (FAISS) | New (ChromaDB RAG) |
|---------|------------------|---------------------|
| Storage | In-memory | Persistent disk |
| Index time | Every run | Once |
| Query speed | Slow (full scan) | Fast (vector search) |
| Context | Single PDF | All PDFs |
| Updates | Full rebuild | Incremental |
| Metadata | Limited | Rich filtering |
| Scalability | <100 PDFs | 1000s of PDFs |

## Performance Tips

### 1. Chunk Size Tuning

```python
# For shorter permits (< 20 pages)
vector_store = PermitVectorStore(
    chunk_size=800,
    chunk_overlap=80
)

# For longer permits (> 50 pages)
vector_store = PermitVectorStore(
    chunk_size=1200,
    chunk_overlap=150
)
```

### 2. Retrieval Tuning

```python
# Retrieve more chunks for complex facilities
rag.extract_processes(
    facility_name="Complex WWTP",
    npdes_no="CA0012345"
)
# Modify TOP_K_CHUNKS in code (default: 5)
```

### 3. Batch Processing with Checkpoints

```python
# Automatically saves every 10 permits
rag.process_batch('all_facilities.csv', 'results.csv')
# Creates: results_checkpoint_10.csv, results_checkpoint_20.csv, etc.
```

## Troubleshooting

### "sqlite3 version too old"

```bash
pip install pysqlite3-binary
```

The code automatically upgrades sqlite3 if needed.

### "Could not connect to Ollama"

```bash
# Make sure Ollama is running
ollama serve

# Check if models are pulled
ollama list

# Should see:
# mistral:7b
# nomic-embed-text
```

### "ChromaDB: no such table"

Database might be corrupted. Clear and rebuild:

```bash
python treatment_process_extractor_rag.py --reset
python treatment_process_extractor_rag.py --index permits/ --metadata metadata.csv
```

### Slow indexing

- **Reduce chunk size**: Fewer, smaller chunks
- **Use faster embedding model**: `nomic-embed-text` (default) is fast
- **Batch process**: Index 10-20 PDFs at a time

### Empty results

1. **Check database**: Query directly to see what's indexed
2. **Verify embeddings**: Make sure `nomic-embed-text` is pulled
3. **Inspect chunks**: Print retrieved chunks to see if relevant
4. **Tune query**: Try different query text

## Example Workflow

### Complete End-to-End

```bash
# 1. Start Ollama
ollama serve &

# 2. Pull models
ollama pull mistral:7b
ollama pull nomic-embed-text

# 3. Clear old database (if needed)
python treatment_process_extractor_rag.py --reset

# 4. Index all permits
python treatment_process_extractor_rag.py \
    --index output/2025-10-8-test1/NPDES \
    --metadata permits_metadata.csv

# 5. Extract treatment processes
python treatment_process_extractor_rag.py \
    --extract facilities_to_extract.csv \
    --output treatment_results.csv \
    --llm ollama \
    --model mistral:7b

# 6. Review results
head -20 treatment_results.csv
```

## What's Next?

1. **Tune retrieval**: Experiment with `TOP_K_CHUNKS` (5-10)
2. **Add ground truth**: Populate `example_permits.csv` with manual labels
3. **Fine-tune prompts**: Edit `build_prompt()` in code
4. **Scale up**: Index all permits, process in batches
5. **Evaluate**: Compare against manual labels

## Key Advantages

✅ **No re-processing**: Index once, query forever  
✅ **Cross-document context**: Learn from all permits  
✅ **Incremental updates**: Add new permits easily  
✅ **Fast queries**: Vector search is much faster  
✅ **Better context**: Semantic search finds relevant sections  
✅ **Metadata filtering**: Search by facility, region, date, etc.

This RAG approach is **production-ready** and can scale to thousands of permits! 🚀
