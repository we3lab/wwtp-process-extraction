# LLM Extraction - Treatment Process Extraction System

This directory contains tools for extracting wastewater treatment processes from NPDES permit PDFs using Large Language Models (LLMs).

## 📁 Directory Contents

```
LLM_extraction/
├── treatment_process_extractor.py      # Original FAISS-based approach
├── treatment_process_extractor_rag.py  # New ChromaDB RAG approach ⭐
├── run_extraction.py                   # CLI for original approach
├── compare_approaches.py               # Comparison guide
├── test_rag_setup.py                   # Test script for RAG system
├── requirements_rag.txt                # Python dependencies
├── templates/                          # Example CSV templates
│   ├── permits_metadata_template.csv
│   └── facilities_to_extract_template.csv
└── chroma_permits/                     # ChromaDB database (created on first run)
```

## 🚀 Quick Start (RAG Approach - Recommended)

### 1. Install Dependencies

```bash
# Install Python packages
pip install -r requirements_rag.txt

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull mistral:7b           # LLM for extraction
ollama pull nomic-embed-text     # Embeddings for semantic search

# Start Ollama
ollama serve
```

### 2. Test Your Setup

```bash
python test_rag_setup.py
```

Should see all green checkmarks! ✅

### 3. Index Your Permits

Create `permits_metadata.csv`:
```csv
pdf_path,facility_name,npdes_no
output/2025-10-8-test1/NPDES/permit1.pdf,City WWTP,CA0012345
output/2025-10-8-test1/NPDES/permit2.pdf,County Plant,CA0067890
```

Index PDFs (one-time operation):
```bash
python treatment_process_extractor_rag.py \
    --index output/2025-10-8-test1/NPDES \
    --metadata permits_metadata.csv
```

### 4. Extract Treatment Processes

Create `facilities_to_extract.csv`:
```csv
facility_name,npdes_no
City WWTP,CA0012345
County Plant,CA0067890
```

Run extraction:
```bash
python treatment_process_extractor_rag.py \
    --extract facilities_to_extract.csv \
    --output results.csv
```

Done! Results saved to `results.csv`.

## 📊 Two Approaches Available

### Original Approach (FAISS)
- **File**: `treatment_process_extractor.py`
- **Best for**: Small datasets (<20 permits), one-off analyses
- **Pros**: Simple, self-contained
- **Cons**: Processes entire PDF each time, no persistence

### RAG Approach (ChromaDB) ⭐ Recommended
- **File**: `treatment_process_extractor_rag.py`
- **Best for**: Large datasets (50+ permits), production use
- **Pros**: Persistent database, semantic search, cross-document learning
- **Cons**: Requires initial indexing

See `compare_approaches.py` for detailed comparison.

## 📖 Documentation

- **[RAG_QUICKSTART.md](../../RAG_QUICKSTART.md)** - Complete guide for RAG approach
- **[OLLAMA_QUICKSTART.md](../../OLLAMA_QUICKSTART.md)** - Ollama + LangChain guide
- **[README_TREATMENT_EXTRACTION.md](../../README_TREATMENT_EXTRACTION.md)** - Original approach guide

## 🔧 Configuration

Edit constants in `treatment_process_extractor_rag.py`:

```python
CHUNK_SIZE = 1000              # Characters per chunk
OVERLAP = 200                  # Overlap between chunks
TOP_K_CHUNKS = 5               # Number of chunks to retrieve
N_SIMILAR_EXAMPLES = 2         # Number of example permits for few-shot
DEFAULT_OLLAMA_MODEL = "mistral:7b"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
```

## 💡 Usage Examples

### Index Single PDF
```python
from treatment_process_extractor_rag import PermitVectorStore

vector_store = PermitVectorStore()
vector_store.index_pdf(
    pdf_path="permit.pdf",
    facility_name="Example WWTP",
    npdes_no="CA0012345"
)
```

### Query Database
```python
results = vector_store.query(
    query_text="What disinfection methods are used?",
    k=10  # Top 10 results
)

for chunk in results:
    print(f"{chunk['facility_name']}: {chunk['text'][:100]}...")
```

### Extract Processes
```python
from treatment_process_extractor_rag import TreatmentProcessRAG

rag = TreatmentProcessRAG(
    process_json_path="../data/treatment_processes.json",
    vector_store=vector_store,
    llm_provider="ollama",
    llm_model="mistral:7b"
)

result = rag.extract_processes(
    facility_name="Example WWTP",
    npdes_no="CA0012345"
)

print(result['processes'])
```

## 🎯 Expected Output

```csv
facility_name,npdes_no,n_processes,process_list,confidence_scores,evidence_snippets
City WWTP,CA0012345,8,"preliminary_screening; activated_sludge; secondary_clarification; disinfection_chlorination","0.95; 0.92; 0.90; 0.88","bar screens and grit removal...; aeration basins...; final clarifiers...; sodium hypochlorite..."
```

## 🔍 Troubleshooting

### "sqlite3 version too old"
```bash
pip install pysqlite3-binary
```

### "Could not connect to Ollama"
```bash
ollama serve  # Make sure Ollama is running
ollama list   # Check if models are pulled
```

### "No chunks retrieved"
Check if PDFs are indexed:
```bash
# Should show existing database
ls chroma_permits/
```

### Slow performance
- Reduce `CHUNK_SIZE` for smaller chunks
- Use faster embedding model: `nomic-embed-text` (default)
- Process in smaller batches

## 🎓 Learning Resources

1. **ChromaDB**: https://docs.trychroma.com/
2. **LangChain**: https://python.langchain.com/
3. **Ollama**: https://ollama.com/
4. **RAG Pattern**: https://python.langchain.com/docs/use_cases/question_answering/

## 📝 Data Files

Required data files (in `../data/`):
- `treatment_processes.json` - Process definitions (25 processes)
- `training_chunks.csv` - Training data for relevance scoring
- `example_permits.csv` - Ground truth examples for few-shot learning

Templates provided in `templates/` directory.

## 🚦 Workflow

```
┌──────────────┐
│ Index PDFs   │  (One time, ~10s per PDF)
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│ ChromaDB Database    │  (Persistent on disk)
│ • Semantic chunks    │
│ • Embeddings         │
│ • Metadata           │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Query & Extract      │  (Fast, ~15s per facility)
│ • Semantic search    │
│ • LLM extraction     │
│ • Structured output  │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Results CSV          │
└──────────────────────┘
```

## 🎉 What Makes RAG Better?

1. **Semantic Understanding**: Finds "UV treatment" when you search "disinfection"
2. **Cross-Document Learning**: Learns from ALL permits, not just current one
3. **Persistent Storage**: Index once, query forever
4. **Fast Queries**: Vector search is much faster than processing full PDFs
5. **Incremental Updates**: Add new permits without full rebuild
6. **Metadata Filtering**: Search by facility, region, date, etc.

## 📞 Need Help?

1. Run test script: `python test_rag_setup.py`
2. Check documentation: `RAG_QUICKSTART.md`
3. Review examples: `compare_approaches.py`
4. Verify setup: Make sure Ollama is running (`ollama list`)

---

**Ready to extract treatment processes at scale! 🚀**
