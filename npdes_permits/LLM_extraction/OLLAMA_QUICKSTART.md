# Treatment Process Extraction with Ollama + LangChain

Updated pipeline using **Ollama** (local LLM) and **LangChain** (better chunking/retrieval) for extracting treatment processes from NPDES permits.

## Setup

### 1. Install Ollama

```bash
# On Linux
curl -fsSL https://ollama.com/install.sh | sh

# On macOS
brew install ollama

# Start Ollama service
ollama serve
```

### 2. Pull the Mistral Model

```bash
# Pull mistral:7b (recommended - good balance of speed and accuracy)
ollama pull mistral:7b

# Or try other models:
# ollama pull llama3:8b
# ollama pull phi3:medium
```

### 3. Install Python Dependencies

```bash
# Core dependencies
pip install pandas numpy PyPDF2 pdfminer.six

# Ollama client
pip install ollama

# LangChain for better chunking and retrieval
pip install langchain sentence-transformers faiss-cpu
```

## Quick Start with Ollama

### Process a Single Permit

```bash
python npdes_permits/LLM_extraction/run_extraction.py \
    --pdf output/2025-10-8-test1/NPDES/example.pdf \
    --facility "Example WWTP" \
    --npdes CA0012345 \
    --llm ollama \
    --model mistral:7b
```

### Process Multiple Permits

Create `permits.csv`:
```csv
pdf_path,facility_name,npdes_no
output/NPDES/permit1.pdf,City WWTP,CA0012345
output/NPDES/permit2.pdf,County Plant,CA0067890
```

Run batch processing:
```bash
python npdes_permits/LLM_extraction/run_extraction.py \
    --input permits.csv \
    --output results.csv \
    --llm ollama \
    --model mistral:7b
```

## What Changed: Ollama + LangChain

### LangChain Improvements

**1. Smarter Text Chunking**
- Instead of fixed-size chunks, uses `RecursiveCharacterTextSplitter`
- Tries to break at natural boundaries: paragraphs → sentences → words
- Better preserves context and meaning

**2. Semantic Relevance Scoring**
- Uses sentence transformers for embeddings (`all-MiniLM-L6-v2`)
- FAISS vector store for fast similarity search
- Much better than TF-IDF for finding relevant sections

**Before (TF-IDF)**:
```python
# Keyword-based matching
"activated sludge" matches "sludge" but not "aeration basin"
```

**After (LangChain Embeddings)**:
```python
# Semantic matching
"activated sludge" matches "aeration basin" (same concept!)
```

### Ollama Benefits

✓ **100% Local** - No API costs, no data leaves your machine
✓ **Fast** - Runs on your GPU/CPU, no network latency
✓ **Private** - Sensitive permit data stays local
✓ **Flexible** - Easy to switch models (mistral, llama3, phi3, etc.)

## Model Comparison

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| `mistral:7b` | 4.1GB | Fast | Good | **Recommended** - best balance |
| `llama3:8b` | 4.7GB | Medium | Excellent | Higher accuracy needed |
| `phi3:medium` | 7.9GB | Slow | Excellent | Complex extractions |
| `mistral:instruct` | 4.1GB | Fast | Good | Following instructions strictly |

Try different models:
```bash
# Download and test
ollama pull llama3:8b
python run_extraction.py --model llama3:8b --pdf test.pdf ...
```

## Configuration Options

### Use LangChain (Recommended)

```bash
# Default: LangChain enabled
python run_extraction.py --pdf permit.pdf ...

# Explicitly enable
python run_extraction.py --use-langchain --pdf permit.pdf ...

# Disable LangChain (use basic methods)
python run_extraction.py --no-langchain --pdf permit.pdf ...
```

### Ollama vs OpenAI

```bash
# Ollama (local, free)
python run_extraction.py --llm ollama --model mistral:7b ...

# OpenAI (cloud, paid)
export OPENAI_API_KEY='your-key'
python run_extraction.py --llm openai --model gpt-4 ...
```

## Advanced Python Usage

```python
from treatment_process_extractor import TreatmentExtractionPipeline

# Initialize with Ollama + LangChain
pipeline = TreatmentExtractionPipeline(
    process_json_path='npdes_permits/data/treatment_processes.json',
    training_chunks_path='npdes_permits/data/training_chunks.csv',
    example_database_path='npdes_permits/data/example_permits.csv',
    llm_provider="ollama",
    llm_model="mistral:7b",
    use_langchain=True  # Enable LangChain features
)

# Process permit
result = pipeline.process_permit(
    pdf_path='permit.pdf',
    facility_name='My WWTP',
    npdes_no='CA0012345'
)

# Check results
for proc in result['processes']:
    print(f"{proc['generic_name']}: {proc['confidence']:.2f}")
    print(f"  Evidence: {proc['evidence'][:100]}...")
```

## Performance Tips

### 1. GPU Acceleration (if available)

Ollama automatically uses GPU if available. Check with:
```bash
ollama ps  # Shows running models and GPU usage
nvidia-smi  # Check NVIDIA GPU usage
```

### 2. Batch Processing with Checkpoints

```python
import pandas as pd

permits = pd.read_csv('permits.csv')
results = []

for idx, row in permits.iterrows():
    print(f"[{idx+1}/{len(permits)}] {row['facility_name']}")
    
    try:
        result = pipeline.process_permit(
            row['pdf_path'], 
            row['facility_name'], 
            row['npdes_no']
        )
        results.append(result)
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append({
            'facility_name': row['facility_name'],
            'npdes_no': row['npdes_no'],
            'error': str(e),
            'processes': []
        })
    
    # Save checkpoint every 10 permits
    if (idx + 1) % 10 == 0:
        pipeline.export_results(results, f'checkpoint_{idx+1}.csv')
        print(f"  ✓ Checkpoint saved")

# Final save
pipeline.export_results(results, 'final_results.csv')
```

### 3. Model Context Length

Different models have different context limits:
- `mistral:7b`: ~8K tokens
- `llama3:8b`: ~8K tokens
- `phi3:medium`: ~128K tokens (use for very long documents)

If permits are very long, either:
- Use a model with larger context window
- Increase chunking (extract more chunks)
- Process in multiple passes

## Troubleshooting

### "Could not connect to Ollama"

```bash
# Make sure Ollama is running
ollama serve

# In another terminal:
ollama list  # Check available models
```

### "Model not found"

```bash
# Pull the model first
ollama pull mistral:7b
```

### LangChain imports fail

```bash
# Install all LangChain dependencies
pip install langchain sentence-transformers faiss-cpu

# If you only want to use basic methods:
python run_extraction.py --no-langchain ...
```

### Ollama is slow

```bash
# Check if GPU is being used
ollama ps

# If CPU only, consider:
# 1. Use smaller model: ollama pull mistral:7b
# 2. Reduce chunk count: edit TOP_K_CHUNKS in extractor.py
# 3. Process in batches overnight
```

### Empty results

```bash
# Test the model directly
ollama run mistral:7b "Extract treatment processes from: The plant uses activated sludge."

# If model works, check:
# 1. PDF text extraction: print chunks to see if text is readable
# 2. Training data: add more examples to training_chunks.csv
# 3. Try different model: ollama pull llama3:8b
```

## Comparison: Basic vs LangChain

| Feature | Basic | LangChain |
|---------|-------|-----------|
| Text Chunking | Fixed size | Semantic boundaries |
| Relevance Scoring | TF-IDF keywords | Sentence embeddings |
| Similarity Search | Cosine similarity | FAISS vector store |
| Accuracy | Good | Better |
| Speed | Faster | Slightly slower |
| Dependencies | Minimal | More packages |

**Recommendation**: Use LangChain unless you need minimal dependencies.

## Example Output

```bash
$ python run_extraction.py --pdf permit.pdf --facility "Example WWTP" --npdes CA0012345

================================================================================
Initializing Treatment Process Extraction Pipeline
================================================================================
Using LangChain for intelligent text chunking
Loaded 7 training chunks
Using LangChain embeddings for relevance scoring
Loaded 2 example permits
Using Ollama with model: mistral:7b
✓ Ollama is running
================================================================================

Processing: Example WWTP (CA0012345)
PDF: permit.pdf
Step 1: Extracting and chunking text...
  Created 45 chunks
Step 2: Finding relevant chunks...
  Selected 5 relevant chunks
Step 3: Finding similar example permits...
  Example 1: City WWTP (similarity: 0.872)
  Example 2: County Plant (similarity: 0.814)
Step 4-5: Querying LLM...
  Found 6 processes

✓ Results saved to: output/treatment_extraction/results.csv

RESULTS
================================================================================
Facility: Example WWTP
NPDES No: CA0012345

Processes found:
  - preliminary_screening (confidence: 0.95)
    Evidence: The facility includes bar screens and grit removal...
  - activated_sludge (confidence: 0.92)
    Evidence: Secondary treatment consists of aeration basins...
  - secondary_clarification (confidence: 0.90)
    Evidence: Following aeration, wastewater flows to final clarifiers...
  - disinfection_chlorination (confidence: 0.88)
    Evidence: Disinfection is achieved using sodium hypochlorite...
```

## Next Steps

1. **Test on sample permits** - Run on 5-10 permits to validate
2. **Tune the model** - Try different Ollama models for your use case
3. **Improve training data** - Add more examples to training_chunks.csv
4. **Scale up** - Process your full permit database
5. **Evaluate results** - Compare against manual labels

## Resources

- Ollama: https://ollama.com/
- LangChain: https://python.langchain.com/
- Sentence Transformers: https://www.sbert.net/
- FAISS: https://github.com/facebookresearch/faiss
