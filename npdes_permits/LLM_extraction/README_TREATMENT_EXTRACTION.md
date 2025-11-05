# Treatment Process Extraction from NPDES Permits

LLM-based system to automatically extract wastewater treatment processes from NPDES permit PDFs.

## Workflow

The system follows a 5-step process:

1. **Text Chunking**: Splits PDF text into overlapping chunks (~1000 chars each)
2. **Relevance Scoring**: Identifies chunks most likely to contain treatment process information using TF-IDF similarity to training examples
3. **Similar Example Finding**: Finds 2 most similar permits from a database of labeled examples
4. **LLM Query Construction**: Builds a prompt with:
   - Target permit chunks
   - Similar example permits with their ground truth processes
   - JSON dictionary of all possible treatment processes and alternative names
5. **LLM Execution & Parsing**: Sends to LLM, receives structured JSON response with process names and confidence scores

## Installation

```bash
# Install required packages
pip install pandas numpy scikit-learn PyPDF2 openai pdfminer.six

# Set your OpenAI API key
export OPENAI_API_KEY='your-api-key-here'
```

## Quick Start

### Process a Single Permit

```bash
python npdes_permits/run_extraction.py \
    --pdf output/2025-10-8-test1/NPDES/example.pdf \
    --facility "Example WWTP" \
    --npdes CA0012345 \
    --output results.csv
```

### Process Multiple Permits from CSV

Create an input CSV with columns: `pdf_path`, `facility_name`, `npdes_no`

```bash
python npdes_permits/run_extraction.py \
    --input permits_to_process.csv \
    --output treatment_results.csv
```

## Configuration Files

### 1. Treatment Processes JSON (`data/treatment_processes.json`)

Defines all possible treatment processes with:
- `generic_name`: Standardized name to use in output
- `category`: Process category (preliminary, primary, secondary, etc.)
- `alternative_names`: List of variations the LLM should recognize

Example:
```json
{
  "treatment_processes": [
    {
      "generic_name": "activated_sludge",
      "category": "secondary_biological",
      "alternative_names": [
        "activated sludge",
        "aeration basin",
        "biological treatment"
      ]
    }
  ]
}
```

**You can extend this JSON** with more processes and alternative names as needed.

### 2. Training Chunks CSV (`data/training_chunks.csv`)

Contains example chunks that are relevant (contain process info) vs not relevant.

Columns:
- `chunk_text`: Text excerpt
- `is_relevant`: 1 if contains process info, 0 if not
- `processes_mentioned`: Semicolon-separated list of generic process names

Used for **Step 2** (relevance scoring).

### 3. Example Permits CSV (`data/example_permits.csv`)

Contains permits with known ground truth process lists.

Columns:
- `pdf_path`: Path to PDF file
- `facility_name`: Facility name
- `npdes_no`: NPDES number
- `processes`: JSON array of generic process names (ground truth)
- `relevant_excerpt`: Short excerpt showing where processes are mentioned

Used for **Step 3** (finding similar examples) and **Step 4** (LLM prompt examples).

## Output Format

The system produces a CSV with one row per facility:

| Column | Description |
|--------|-------------|
| `facility_name` | Facility name |
| `npdes_no` | NPDES permit number |
| `pdf_path` | Path to PDF file |
| `n_processes` | Number of processes found |
| `process_list` | Semicolon-separated list of generic process names |
| `confidence_scores` | Semicolon-separated confidence scores (0-1) |
| `evidence_snippets` | Pipe-separated text excerpts supporting each process |
| `error` | Error message if processing failed |

Example row:
```
Example WWTP, CA0012345, example.pdf, 6, 
"preliminary_screening; grit_removal; activated_sludge; secondary_clarification; disinfection_uv", 
"0.95; 0.92; 0.98; 0.95; 0.88",
"bar screens and grit chambers | aeration basins | final clarifiers | UV disinfection",
""
```

## Customization

### Add More Training Data

Edit `data/training_chunks.csv` to add more examples of relevant vs irrelevant chunks. The more examples, the better the relevance scoring.

### Add More Example Permits

Edit `data/example_permits.csv` to add more ground truth examples. These provide few-shot learning examples to the LLM.

### Add More Processes

Edit `data/treatment_processes.json` to add new processes or alternative names. The LLM will use these to standardize its output.

### Adjust Chunking

In `treatment_process_extractor.py`, modify:
```python
CHUNK_SIZE = 1000  # characters per chunk
OVERLAP = 200      # overlap between chunks
TOP_K_CHUNKS = 5   # number of chunks to send to LLM
```

### Change LLM Model

In `run_extraction.py` or when calling the API, change:
```python
model="gpt-4"  # or "gpt-4-turbo", "gpt-3.5-turbo", etc.
```

## Advanced Usage

### Use as Python Library

```python
from treatment_process_extractor import TreatmentExtractionPipeline

pipeline = TreatmentExtractionPipeline(
    process_json_path='data/treatment_processes.json',
    training_chunks_path='data/training_chunks.csv',
    example_database_path='data/example_permits.csv'
)

result = pipeline.process_permit(
    pdf_path='path/to/permit.pdf',
    facility_name='My WWTP',
    npdes_no='CA0012345'
)

print(result['processes'])
```

### Batch Processing with Progress

```python
import pandas as pd

permits = pd.read_csv('permits.csv')
results = []

for idx, row in permits.iterrows():
    print(f"Processing {idx+1}/{len(permits)}: {row['facility_name']}")
    result = pipeline.process_permit(
        row['pdf_path'],
        row['facility_name'],
        row['npdes_no']
    )
    results.append(result)
    
    # Save intermediate results
    if (idx + 1) % 10 == 0:
        pipeline.export_results(results, f'results_checkpoint_{idx+1}.csv')
```

## Evaluation

To evaluate the system against ground truth:

```python
import pandas as pd

# Load results and ground truth
results = pd.read_csv('results.csv')
ground_truth = pd.read_csv('ground_truth.csv')

# Compare process lists
for idx, row in results.iterrows():
    predicted = set(row['process_list'].split('; '))
    actual = set(ground_truth.loc[ground_truth['npdes_no'] == row['npdes_no'], 'processes'].iloc[0].split('; '))
    
    precision = len(predicted & actual) / len(predicted) if predicted else 0
    recall = len(predicted & actual) / len(actual) if actual else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"{row['facility_name']}: P={precision:.2f}, R={recall:.2f}, F1={f1:.2f}")
```

## Troubleshooting

### "No relevant chunks found"
- Add more training examples to `training_chunks.csv`
- Lower the relevance threshold
- Increase `TOP_K_CHUNKS` to include more chunks

### "LLM returns empty process list"
- Check if PDF text extraction worked (print the chunks)
- Verify API key is set correctly
- Add more similar examples to the database
- Try a more powerful model (gpt-4 vs gpt-3.5)

### "JSON parse error"
- The LLM sometimes returns malformed JSON
- Retry with lower temperature (increase consistency)
- Add more explicit formatting instructions in prompt

## Files Structure

```
LLM_extraction/
├── treatment_process_extractor.py  # Main pipeline code
├── run_extraction.py               # Command-line runner
├── data/
│   ├── treatment_processes.json    # Process definitions
│   ├── training_chunks.csv         # Training data for relevance
│   └── example_permits.csv         # Example permits with ground truth
└── output/
    └── treatment_extraction/
        └── results.csv              # Output results
```

## Next Steps

1. **Collect Training Data**: Label 50-100 chunks as relevant/not relevant
2. **Build Ground Truth Database**: Manually label 20-30 permits with their processes
3. **Run on Test Set**: Process unlabeled permits
4. **Evaluate & Iterate**: Compare against manual labels, improve process JSON and examples
5. **Scale Up**: Process entire permit database

## License

MIT
