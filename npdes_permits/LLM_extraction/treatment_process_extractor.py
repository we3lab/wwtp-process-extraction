"""
LLM-based Treatment Process Extraction from NPDES Permits

Workflow:
1. Chunk PDF text into manageable segments
2. Find relevant chunks using training set comparison
3. Find 2 similar example permits from test database
4. Create LLM query with examples and process JSON
5. Execute LLM query and parse results
6. Export to CSV
"""

import os
import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import PyPDF2

# LLM options: Ollama (local) or OpenAI
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Warning: ollama not installed. Install with: pip install ollama")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# LangChain for better chunking and retrieval
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.docstore.document import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: langchain not installed. Install with: pip install langchain sentence-transformers faiss-cpu")
    # Fallback to sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

# Configure paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output" / "treatment_extraction"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
CHUNK_SIZE = 1000  # characters per chunk
OVERLAP = 200  # character overlap between chunks
TOP_K_CHUNKS = 5  # number of relevant chunks to extract
N_SIMILAR_EXAMPLES = 2  # number of similar permits to use as examples

# LLM Configuration
DEFAULT_LLM_PROVIDER = "ollama"  # "ollama" or "openai"
DEFAULT_OLLAMA_MODEL = "mistral:7b"  # or "llama3", "phi3", etc.
DEFAULT_OPENAI_MODEL = "gpt-4"


class PDFChunker:
    """Extract and chunk text from PDF files using LangChain or basic method."""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP, use_langchain: bool = LANGCHAIN_AVAILABLE):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.use_langchain = use_langchain and LANGCHAIN_AVAILABLE
        
        if self.use_langchain:
            # LangChain's RecursiveCharacterTextSplitter for better semantic chunking
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""],  # Try to break at natural boundaries
                is_separator_regex=False
            )
            print("Using LangChain for intelligent text chunking")
        else:
            print("Using basic text chunking (install langchain for better results)")
    
    def extract_text(self, pdf_path: str) -> str:
        """Extract all text from a PDF file."""
        text_parts = []
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    try:
                        page_text = page.extract_text() or ''
                        text_parts.append(page_text)
                    except Exception as e:
                        print(f"Error extracting page: {e}")
                        continue
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            # Try pdfminer fallback
            try:
                from pdfminer.high_level import extract_text
                return extract_text(pdf_path)
            except Exception:
                return ""
        
        return '\n'.join(text_parts)
    
    def chunk_text(self, text: str) -> List[Dict[str, any]]:
        """Split text into chunks using LangChain or basic method."""
        if self.use_langchain:
            # Use LangChain's intelligent splitting
            text_chunks = self.text_splitter.split_text(text)
            chunks = []
            current_pos = 0
            for chunk_id, chunk_text in enumerate(text_chunks):
                # Find approximate position in original text
                start = text.find(chunk_text[:50], current_pos) if len(chunk_text) > 50 else current_pos
                if start == -1:
                    start = current_pos
                end = start + len(chunk_text)
                
                chunks.append({
                    'id': chunk_id,
                    'text': chunk_text.strip(),
                    'start': start,
                    'end': end
                })
                current_pos = end
            return chunks
        else:
            # Basic chunking (original method)
            return self._basic_chunk_text(text)
    
    def _basic_chunk_text(self, text: str) -> List[Dict[str, any]]:
        """Basic chunking method (fallback)."""
        chunks = []
        text_length = len(text)
        start = 0
        chunk_id = 0
        
        while start < text_length:
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Try to break at sentence boundary
            if end < text_length:
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n')
                break_point = max(last_period, last_newline)
                if break_point > self.chunk_size * 0.7:  # at least 70% of chunk
                    end = start + break_point + 1
                    chunk_text = text[start:end]
            
            chunks.append({
                'id': chunk_id,
                'text': chunk_text.strip(),
                'start': start,
                'end': end
            })
            
            chunk_id += 1
            start = end - self.overlap
        
        return chunks


class RelevanceScorer:
    """Find relevant chunks using training examples with LangChain or TF-IDF."""
    
    def __init__(self, training_chunks: List[str], use_langchain: bool = LANGCHAIN_AVAILABLE):
        """
        Initialize with training examples of relevant chunks.
        
        Args:
            training_chunks: List of text chunks that are known to contain
                           treatment process information
            use_langchain: Whether to use LangChain embeddings (better) or TF-IDF
        """
        self.training_chunks = training_chunks
        self.use_langchain = use_langchain and LANGCHAIN_AVAILABLE
        
        if self.use_langchain and training_chunks:
            # Use HuggingFace embeddings for semantic similarity
            print("Using LangChain embeddings for relevance scoring")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",  # Fast and efficient
                model_kwargs={'device': 'cpu'}
            )
            
            # Create FAISS vector store with training examples
            training_docs = [Document(page_content=chunk) for chunk in training_chunks]
            self.vectorstore = FAISS.from_documents(training_docs, self.embeddings)
        elif training_chunks:
            # Fallback to TF-IDF
            print("Using TF-IDF for relevance scoring (install langchain for better results)")
            self.vectorizer = TfidfVectorizer(
                max_features=500,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.training_vectors = self.vectorizer.fit_transform(training_chunks)
        else:
            self.vectorstore = None
            self.training_vectors = None
    
    def score_chunks(self, chunks: List[Dict[str, any]], top_k: int = TOP_K_CHUNKS) -> List[Dict[str, any]]:
        """
        Score chunks by similarity to training examples.
        
        Returns:
            List of top_k most relevant chunks with scores
        """
        if not chunks:
            return chunks[:top_k]
        
        if self.use_langchain and self.vectorstore:
            return self._score_with_langchain(chunks, top_k)
        elif self.training_vectors is not None:
            return self._score_with_tfidf(chunks, top_k)
        else:
            return chunks[:top_k]
    
    def _score_with_langchain(self, chunks: List[Dict[str, any]], top_k: int) -> List[Dict[str, any]]:
        """Score using LangChain embeddings and FAISS similarity search."""
        for chunk in chunks:
            # Query vector store for similarity to training examples
            results = self.vectorstore.similarity_search_with_score(chunk['text'], k=1)
            if results:
                # FAISS returns (document, distance) - lower distance = more similar
                # Convert distance to similarity score (0-1 range)
                distance = results[0][1]
                similarity = 1 / (1 + distance)  # Convert to 0-1 range
                chunk['relevance_score'] = float(similarity)
            else:
                chunk['relevance_score'] = 0.0
        
        # Sort by score and return top_k
        sorted_chunks = sorted(chunks, key=lambda x: x['relevance_score'], reverse=True)
        return sorted_chunks[:top_k]
    
    def _score_with_tfidf(self, chunks: List[Dict[str, any]], top_k: int) -> List[Dict[str, any]]:
        """Score using TF-IDF (fallback method)."""
        chunk_texts = [c['text'] for c in chunks]
        chunk_vectors = self.vectorizer.transform(chunk_texts)
        
        # Calculate similarity to each training example
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(chunk_vectors, self.training_vectors)
        
        # Use max similarity across all training examples
        max_similarities = similarities.max(axis=1)
        
        # Add scores to chunks
        for i, chunk in enumerate(chunks):
            chunk['relevance_score'] = float(max_similarities[i])
        
        # Sort by score and return top_k
        sorted_chunks = sorted(chunks, key=lambda x: x['relevance_score'], reverse=True)
        return sorted_chunks[:top_k]


class SimilarPermitFinder:
    """Find similar permits from test database."""
    
    def __init__(self, permit_database: pd.DataFrame):
        """
        Initialize with database of permits.
        
        Args:
            permit_database: DataFrame with columns: pdf_path, facility_name, 
                           npdes_no, extracted_text (optional), processes (ground truth)
        """
        self.database = permit_database
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        
        # Vectorize all permits
        if 'extracted_text' in permit_database.columns:
            texts = permit_database['extracted_text'].fillna('')
        else:
            # Extract text on demand
            texts = []
            chunker = PDFChunker()
            for pdf_path in permit_database['pdf_path']:
                text = chunker.extract_text(pdf_path)
                texts.append(text)
            permit_database['extracted_text'] = texts
        
        self.permit_vectors = self.vectorizer.fit_transform(texts)
    
    def find_similar(self, query_text: str, n: int = N_SIMILAR_EXAMPLES) -> List[Dict[str, any]]:
        """
        Find n most similar permits to query text.
        
        Returns:
            List of dicts with permit info and ground truth processes
        """
        query_vector = self.vectorizer.transform([query_text])
        similarities = cosine_similarity(query_vector, self.permit_vectors)[0]
        
        # Get top n indices
        top_indices = np.argsort(similarities)[-n:][::-1]
        
        results = []
        for idx in top_indices:
            permit_info = self.database.iloc[idx].to_dict()
            permit_info['similarity_score'] = float(similarities[idx])
            results.append(permit_info)
        
        return results


class TreatmentProcessLLM:
    """LLM-based treatment process extraction using Ollama or OpenAI."""
    
    def __init__(self, 
                 process_json_path: str, 
                 llm_provider: str = DEFAULT_LLM_PROVIDER,
                 model: str = None,
                 api_key: Optional[str] = None):
        """
        Initialize with process definitions.
        
        Args:
            process_json_path: Path to JSON file with process definitions
            llm_provider: "ollama" or "openai"
            model: Model name (e.g., "mistral:7b" for Ollama, "gpt-4" for OpenAI)
            api_key: OpenAI API key (only needed if llm_provider="openai")
        """
        with open(process_json_path, 'r') as f:
            self.process_definitions = json.load(f)
        
        self.llm_provider = llm_provider.lower()
        
        # Set default models
        if model is None:
            if self.llm_provider == "ollama":
                self.model = DEFAULT_OLLAMA_MODEL
            else:
                self.model = DEFAULT_OPENAI_MODEL
        else:
            self.model = model
        
        # Initialize LLM client
        if self.llm_provider == "ollama":
            if not OLLAMA_AVAILABLE:
                raise ImportError("Ollama not installed. Install with: pip install ollama")
            print(f"Using Ollama with model: {self.model}")
            # Check if Ollama is running
            try:
                ollama.list()
                print("✓ Ollama is running")
            except Exception as e:
                print(f"Warning: Could not connect to Ollama. Make sure it's running: {e}")
        
        elif self.llm_provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI not installed. Install with: pip install openai")
            if api_key:
                openai.api_key = api_key
            else:
                openai.api_key = os.getenv('OPENAI_API_KEY')
            print(f"Using OpenAI with model: {self.model}")
        
        else:
            raise ValueError(f"Unknown LLM provider: {llm_provider}. Use 'ollama' or 'openai'")
    
    def build_prompt(self, 
                    relevant_chunks: List[Dict[str, any]], 
                    similar_examples: List[Dict[str, any]]) -> str:
        """
        Build LLM prompt with examples and process definitions.
        
        Args:
            relevant_chunks: Chunks from target permit
            similar_examples: Similar permits with ground truth processes
        
        Returns:
            Complete prompt string
        """
        prompt_parts = []
        
        # System instruction
        prompt_parts.append("""You are an expert at extracting wastewater treatment process information from NPDES permit documents.

Your task is to identify all treatment processes mentioned in the provided permit text and return them using the standardized process names from the provided JSON dictionary.

Instructions:
1. Read the permit text carefully
2. Identify all treatment processes mentioned
3. Map each process to its standardized name from the JSON (use alternative names to help identify)
4. Return ONLY the standardized generic names as a JSON array
5. If uncertain, include the process with a confidence score

""")
        
        # Process definitions
        prompt_parts.append("=== TREATMENT PROCESS DEFINITIONS ===\n")
        prompt_parts.append(json.dumps(self.process_definitions, indent=2))
        prompt_parts.append("\n\n")
        
        # Examples
        if similar_examples:
            prompt_parts.append("=== EXAMPLES FROM SIMILAR PERMITS ===\n\n")
            for i, example in enumerate(similar_examples, 1):
                prompt_parts.append(f"Example {i}:\n")
                prompt_parts.append(f"Facility: {example.get('facility_name', 'Unknown')}\n")
                prompt_parts.append(f"NPDES No: {example.get('npdes_no', 'Unknown')}\n")
                
                # Excerpt from example permit
                if 'relevant_excerpt' in example:
                    prompt_parts.append(f"Permit Text Excerpt:\n{example['relevant_excerpt'][:500]}...\n")
                
                # Ground truth processes
                if 'processes' in example:
                    processes = example['processes']
                    if isinstance(processes, str):
                        processes = json.loads(processes)
                    prompt_parts.append(f"Identified Processes: {json.dumps(processes)}\n")
                
                prompt_parts.append("\n")
        
        # Target permit chunks
        prompt_parts.append("=== TARGET PERMIT TEXT ===\n\n")
        for chunk in relevant_chunks:
            prompt_parts.append(f"[Chunk {chunk['id']}, Relevance: {chunk.get('relevance_score', 0):.3f}]\n")
            prompt_parts.append(f"{chunk['text']}\n\n")
        
        # Output format instruction
        prompt_parts.append("""
=== YOUR TASK ===

Based on the target permit text above, identify all treatment processes and return them as a JSON object with this structure:

{
  "processes": [
    {
      "generic_name": "standardized_process_name_from_json",
      "confidence": 0.95,
      "evidence": "brief quote from text"
    }
  ]
}

Return ONLY valid JSON, no additional text.
""")
        
        return ''.join(prompt_parts)
    
    def extract_processes(self, 
                         relevant_chunks: List[Dict[str, any]], 
                         similar_examples: List[Dict[str, any]]) -> Dict[str, any]:
        """
        Execute LLM query to extract processes.
        
        Returns:
            Dict with processes and metadata
        """
        prompt = self.build_prompt(relevant_chunks, similar_examples)
        
        if self.llm_provider == "ollama":
            return self._query_ollama(prompt)
        else:
            return self._query_openai(prompt)
    
    def _query_ollama(self, prompt: str) -> Dict[str, any]:
        """Query Ollama API."""
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at extracting treatment process information from wastewater permits. Always respond with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                options={
                    "temperature": 0.1,  # Low temperature for consistency
                    "num_predict": 2000,  # Max tokens
                }
            )
            
            result_text = response['message']['content'].strip()
            
            # Parse JSON response
            result_text = re.sub(r'```json\s*', '', result_text)
            result_text = re.sub(r'```\s*$', '', result_text)
            
            result = json.loads(result_text)
            return result
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse Ollama response as JSON: {e}")
            print(f"Response: {result_text[:500]}...")
            return {"processes": [], "error": "JSON parse error", "raw_response": result_text}
        
        except Exception as e:
            print(f"Ollama query failed: {e}")
            return {"processes": [], "error": str(e)}
    
    def _query_openai(self, prompt: str) -> Dict[str, any]:
        """Query OpenAI API."""
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at extracting treatment process information from wastewater permits."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=2000
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            result_text = re.sub(r'```json\s*', '', result_text)
            result_text = re.sub(r'```\s*$', '', result_text)
            
            result = json.loads(result_text)
            return result
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse OpenAI response as JSON: {e}")
            print(f"Response: {result_text[:500]}...")
            return {"processes": [], "error": "JSON parse error", "raw_response": result_text}
        
        except Exception as e:
            print(f"OpenAI query failed: {e}")
            return {"processes": [], "error": str(e)}


class TreatmentExtractionPipeline:
    """Complete pipeline for treatment process extraction."""
    
    def __init__(self, 
                 process_json_path: str,
                 training_chunks_path: Optional[str] = None,
                 example_database_path: Optional[str] = None,
                 llm_provider: str = DEFAULT_LLM_PROVIDER,
                 llm_model: str = None,
                 api_key: Optional[str] = None,
                 use_langchain: bool = LANGCHAIN_AVAILABLE):
        """
        Initialize complete pipeline.
        
        Args:
            process_json_path: Path to process definitions JSON
            training_chunks_path: Path to CSV with training chunk examples
            example_database_path: Path to CSV with example permits and ground truth
            llm_provider: "ollama" or "openai"
            llm_model: Model name (e.g., "mistral:7b")
            api_key: OpenAI API key (only for OpenAI)
            use_langchain: Whether to use LangChain for chunking/retrieval
        """
        print(f"\n{'='*80}")
        print("Initializing Treatment Process Extraction Pipeline")
        print(f"{'='*80}")
        
        self.chunker = PDFChunker(use_langchain=use_langchain)
        
        # Load training chunks
        training_chunks = []
        if training_chunks_path and os.path.exists(training_chunks_path):
            df = pd.read_csv(training_chunks_path)
            if 'chunk_text' in df.columns:
                training_chunks = df['chunk_text'].dropna().tolist()
                print(f"Loaded {len(training_chunks)} training chunks")
        
        self.relevance_scorer = RelevanceScorer(training_chunks, use_langchain=use_langchain)
        
        # Load example database
        if example_database_path and os.path.exists(example_database_path):
            example_db = pd.read_csv(example_database_path)
            self.similar_finder = SimilarPermitFinder(example_db)
            print(f"Loaded {len(example_db)} example permits")
        else:
            self.similar_finder = None
            print("No example database provided")
        
        self.llm = TreatmentProcessLLM(
            process_json_path, 
            llm_provider=llm_provider,
            model=llm_model,
            api_key=api_key
        )
        
        print(f"{'='*80}\n")
    
    def process_permit(self, pdf_path: str, facility_name: str, npdes_no: str) -> Dict[str, any]:
        """
        Process a single permit through the complete pipeline.
        
        Returns:
            Dict with extracted processes and metadata
        """
        print(f"\nProcessing: {facility_name} ({npdes_no})")
        print(f"PDF: {pdf_path}")
        
        # Step 1: Extract and chunk text
        print("Step 1: Extracting and chunking text...")
        text = self.chunker.extract_text(pdf_path)
        if not text:
            return {
                'facility_name': facility_name,
                'npdes_no': npdes_no,
                'pdf_path': pdf_path,
                'error': 'Failed to extract text from PDF',
                'processes': []
            }
        
        chunks = self.chunker.chunk_text(text)
        print(f"  Created {len(chunks)} chunks")
        
        # Step 2: Find relevant chunks
        print("Step 2: Finding relevant chunks...")
        relevant_chunks = self.relevance_scorer.score_chunks(chunks)
        print(f"  Selected {len(relevant_chunks)} relevant chunks")
        
        # Step 3: Find similar examples
        similar_examples = []
        if self.similar_finder:
            print("Step 3: Finding similar example permits...")
            similar_examples = self.similar_finder.find_similar(text)
            for i, ex in enumerate(similar_examples, 1):
                print(f"  Example {i}: {ex.get('facility_name')} (similarity: {ex['similarity_score']:.3f})")
        else:
            print("Step 3: Skipping (no example database provided)")
        
        # Step 4 & 5: Build prompt and execute LLM query
        print("Step 4-5: Querying LLM...")
        result = self.llm.extract_processes(relevant_chunks, similar_examples)
        
        # Add metadata
        result['facility_name'] = facility_name
        result['npdes_no'] = npdes_no
        result['pdf_path'] = pdf_path
        result['n_chunks'] = len(chunks)
        result['n_relevant_chunks'] = len(relevant_chunks)
        
        print(f"  Found {len(result.get('processes', []))} processes")
        
        return result
    
    def process_batch(self, permits_csv: str, output_csv: str):
        """
        Process multiple permits from CSV.
        
        Args:
            permits_csv: CSV with columns: pdf_path, facility_name, npdes_no
            output_csv: Path to output CSV
        """
        df = pd.read_csv(permits_csv)
        results = []
        
        for idx, row in df.iterrows():
            result = self.process_permit(
                row['pdf_path'],
                row['facility_name'],
                row['npdes_no']
            )
            results.append(result)
        
        # Convert to DataFrame
        self.export_results(results, output_csv)
    
    def export_results(self, results: List[Dict[str, any]], output_path: str):
        """
        Export results to CSV.
        
        Creates a CSV with one row per facility, with process lists.
        """
        rows = []
        for result in results:
            processes = result.get('processes', [])
            
            # Extract process names and confidences
            process_names = [p.get('generic_name', '') for p in processes]
            confidences = [p.get('confidence', 0.0) for p in processes]
            evidences = [p.get('evidence', '') for p in processes]
            
            row = {
                'facility_name': result.get('facility_name', ''),
                'npdes_no': result.get('npdes_no', ''),
                'pdf_path': result.get('pdf_path', ''),
                'n_processes': len(processes),
                'process_list': '; '.join(process_names),
                'confidence_scores': '; '.join([f"{c:.2f}" for c in confidences]),
                'evidence_snippets': ' | '.join([e[:100] for e in evidences]),
                'error': result.get('error', '')
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"\nResults exported to: {output_path}")


# Example usage
if __name__ == "__main__":
    # Example: Process a single permit using Ollama + LangChain
    pipeline = TreatmentExtractionPipeline(
        process_json_path='npdes_permits/data/treatment_processes.json',
        training_chunks_path='npdes_permits/data/training_chunks.csv',
        example_database_path='npdes_permits/data/example_permits.csv',
        llm_provider="ollama",
        llm_model="mistral:7b",
        use_langchain=True  # Use LangChain for better chunking/retrieval
    )
    
    # Process single permit
    result = pipeline.process_permit(
        pdf_path='output/2025-10-8-test1/NPDES/example.pdf',
        facility_name='Example WWTP',
        npdes_no='CA0012345'
    )
    
    print(json.dumps(result, indent=2))
    
    # Or process batch
    # pipeline.process_batch('permits_to_process.csv', 'treatment_results.csv')
