"""
LLM-based Treatment Process Extraction from NPDES Permits using RAG (ChromaDB)

Improved workflow with Retrieval Augmented Generation:
1. Index all PDFs into ChromaDB vector database
2. For each query, retrieve most relevant chunks semantically
3. Find similar example permits from ground truth database
4. Create LLM query with retrieved context and examples
5. Execute LLM query and parse results
6. Export to CSV
"""

# SQLite3 version check for ChromaDB compatibility
import sqlite3
import sys

if (sqlite3.sqlite_version_info[0] < 3) or (
    (sqlite3.sqlite_version_info[0] == 3) and (sqlite3.sqlite_version_info[1] < 35)
):
    print(f"Upgrading sqlite3 version from {sqlite3.sqlite_version}")
    try:
        import pysqlite3
        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
        import sqlite3
        print(f"New sqlite3 version is {sqlite3.sqlite_version}")
    except ImportError:
        print("Warning: pysqlite3 not installed. Install with: pip install pysqlite3-binary")

import os
import json
import re
import shutil
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import PyPDF2

# LangChain imports (using only LangChain for Ollama integration)
try:
    from langchain.schema.document import Document
    from langchain.prompts import ChatPromptTemplate
    from langchain_community.llms.ollama import Ollama as LangChainOllama
    from langchain_community.vectorstores import Chroma
    from langchain_core._api import LangChainDeprecationWarning
    from langchain_community.embeddings.ollama import OllamaEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    LANGCHAIN_AVAILABLE = True
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: langchain not installed. Install with: pip install langchain langchain-community chromadb")
    raise ImportError("LangChain is required for this RAG implementation")

# Configure paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "data"
OUTPUT_DIR = BASE_DIR / "output" / "treatment_extraction_rag"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ChromaDB path
CHROMA_PATH = str(BASE_DIR / "chroma_permits")

# Configuration
CHUNK_SIZE = 1000  # characters per chunk
OVERLAP = 200  # character overlap between chunks
TOP_K_CHUNKS = 5  # number of relevant chunks to retrieve
N_SIMILAR_EXAMPLES = 2  # number of similar permits to use as examples

# LLM Configuration
DEFAULT_OLLAMA_MODEL = "mistral:7b"  # or "llama3", "phi3", etc.
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"  # Ollama embedding model


class PermitVectorStore:
    """Manage ChromaDB vector store for permit documents."""
    
    def __init__(self, 
                 chroma_path: str = CHROMA_PATH,
                 embedding_model: str = DEFAULT_EMBEDDING_MODEL,
                 chunk_size: int = CHUNK_SIZE,
                 chunk_overlap: int = OVERLAP):
        """Initialize vector store with ChromaDB."""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required for RAG. Install with: pip install langchain langchain-community chromadb")
        
        self.chroma_path = chroma_path
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
        print(f"Vector store initialized with {embedding_model} embeddings")
    
    def clear_database(self):
        """Clear the ChromaDB database."""
        if os.path.exists(self.chroma_path):
            shutil.rmtree(self.chroma_path)
            print(f"Cleared database at {self.chroma_path}")
    
    def index_pdf(self, pdf_path: str, facility_name: str, npdes_no: str) -> int:
        """
        Index a single PDF into the vector store.
        
        Args:
            pdf_path: Path to PDF file
            facility_name: Name of facility
            npdes_no: NPDES permit number
        
        Returns:
            Number of chunks added
        """
        # Extract text from PDF
        text_parts = []
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text() or ''
                        text_parts.append(page_text)
                    except Exception as e:
                        print(f"  Warning: Error extracting page {page_num}: {e}")
                        continue
        except Exception as e:
            print(f"  Error reading PDF {pdf_path}: {e}")
            return 0
        
        full_text = '\n'.join(text_parts)
        
        # Split into chunks
        text_chunks = self.text_splitter.split_text(full_text)
        
        # Create documents with metadata
        documents = []
        for idx, chunk_text in enumerate(text_chunks):
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "source": pdf_path,
                    "facility_name": facility_name,
                    "npdes_no": npdes_no,
                    "chunk_id": idx,
                    "id": f"{npdes_no}:{idx}"
                }
            )
            documents.append(doc)
        
        # Add to ChromaDB
        if documents:
            db = Chroma(
                persist_directory=self.chroma_path, 
                embedding_function=self.embeddings
            )
            
            # Get existing IDs to avoid duplicates
            existing_items = db.get(include=[])
            existing_ids = set(existing_items["ids"])
            
            # Filter out documents that already exist
            new_docs = []
            new_ids = []
            for doc in documents:
                doc_id = doc.metadata["id"]
                if doc_id not in existing_ids:
                    new_docs.append(doc)
                    new_ids.append(doc_id)
            
            if new_docs:
                db.add_documents(new_docs, ids=new_ids)
                print(f"  Added {len(new_docs)} chunks to database")
            else:
                print(f"  No new chunks to add (already indexed)")
            
            return len(new_docs)
        
        return 0
    
    def index_directory(self, pdf_directory: str, metadata_csv: Optional[str] = None):
        """
        Index all PDFs in a directory.
        
        Args:
            pdf_directory: Directory containing PDF files
            metadata_csv: Optional CSV with columns: pdf_path, facility_name, npdes_no
        """
        # Load metadata if provided
        metadata_map = {}
        if metadata_csv and os.path.exists(metadata_csv):
            df = pd.read_csv(metadata_csv)
            for _, row in df.iterrows():
                pdf_path = row.get('pdf_path', '')
                metadata_map[os.path.basename(pdf_path)] = {
                    'facility_name': row.get('facility_name', 'Unknown'),
                    'npdes_no': row.get('npdes_no', 'Unknown')
                }
        
        # Find all PDFs
        pdf_files = list(Path(pdf_directory).glob("**/*.pdf"))
        print(f"\nIndexing {len(pdf_files)} PDFs from {pdf_directory}")
        
        total_chunks = 0
        for idx, pdf_path in enumerate(pdf_files, 1):
            pdf_name = pdf_path.name
            
            # Get metadata
            if pdf_name in metadata_map:
                facility_name = metadata_map[pdf_name]['facility_name']
                npdes_no = metadata_map[pdf_name]['npdes_no']
            else:
                # Extract from filename if possible
                facility_name = pdf_name.replace('.pdf', '').replace('_', ' ')
                npdes_no = pdf_name.replace('.pdf', '')
            
            print(f"[{idx}/{len(pdf_files)}] {facility_name} ({npdes_no})")
            
            n_chunks = self.index_pdf(str(pdf_path), facility_name, npdes_no)
            total_chunks += n_chunks
        
        print(f"\n✓ Indexed {total_chunks} total chunks from {len(pdf_files)} PDFs")
    
    def query(self, query_text: str, k: int = TOP_K_CHUNKS, 
              filter_facility: Optional[str] = None) -> List[Dict[str, any]]:
        """
        Query the vector store for relevant chunks.
        
        Args:
            query_text: Query string
            k: Number of results to return
            filter_facility: Optional facility name to filter by
        
        Returns:
            List of dicts with chunk content and metadata
        """
        db = Chroma(
            persist_directory=self.chroma_path, 
            embedding_function=self.embeddings
        )
        
        # Build filter if needed
        where_filter = None
        if filter_facility:
            where_filter = {"facility_name": filter_facility}
        
        # Search with scores
        results = db.similarity_search_with_score(
            query_text, 
            k=k,
            filter=where_filter
        )
        
        # Format results
        chunks = []
        for doc, score in results:
            chunks.append({
                'text': doc.page_content,
                'score': float(score),
                'facility_name': doc.metadata.get('facility_name', ''),
                'npdes_no': doc.metadata.get('npdes_no', ''),
                'source': doc.metadata.get('source', ''),
                'chunk_id': doc.metadata.get('chunk_id', 0)
            })
        
        return chunks


class TreatmentProcessRAG:
    """RAG-based treatment process extraction using ChromaDB + Ollama (via LangChain)."""
    
    def __init__(self,
                 process_json_path: str,
                 vector_store: PermitVectorStore,
                 example_database_path: Optional[str] = None,
                 llm_model: str = None):
        """
        Initialize RAG pipeline with LangChain Ollama.
        
        Args:
            process_json_path: Path to process definitions JSON
            vector_store: PermitVectorStore instance
            example_database_path: Path to CSV with ground truth examples
            llm_model: Ollama model name (default: mistral:7b)
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required. Install with: pip install langchain langchain-community chromadb")
        
        # Load process definitions
        with open(process_json_path, 'r') as f:
            self.process_definitions = json.load(f)
        
        self.vector_store = vector_store
        
        # Load example database for few-shot learning
        self.example_database = None
        if example_database_path and os.path.exists(example_database_path):
            self.example_database = pd.read_csv(example_database_path)
            print(f"Loaded {len(self.example_database)} example permits for few-shot learning")
        
        # Initialize LLM using LangChain's Ollama
        self.model_name = llm_model or DEFAULT_OLLAMA_MODEL
        print(f"Initializing Ollama with model: {self.model_name}")
        
        try:
            self.llm = LangChainOllama(
                model=self.model_name,
                temperature=0.1,
                num_predict=2000,
            )
            # Test connection with a simple query
            test_response = self.llm.invoke("test")
            print("✓ Ollama is running and responding")
        except Exception as e:
            print(f"Warning: Could not connect to Ollama: {e}")
            print("Make sure Ollama is running: ollama serve")
            print(f"And the model is pulled: ollama pull {self.model_name}")
        
        print(f"\n{'='*80}\n")
    
    def build_prompt(self, 
                    retrieved_chunks: List[Dict[str, any]], 
                    similar_examples: List[Dict[str, any]]) -> str:
        """Build LLM prompt with retrieved context and examples."""
        prompt_parts = []
        
        # System instruction
        prompt_parts.append("""You are an expert at extracting wastewater treatment process information from NPDES permit documents.

Your task is to identify all treatment processes mentioned in the provided permit text and return them using the standardized process names from the provided JSON dictionary.

Instructions:
1. Read the permit text carefully
2. Identify all treatment processes mentioned
3. Map each process to its standardized name from the JSON (use alternative names to help identify)
4. Return ONLY the standardized generic names as a JSON array
5. Include a confidence score (0.0-1.0) for each process
6. Include a brief text evidence quote for each process

""")
        
        # Process definitions
        prompt_parts.append("=== TREATMENT PROCESS DEFINITIONS ===\n")
        prompt_parts.append(json.dumps(self.process_definitions, indent=2))
        prompt_parts.append("\n\n")
        
        # Few-shot examples
        if similar_examples:
            prompt_parts.append("=== EXAMPLES FROM SIMILAR PERMITS ===\n\n")
            for i, example in enumerate(similar_examples, 1):
                prompt_parts.append(f"Example {i}:\n")
                prompt_parts.append(f"Facility: {example.get('facility_name', 'Unknown')}\n")
                prompt_parts.append(f"NPDES No: {example.get('npdes_no', 'Unknown')}\n")
                
                if 'relevant_excerpt' in example:
                    prompt_parts.append(f"Permit Text:\n{example['relevant_excerpt'][:500]}...\n")
                
                if 'processes' in example:
                    processes = example['processes']
                    if isinstance(processes, str):
                        try:
                            processes = json.loads(processes)
                        except:
                            pass
                    prompt_parts.append(f"Identified Processes: {json.dumps(processes)}\n")
                
                prompt_parts.append("\n")
        
        # Retrieved context from vector store
        prompt_parts.append("=== RETRIEVED PERMIT TEXT ===\n\n")
        for i, chunk in enumerate(retrieved_chunks, 1):
            prompt_parts.append(f"[Chunk {i}, Relevance Score: {chunk.get('score', 0):.3f}]\n")
            prompt_parts.append(f"{chunk['text']}\n\n")
        
        # Output format
        prompt_parts.append("""
=== YOUR TASK ===

Based on the retrieved permit text above, identify all treatment processes and return them as a JSON object:

{
  "processes": [
    {
      "generic_name": "standardized_process_name_from_json",
      "confidence": 0.95,
      "evidence": "brief quote from text showing this process"
    }
  ]
}

Return ONLY valid JSON, no additional text.
""")
        
        return ''.join(prompt_parts)
    
    def extract_processes(self, 
                         facility_name: str,
                         npdes_no: str,
                         query_text: Optional[str] = None) -> Dict[str, any]:
        """
        Extract treatment processes using RAG.
        
        Args:
            facility_name: Facility name
            npdes_no: NPDES permit number
            query_text: Optional specific query (default: generic treatment process query)
        
        Returns:
            Dict with extracted processes and metadata
        """
        print(f"\nExtracting processes: {facility_name} ({npdes_no})")
        
        # Default query if not provided
        if not query_text:
            query_text = (
                "What wastewater treatment processes are used at this facility? "
                "Include preliminary treatment, primary treatment, secondary treatment, "
                "tertiary treatment, disinfection, and sludge treatment processes."
            )
        
        # Retrieve relevant chunks from vector store
        print("Step 1: Retrieving relevant chunks from vector store...")
        retrieved_chunks = self.vector_store.query(
            query_text,
            k=TOP_K_CHUNKS,
            filter_facility=None  # Search across all permits, not just this one
        )
        print(f"  Retrieved {len(retrieved_chunks)} relevant chunks")
        
        # Find similar examples for few-shot learning
        similar_examples = []
        if self.example_database is not None:
            print("Step 2: Finding similar example permits...")
            # Simple similarity based on retrieved chunks
            chunk_texts = [c['text'] for c in retrieved_chunks]
            combined_text = ' '.join(chunk_texts[:3])  # Use top 3 chunks
            
            # Get 2 random examples for now (could implement better similarity)
            similar_examples = self.example_database.head(N_SIMILAR_EXAMPLES).to_dict('records')
            print(f"  Using {len(similar_examples)} example permits")
        
        # Build prompt
        print("Step 3: Building prompt and querying LLM...")
        prompt = self.build_prompt(retrieved_chunks, similar_examples)
        
        # Query LLM using LangChain
        result = self._query_llm(prompt)
        
        # Add metadata
        result['facility_name'] = facility_name
        result['npdes_no'] = npdes_no
        result['n_chunks_retrieved'] = len(retrieved_chunks)
        result['chunk_sources'] = [c.get('source', '') for c in retrieved_chunks]
        
        print(f"  Found {len(result.get('processes', []))} processes\n")
        
        return result
    
    def _query_llm(self, prompt: str) -> Dict[str, any]:
        """Query Ollama via LangChain."""
        try:
            # Invoke LLM
            response = self.llm.invoke(prompt)
            
            # Clean up response (remove markdown code blocks if present)
            result_text = response.strip()
            result_text = re.sub(r'```json\s*', '', result_text)
            result_text = re.sub(r'```\s*$', '', result_text)
            
            # Parse JSON
            result = json.loads(result_text)
            return result
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            print(f"Response preview: {result_text[:500]}...")
            return {
                "processes": [], 
                "error": "JSON parse error", 
                "raw_response": result_text[:500]
            }
        except Exception as e:
            print(f"LLM query failed: {e}")
            return {"processes": [], "error": str(e)}
    
    def process_batch(self, permits_csv: str, output_csv: str):
        """Process multiple permits from CSV."""
        df = pd.read_csv(permits_csv)
        results = []
        
        for idx, row in df.iterrows():
            result = self.extract_processes(
                facility_name=row['facility_name'],
                npdes_no=row['npdes_no']
            )
            results.append(result)
            
            # Save checkpoint every 10
            if (idx + 1) % 10 == 0:
                self._export_results(results, output_csv.replace('.csv', f'_checkpoint_{idx+1}.csv'))
                print(f"✓ Checkpoint saved at {idx+1} permits")
        
        self._export_results(results, output_csv)
    
    def _export_results(self, results: List[Dict[str, any]], output_path: str):
        """Export results to CSV."""
        rows = []
        for result in results:
            processes = result.get('processes', [])
            
            process_names = [p.get('generic_name', '') for p in processes]
            confidences = [p.get('confidence', 0.0) for p in processes]
            evidences = [p.get('evidence', '') for p in processes]
            
            row = {
                'facility_name': result.get('facility_name', ''),
                'npdes_no': result.get('npdes_no', ''),
                'n_processes': len(processes),
                'process_list': '; '.join(process_names),
                'confidence_scores': '; '.join([f"{c:.2f}" for c in confidences]),
                'evidence_snippets': ' | '.join([e[:100] for e in evidences]),
                'n_chunks_retrieved': result.get('n_chunks_retrieved', 0),
                'error': result.get('error', '')
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to: {output_path}")


# Command-line interface
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG-based treatment process extraction")
    parser.add_argument('--reset', action='store_true', help='Clear ChromaDB database')
    parser.add_argument('--index', type=str, help='Directory of PDFs to index')
    parser.add_argument('--metadata', type=str, help='CSV with PDF metadata (pdf_path, facility_name, npdes_no)')
    parser.add_argument('--extract', type=str, help='CSV with permits to extract (facility_name, npdes_no)')
    parser.add_argument('--output', type=str, default='treatment_results_rag.csv', help='Output CSV path')
    parser.add_argument('--processes-json', type=str, default=str(DATA_DIR / 'treatment_processes.json'))
    parser.add_argument('--examples', type=str, default=str(DATA_DIR / 'example_permits.csv'))
    parser.add_argument('--model', type=str, default='mistral:7b', help='Ollama model name')
    parser.add_argument('--embedding-model', type=str, default='nomic-embed-text', help='Ollama embedding model')
    
    args = parser.parse_args()
    
    # Initialize vector store
    vector_store = PermitVectorStore(
        chroma_path=CHROMA_PATH,
        embedding_model=args.embedding_model
    )
    
    # Reset database if requested
    if args.reset:
        print("Clearing ChromaDB database...")
        vector_store.clear_database()
        print("✓ Database cleared")
        return
    
    # Index PDFs if requested
    if args.index:
        print(f"\nIndexing PDFs from: {args.index}")
        vector_store.index_directory(args.index, args.metadata)
        print("✓ Indexing complete")
        return
    
    # Extract processes if requested
    if args.extract:
        print(f"\nExtracting treatment processes from: {args.extract}")
        
        # Initialize RAG pipeline (using LangChain Ollama only)
        rag = TreatmentProcessRAG(
            process_json_path=args.processes_json,
            vector_store=vector_store,
            example_database_path=args.examples,
            llm_model=args.model
        )
        
        # Process batch
        rag.process_batch(args.extract, args.output)
        print("✓ Extraction complete")
    
    else:
        print("No action specified. Use --reset, --index, or --extract")
        parser.print_help()


if __name__ == "__main__":
    main()
