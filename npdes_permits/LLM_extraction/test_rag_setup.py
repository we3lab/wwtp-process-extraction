#!/usr/bin/env python3
"""
Quick test script for RAG-based treatment process extraction

This script helps verify your setup is working correctly.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def check_dependencies():
    """Check if all required packages are installed."""
    print("Checking dependencies...")
    
    missing = []
    
    try:
        import ollama
        print("  ✓ ollama")
    except ImportError:
        print("  ✗ ollama - Install with: pip install ollama")
        missing.append("ollama")
    
    try:
        import langchain
        print("  ✓ langchain")
    except ImportError:
        print("  ✗ langchain - Install with: pip install langchain")
        missing.append("langchain")
    
    try:
        import chromadb
        print("  ✓ chromadb")
    except ImportError:
        print("  ✗ chromadb - Install with: pip install chromadb")
        missing.append("chromadb")
    
    try:
        import PyPDF2
        print("  ✓ PyPDF2")
    except ImportError:
        print("  ✗ PyPDF2 - Install with: pip install PyPDF2")
        missing.append("PyPDF2")
    
    try:
        import pandas
        print("  ✓ pandas")
    except ImportError:
        print("  ✗ pandas - Install with: pip install pandas")
        missing.append("pandas")
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("\nInstall all with:")
        print("  pip install ollama langchain langchain-community chromadb PyPDF2 pandas pysqlite3-binary")
        return False
    
    print("\n✅ All dependencies installed!\n")
    return True


def check_ollama():
    """Check if Ollama is running and models are available."""
    print("Checking Ollama setup...")
    
    try:
        import ollama
        
        # Check if Ollama is running
        try:
            models = ollama.list()
            print("  ✓ Ollama is running")
            
            # Check for required models
            model_names = [m['name'] for m in models.get('models', [])]
            
            has_llm = any('mistral' in m or 'llama' in m or 'phi' in m for m in model_names)
            has_embed = any('nomic-embed' in m or 'embed' in m for m in model_names)
            
            if has_llm:
                print("  ✓ LLM model available")
            else:
                print("  ⚠ No LLM model found - Run: ollama pull mistral:7b")
            
            if has_embed:
                print("  ✓ Embedding model available")
            else:
                print("  ⚠ No embedding model found - Run: ollama pull nomic-embed-text")
            
            if not has_llm or not has_embed:
                print("\n⚠ Some models missing. Install with:")
                print("  ollama pull mistral:7b")
                print("  ollama pull nomic-embed-text")
                return False
            
            print("\n✅ Ollama setup complete!\n")
            return True
            
        except Exception as e:
            print(f"  ✗ Ollama is not running - Start with: ollama serve")
            print(f"    Error: {e}")
            return False
    
    except ImportError:
        print("  ✗ ollama package not installed")
        return False


def test_vector_store():
    """Test ChromaDB vector store."""
    print("Testing ChromaDB vector store...")
    
    try:
        from treatment_process_extractor_rag import PermitVectorStore
        
        # Create test vector store
        vector_store = PermitVectorStore(
            chroma_path="chroma_test",
            embedding_model="nomic-embed-text"
        )
        
        print("  ✓ Vector store initialized")
        
        # Clean up test database
        import shutil
        if os.path.exists("chroma_test"):
            shutil.rmtree("chroma_test")
            print("  ✓ Test database cleaned up")
        
        print("\n✅ Vector store test passed!\n")
        return True
        
    except Exception as e:
        print(f"  ✗ Vector store test failed: {e}")
        return False


def test_sample_pdf():
    """Test PDF processing if sample PDF exists."""
    print("Looking for sample PDFs...")
    
    # Look for PDFs in output directory
    pdf_dirs = [
        Path("../output/2025-10-8-test1/NPDES"),
        Path("output/2025-10-8-test1/NPDES"),
        Path("../../output/2025-10-8-test1/NPDES"),
    ]
    
    sample_pdf = None
    for pdf_dir in pdf_dirs:
        if pdf_dir.exists():
            pdfs = list(pdf_dir.glob("*.pdf"))
            if pdfs:
                sample_pdf = pdfs[0]
                break
    
    if not sample_pdf:
        print("  ⚠ No sample PDFs found in output/2025-10-8-test1/NPDES")
        print("    Skipping PDF test")
        print("    (This is OK - just means we can't test with real data)")
        return True
    
    print(f"  ✓ Found sample PDF: {sample_pdf.name}")
    
    try:
        from treatment_process_extractor_rag import PermitVectorStore
        
        # Create test vector store
        vector_store = PermitVectorStore(
            chroma_path="chroma_test_pdf",
            embedding_model="nomic-embed-text"
        )
        
        # Index the PDF
        print("  Testing PDF indexing...")
        n_chunks = vector_store.index_pdf(
            str(sample_pdf),
            "Test Facility",
            "TEST0001"
        )
        
        print(f"  ✓ Indexed {n_chunks} chunks")
        
        # Test query
        print("  Testing semantic search...")
        results = vector_store.query("wastewater treatment processes", k=3)
        
        print(f"  ✓ Retrieved {len(results)} chunks")
        
        if results:
            print(f"    Top result score: {results[0]['score']:.3f}")
            print(f"    Chunk preview: {results[0]['text'][:100]}...")
        
        # Clean up
        import shutil
        if os.path.exists("chroma_test_pdf"):
            shutil.rmtree("chroma_test_pdf")
            print("  ✓ Test database cleaned up")
        
        print("\n✅ PDF processing test passed!\n")
        return True
        
    except Exception as e:
        print(f"  ✗ PDF test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*80)
    print("RAG SYSTEM TEST")
    print("="*80)
    print()
    
    results = []
    
    # Test 1: Dependencies
    results.append(("Dependencies", check_dependencies()))
    
    # Test 2: Ollama
    results.append(("Ollama Setup", check_ollama()))
    
    # Test 3: Vector Store
    results.append(("Vector Store", test_vector_store()))
    
    # Test 4: Sample PDF (if available)
    results.append(("PDF Processing", test_sample_pdf()))
    
    # Summary
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:20s} {status}")
        if not passed:
            all_passed = False
    
    print("="*80)
    
    if all_passed:
        print("\n🎉 All tests passed! Your RAG system is ready to use.")
        print("\nNext steps:")
        print("1. Index your PDFs:")
        print("   python treatment_process_extractor_rag.py --index output/2025-10-8-test1/NPDES --metadata metadata.csv")
        print("\n2. Extract processes:")
        print("   python treatment_process_extractor_rag.py --extract facilities.csv --output results.csv")
        print("\nSee RAG_QUICKSTART.md for full guide.")
    else:
        print("\n⚠ Some tests failed. Please fix the issues above.")
        print("\nQuick fix:")
        print("1. Install dependencies: pip install ollama langchain langchain-community chromadb PyPDF2 pandas pysqlite3-binary")
        print("2. Start Ollama: ollama serve")
        print("3. Pull models: ollama pull mistral:7b && ollama pull nomic-embed-text")
        print("4. Re-run this test: python test_rag_setup.py")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
