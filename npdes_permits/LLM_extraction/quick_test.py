#!/usr/bin/env python3
"""
Quick and easy test for RAG-based treatment process extraction

This creates a minimal test case to verify everything works.
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("QUICK RAG TEST")
print("="*80)
print()

# Step 1: Check dependencies
print("Step 1: Checking dependencies...")
try:
    from langchain_community.llms.ollama import Ollama
    from langchain_community.embeddings.ollama import OllamaEmbeddings
    from langchain_community.vectorstores import Chroma
    import chromadb
    print("  ✓ All LangChain packages available")
except ImportError as e:
    print(f"  ✗ Missing package: {e}")
    print("\nInstall with:")
    print("  pip install langchain langchain-community chromadb")
    sys.exit(1)

# Step 2: Check Ollama connection
print("\nStep 2: Testing Ollama connection...")
try:
    # Test LLM
    llm = Ollama(model="mistral:7b")
    response = llm.invoke("Say 'hello' in one word.")
    print(f"  ✓ LLM works: {response[:50]}")
    
    # Test embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    test_embed = embeddings.embed_query("test")
    print(f"  ✓ Embeddings work: {len(test_embed)} dimensions")
    
except Exception as e:
    print(f"  ✗ Ollama error: {e}")
    print("\nMake sure Ollama is running:")
    print("  ollama serve")
    print("\nAnd models are pulled:")
    print("  ollama pull mistral:7b")
    print("  ollama pull nomic-embed-text")
    sys.exit(1)

# Step 3: Create test vector store
print("\nStep 3: Creating test vector store...")
try:
    from langchain.schema.document import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    # Create test documents
    test_docs = [
        Document(
            page_content="The wastewater treatment plant uses preliminary screening with bar screens and grit removal to remove large debris.",
            metadata={"source": "test1", "facility": "Test Plant A", "npdes": "TEST001"}
        ),
        Document(
            page_content="Secondary treatment consists of activated sludge process with aeration basins and return activated sludge.",
            metadata={"source": "test2", "facility": "Test Plant A", "npdes": "TEST001"}
        ),
        Document(
            page_content="Disinfection is achieved using ultraviolet light treatment with UV lamps at the final stage.",
            metadata={"source": "test3", "facility": "Test Plant B", "npdes": "TEST002"}
        ),
        Document(
            page_content="The facility includes secondary clarifiers for solids separation followed by chlorination for disinfection.",
            metadata={"source": "test4", "facility": "Test Plant C", "npdes": "TEST003"}
        ),
    ]
    
    # Create vector store
    test_chroma_path = "chroma_quick_test"
    
    # Clean up old test database
    import shutil
    if os.path.exists(test_chroma_path):
        shutil.rmtree(test_chroma_path)
    
    vectorstore = Chroma.from_documents(
        documents=test_docs,
        embedding=embeddings,
        persist_directory=test_chroma_path
    )
    
    print(f"  ✓ Vector store created with {len(test_docs)} documents")
    
except Exception as e:
    print(f"  ✗ Vector store error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Test semantic search
print("\nStep 4: Testing semantic search...")
try:
    queries = [
        "What screening methods are used?",
        "Tell me about disinfection processes",
        "What is the secondary treatment process?"
    ]
    
    for query in queries:
        results = vectorstore.similarity_search(query, k=2)
        print(f"\n  Query: '{query}'")
        print(f"  Found {len(results)} results:")
        for i, doc in enumerate(results, 1):
            print(f"    {i}. [{doc.metadata['facility']}] {doc.page_content[:60]}...")
    
    print("\n  ✓ Semantic search works!")
    
except Exception as e:
    print(f"  ✗ Search error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Test LLM extraction with context
print("\nStep 5: Testing LLM with retrieved context...")
try:
    query = "What treatment processes are mentioned?"
    context_docs = vectorstore.similarity_search(query, k=3)
    
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    
    prompt = f"""Based on the following wastewater treatment plant descriptions, list all the treatment processes mentioned.

Context:
{context_text}

List the processes in a simple bullet list."""
    
    llm_response = llm.invoke(prompt)
    
    print(f"\n  LLM Response:")
    print(f"  {'-'*70}")
    print(f"  {llm_response}")
    print(f"  {'-'*70}")
    print("\n  ✓ LLM extraction works!")
    
except Exception as e:
    print(f"  ✗ LLM extraction error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 6: Test with actual file if available
print("\nStep 6: Looking for actual PDF to test...")
pdf_dirs = [
    Path("../../output/2025-10-8-test1/NPDES"),
    Path("../output/2025-10-8-test1/NPDES"),
]

sample_pdf = None
for pdf_dir in pdf_dirs:
    if pdf_dir.exists():
        pdfs = list(pdf_dir.glob("*.pdf"))
        if pdfs:
            sample_pdf = pdfs[0]
            break

if sample_pdf:
    print(f"  ✓ Found sample PDF: {sample_pdf.name}")
    
    try:
        import PyPDF2
        
        # Extract text from first page
        with open(sample_pdf, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            if len(reader.pages) > 0:
                first_page = reader.pages[0].extract_text()[:500]
                print(f"\n  First 500 chars of PDF:")
                print(f"  {'-'*70}")
                print(f"  {first_page}")
                print(f"  {'-'*70}")
                
                # Test semantic search with real content
                results = vectorstore.similarity_search(first_page[:200], k=1)
                print(f"\n  Similarity to test docs: {len(results)} matches found")
                print("  ✓ PDF processing would work!")
        
    except Exception as e:
        print(f"  ⚠ Could not process PDF (not critical): {e}")
else:
    print("  ⚠ No sample PDFs found (optional)")
    print("    Place PDFs in output/2025-10-8-test1/NPDES to test with real data")

# Cleanup
print("\nCleaning up test database...")
import shutil
if os.path.exists(test_chroma_path):
    shutil.rmtree(test_chroma_path)
print("  ✓ Cleaned up")

# Final summary
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print()
print("✅ All core components working!")
print()
print("Your RAG system is ready to use. Next steps:")
print()
print("1. Index your PDFs:")
print("   python treatment_process_extractor_rag.py \\")
print("       --index output/2025-10-8-test1/NPDES \\")
print("       --metadata permits_metadata.csv")
print()
print("2. Extract processes:")
print("   python treatment_process_extractor_rag.py \\")
print("       --extract facilities.csv \\")
print("       --output results.csv")
print()
print("See RAG_QUICKSTART.md for full documentation.")
print("="*80)
