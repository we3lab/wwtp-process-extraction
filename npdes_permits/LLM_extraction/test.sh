#!/bin/bash
# Ultra-quick test - just runs the quick test script

cd "$(dirname "$0")"

echo "Running quick RAG test..."
echo ""

python3 quick_test.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Success! Your RAG system is working."
    echo ""
    echo "Try a real example:"
    echo "  python treatment_process_extractor_rag.py --help"
else
    echo ""
    echo "❌ Test failed. Check the errors above."
    echo ""
    echo "Common fixes:"
    echo "  1. Start Ollama: ollama serve"
    echo "  2. Pull models: ollama pull mistral:7b && ollama pull nomic-embed-text"
    echo "  3. Install deps: pip install -r requirements_rag.txt"
fi
