"""
Simple runner script for treatment process extraction.

Usage:
    python run_extraction.py --input permits.csv --output results.csv
    
Or process a single file:
    python run_extraction.py --pdf path/to/permit.pdf --facility "Plant Name" --npdes CA0012345
"""

import argparse
import os
from pathlib import Path
from npdes_permits.LLM_extraction.treatment_process_extractor import TreatmentExtractionPipeline

def main():
    parser = argparse.ArgumentParser(description='Extract treatment processes from NPDES permits')
    
    # Input options
    parser.add_argument('--input', '-i', help='CSV file with permits to process (columns: pdf_path, facility_name, npdes_no)')
    parser.add_argument('--pdf', help='Single PDF file to process')
    parser.add_argument('--facility', help='Facility name (required if using --pdf)')
    parser.add_argument('--npdes', help='NPDES number (required if using --pdf)')
    
    # Output
    parser.add_argument('--output', '-o', default='output/treatment_extraction/results.csv',
                       help='Output CSV file path')
    
    # Configuration files
    parser.add_argument('--processes-json', default='npdes_permits/data/treatment_processes.json',
                       help='Path to treatment processes JSON')
    parser.add_argument('--training-chunks', default='npdes_permits/data/training_chunks.csv',
                       help='Path to training chunks CSV')
    parser.add_argument('--examples', default='npdes_permits/data/example_permits.csv',
                       help='Path to example permits CSV')
    
    # LLM options
    parser.add_argument('--llm', default='ollama', choices=['ollama', 'openai'],
                       help='LLM provider (default: ollama)')
    parser.add_argument('--model', default='mistral:7b',
                       help='Model name (default: mistral:7b for Ollama, gpt-4 for OpenAI)')
    parser.add_argument('--api-key', help='OpenAI API key (only needed for OpenAI)')
    
    # LangChain option
    parser.add_argument('--use-langchain', action='store_true', default=True,
                       help='Use LangChain for chunking and retrieval (default: True)')
    parser.add_argument('--no-langchain', dest='use_langchain', action='store_false',
                       help='Disable LangChain (use basic methods)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.input and not args.pdf:
        parser.error('Either --input or --pdf must be specified')
    
    if args.pdf and (not args.facility or not args.npdes):
        parser.error('--facility and --npdes are required when using --pdf')
    
    # Initialize pipeline
    pipeline = TreatmentExtractionPipeline(
        process_json_path=args.processes_json,
        training_chunks_path=args.training_chunks if os.path.exists(args.training_chunks) else None,
        example_database_path=args.examples if os.path.exists(args.examples) else None,
        llm_provider=args.llm,
        llm_model=args.model,
        api_key=args.api_key,
        use_langchain=args.use_langchain
    )
    
    # Process
    if args.input:
        print(f"Processing permits from: {args.input}")
        pipeline.process_batch(args.input, args.output)
    else:
        print(f"Processing single permit: {args.pdf}")
        result = pipeline.process_permit(args.pdf, args.facility, args.npdes)
        
        # Export single result
        pipeline.export_results([result], args.output)
        
        # Also print to console
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(f"Facility: {result['facility_name']}")
        print(f"NPDES No: {result['npdes_no']}")
        print(f"\nProcesses found:")
        for process in result.get('processes', []):
            print(f"  - {process['generic_name']} (confidence: {process.get('confidence', 0):.2f})")
            if 'evidence' in process:
                print(f"    Evidence: {process['evidence'][:100]}...")
    
    print(f"\n✓ Results saved to: {args.output}")


if __name__ == '__main__':
    main()
