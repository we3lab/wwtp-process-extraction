# WWTP Unit Process Extraction from Permits Tool
These tools are designed to access unit process data from the National Pollutant Discharge Elimination System (NPDES) permits for California.

## Description 
Researchers have utilized the CWNS to aggregate WWTP unit processes. However, this data is infrequent, voluntary, and sparse. To address these limitations, we utilize regulatory permits. The following Python tools are used to collect its data:


1. Build CWNS process tables
    - [wwtp_process_extraction/step1_build_cwns_table.py](wwtp_process_extraction/step1_build_cwns_table.py): creates unit_processes_by_facility_cwns.csv (and table_s1) from CWNS 2004/2008/2012 data. CWNS 2022 lists CA facilities but no CA unit processes

2. Scrape permits and site metadata
    - [wwtp_process_extraction/step2_scrape_npdes.py](wwtp_process_extraction/step2_scrape_npdes.py): scrapes CIWQS, downloads permit PDFs into output/permits/, and writes facilities.json, site_data_all.csv and site_data_relevant.csv
        - set *AS_OF = "YYYY-MM-DD"* at the top of the script to select the permit in force at a past date instead of today's active one; each run snapshots its facilities.json and site_data_relevant.csv to output/site_data/<AS_OF>/
        - every run writes facilities.json, site_data_all.csv and site_data_relevant.csv only to output/site_data/<date>/; the top-level files steps 3-6 read stay frozen unless *UPDATE_TOP_LEVEL = True*
        - with *UPDATE_TOP_LEVEL = True*, unions all dated snapshots into the top-level site_data_relevant.csv, so steps 3-6 process each document once

3. Extract permit text
    - [wwtp_process_extraction/step3_get_facility_descriptions.py](wwtp_process_extraction/step3_get_facility_descriptions.py): extracts relevant text sections from permit PDFs into per-facility text files

4. Detect treatment processes with keyword search
    - [wwtp_process_extraction/step4_keyword_extraction.py](wwtp_process_extraction/step4_keyword_extraction.py): scans permit text against unitprocess_keywords.json and writes unit_processes_by_pdf_kw.csv / unit_processes_by_facility_kw.csv with present/future status

5. Detect treatment processes with LLM extraction
    - [wwtp_process_extraction/step5_llm_extraction.py](wwtp_process_extraction/step5_llm_extraction.py): runs LLM extraction on permit text
        - default: ontology-based gpt-5-mini on the manually-read facilities; *--all_facilities* runs the full CA set
        - *--method* / *--model* pick one config; *--all_methods* / *--all_models* loop over all of them
        - the ontology-based prompt uses data/llm_extraction/input/ontology.txt, regenerated each run from the pinned Zenodo water ontology release (cached in data/ontology_cache/)
        - *--web_search* uses the claude CLI with web search; *--waterrag_context* adds literature context from step5b; *--repeat_runs N* repeats the benchmark for run-to-run variance
        - needs a Stanford AI API key in wwtp_process_extraction/API_key.txt (except *--web_search*)
        - results saved as JSON under output/llm_extraction/<method>_<model>/
    - [wwtp_process_extraction/step5b_waterrag_retrieval.py](wwtp_process_extraction/step5b_waterrag_retrieval.py): retrieves and reranks WaterRAG literature chunks per facility into output/waterrag_retrieval/ (separate Python 3.11 env)

6. Post-process LLM output back to CWNS format
    - [wwtp_process_extraction/step6_postprocess_llm_output.py](wwtp_process_extraction/step6_postprocess_llm_output.py): maps LLM outputs onto the unit process columns and writes unit_processes_by_pdf_llm.csv / unit_processes_by_facility_llm.csv with present/future/past/offsite status

**Figures and tables**
- [wwtp_process_extraction/table_1_evaluate_model_performance.py](wwtp_process_extraction/table_1_evaluate_model_performance.py): evaluates keyword and LLM methods against manual labels (table_1, table_s3, table_s5)
- [wwtp_process_extraction/figure_2_data_source_comparison.py](wwtp_process_extraction/figure_2_data_source_comparison.py): compares CWNS and the manual permit reading to the ground-truth supplemental data
- [wwtp_process_extraction/figure_3_extraction_comparison.py](wwtp_process_extraction/figure_3_extraction_comparison.py): CWNS vs LLM (and keyword) unit process detection counts; also updates data/ciwqs_to_cwns.csv with new name matches
- [wwtp_process_extraction/figure_4_ca_needs_comparison.py](wwtp_process_extraction/figure_4_ca_needs_comparison.py): CA treatment capital needs rebuilt from permit-extracted planned changes using EPA's CWNS 2022 cost curves, against CWNS's own reported needs
- [wwtp_process_extraction/figure_5_ca_ghg_comparison.py](wwtp_process_extraction/figure_5_ca_ghg_comparison.py): computes CA WWTP GHG emissions across treatment process data sources (El Abbadi all-sources, El Abbadi CWNS-only, permit extraction); fetches El Abbadi data from GitHub
- [wwtp_process_extraction/figure_s2_method_comparison.py](wwtp_process_extraction/figure_s2_method_comparison.py): keyword vs LLM method comparison
- [wwtp_process_extraction/figure_s3_category_f1_method.py](wwtp_process_extraction/figure_s3_category_f1_method.py): per-category F1 score comparison across methods

## Installation

```bash
pip install -e .
```

## How to Run
Executing from the repository root directory:

```bash
python wwtp_process_extraction/step1_build_cwns_table.py
python wwtp_process_extraction/step2_scrape_npdes.py
# YEAR-OVER-YEAR: set AS_OF in step2 to each past date (2026-06-01 ... 2021-06-01) and rerun.
# PDFs and LLM outputs stay in the shared folders -- a document's extraction does not change
# between years -- so only genuinely new orders cost anything.
python wwtp_process_extraction/step3_get_facility_descriptions.py
python wwtp_process_extraction/step4_keyword_extraction.py
# MODEL COMPARISON (manually-read facilities)
python wwtp_process_extraction/step5_llm_extraction.py --all_models --all_methods
# WEB SEARCH (claude-sonnet-4-6)
python wwtp_process_extraction/step5_llm_extraction.py --model claude-sonnet-4-6 --web_search --all_methods
# WATERRAG (gpt-5-mini)
# step5b runs in a SEPARATE Python 3.11 env with torch/langchain/faiss (no torch for 3.12)
# cached contexts in output/waterrag_retrieval/ are skipped; delete them to redo
conda run --no-capture-output -n waterrag python wwtp_process_extraction/step5b_waterrag_retrieval.py
python wwtp_process_extraction/step5_llm_extraction.py --waterrag_context
# FULL CA, ONTOLOGY-BASED, GPT-5-MINI
python wwtp_process_extraction/step5_llm_extraction.py --all_facilities
# F1 VARIANCE: 2 extra benchmark runs
python wwtp_process_extraction/step5_llm_extraction.py --repeat_runs 2
python wwtp_process_extraction/step6_postprocess_llm_output.py
python wwtp_process_extraction/table_1_evaluate_model_performance.py
python wwtp_process_extraction/figure_2_data_source_comparison.py
python wwtp_process_extraction/figure_3_extraction_comparison.py
python wwtp_process_extraction/figure_4_ca_needs_comparison.py
python wwtp_process_extraction/figure_5_ca_ghg_comparison.py
python wwtp_process_extraction/figure_s2_method_comparison.py
python wwtp_process_extraction/figure_s3_category_f1_method.py
```

## Contact 

Daly Wettermark - dalyw@stanford.edu

Constance Rouffet - rouffetc@stanford.edu

Fletcher Chapin - fchapin@stanford.edu

Ashley Ramirez - ashlecr3@uci.edu

Meagan Mauter - mauter@stanford.edu

## Acknowledgements

This work is funded in part by:
Stanford Woods Institute for the Environment's Realizing Environmental Innovation Program (REIP)
Stanford SURGE program
National Alliance for Water Innovation

We acknowledge the use of Claude Code and Anthropic LLMs for drafting and refining elements of web scraping, data manipulation, and visualization code throughout the codebase.