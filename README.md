# WWTP Unit Process Extraction from Permits Tool
These tools are designed to access unit process data from the National Pollutant Discharge Elimination System (NPDES) permits for California.

## Description 
Researchers have utilized the CWNS to aggregate WWTP unit processes. However, this data is infrequent, voluntary, and sparse. To address these limitations, we utilize regulatory permits. The following Python tools are used to collect its data:


1. Build CWNS process tables
    - [wwtp_process_extraction/step1_build_cwns_table.py](wwtp_process_extraction/step1_build_cwns_table.py): creates unit_processes_by_facility_cwns.csv from CWNS 2004/2008/2012 data. 2022 doesn't include CA

2. Scrape permits and site metadata
    - [wwtp_process_extraction/step2_scrape_npdes.py](wwtp_process_extraction/step2_scrape_npdes.py): downloads NPDES permit PDFs and writes site_data_relevant and matched_cwns_npdes_ca.csv

3. Extract permit text
    - [wwtp_process_extraction/step3_get_facility_descriptions.py](wwtp_process_extraction/step3_get_facility_descriptions.py): extracts relevant text sections from permit PDFs into per-facility text files

4. Detect treatment processes with keyword search
    - [wwtp_process_extraction/step4_keyword_extraction.py](wwtp_process_extraction/step4_keyword_extraction.py): scans permit text against unitprocess_keywords and writes kw_unit_processes.csv with present/future status

5. Detect treatment processes with LLM extraction
    - [wwtp_process_extraction/step5_llm_extraction.py](wwtp_process_extraction/step5_llm_extraction.py): runs LLM extraction on permit text
        - use *--method ontology-based* (default) or *--method list-based* to select prompting strategy
        - use *--model "model_name" --txt_folder "path_to_txt_folder" --facilities_information "path_to_facilities_csv"*
        - results saved as JSON under output/date/llm_search_ontology or llm_search_list

6. Post-process LLM output back to CWNS format
    - [wwtp_process_extraction/step6_postprocess_llm_output.py](wwtp_process_extraction/step6_postprocess_llm_output.py): post-processes LLM outputs and writes unit_processes_by_facility_llm.csv with present/future/past status

**Figures and tables**
- [wwtp_process_extraction/figure_2_data_source_comparison.py](wwtp_process_extraction/figure_2_data_source_comparison.py): ground-truth comparison of extraction methods
- [wwtp_process_extraction/figure_3_source_comparison.py](wwtp_process_extraction/figure_3_source_comparison.py): CWNS vs LLM unit process detection comparison
- [wwtp_process_extraction/figure_4_ca_ghg_comparison.py](wwtp_process_extraction/figure_4_ca_ghg_comparison.py): computes CA WWTP GHG emissions across three treatment process data sources (El Abbadi all-sources, El Abbadi CWNS-only, LLM permit scraping)
- [wwtp_process_extraction/figure_5_ca_needs_comparison.py](wwtp_process_extraction/figure_5_ca_needs_comparison.py): CA treatment capital needs rebuilt from permit-extracted planned changes using EPA's CWNS 2022 cost curves, against CWNS's own reported needs
- [wwtp_process_extraction/table_1_evaluate_model_performance.py](wwtp_process_extraction/table_1_evaluate_model_performance.py): evaluates LLM model performance against truth labels
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
python wwtp_process_extraction/step3_get_facility_descriptions.py
python wwtp_process_extraction/step4_keyword_extraction.py
# MODEL COMPARISON (all manual-read facilities, default FACILITIES_INFO_PATH)
python wwtp_process_extraction/step5_llm_extraction.py --all_models
# WEB SEARCH (claude-sonnet-4-6)
python wwtp_process_extraction/step5_llm_extraction.py  --model claude-sonnet-4-6 --web_search --all_methods
# WATERRAG (gpt-5-mini)
# Run step5b must in a SEPARATE Python 3.11 env with torch/langchain/faiss. 
# wwtp-process-extraction uses Python 3.12 which has no torch
# --overwrite; it redoes all facilities
conda run --no-capture-output -n waterrag python wwtp_process_extraction/step5b_waterrag_retrieval.py
python wwtp_process_extraction/step5_llm_extraction.py --method ontology-based --model gpt-5-mini --waterrag_context
# FULL CA, ONTOLOGY-BASED, GPT-5-MINI
python wwtp_process_extraction/step5_llm_extraction.py --all_facilities
# F1 VARIANCE: 2 extra benchmark runs
python wwtp_process_extraction/step5_llm_extraction.py --repeat_runs 2
python wwtp_process_extraction/step6_postprocess_llm_output.py
python wwtp_process_extraction/table_1_evaluate_model_performance.py
python wwtp_process_extraction/figure_2_data_source_comparison.py
python wwtp_process_extraction/figure_3_extraction_comparison.py
python wwtp_process_extraction/figure_4_ca_ghg_comparison.py
# CA capital needs from permit-extracted planned changes, costed with EPA's CWNS 2022
# cost curves (data/cwns_cost_curves.json, from the Cost Estimation Tool Methods Table 2-3)
python wwtp_process_extraction/figure_5_ca_needs_comparison.py
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