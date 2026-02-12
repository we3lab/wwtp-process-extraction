# NPDES Permit Analysis Tools 
These tools are designed to access unit process data from the National Pollutant Discharge Elimination System (NPDES) permits for California.

## Description 
Researchers have utilized the CWNS to quantify greenhouse gas emissions. However, this data is infrequent, voluntary, and sparse. To address these limitations, we utilize NPDES permits. The following Python tools are used to collect its data:


0. Build CWNS process tables
    - [npdes_permits/step0_build_cwns_table.py](npdes_permits/step0_build_cwns_table.py): creates unit_processes_by_facility.csv from CWNS 2004/2008/2012 data. 2022 doesn't include CA

1. Scrape permits and site metadata
    - [npdes_permits/step1_scrape_npdes.py](npdes_permits/scrape_npdes.py): downloads NPDES permit PDFs and writes site_data.csv and matched_cwns_npdes_ca.csv
    
    Uses npdes_detection.py helpers to detect which files are actually NPDES

2. Detect treatment processes in permits with keyword search
    - [npdes_permits/step2_search_npdes_text.py](npdes_permits/search_npdes_text.py): scans PDFs against unitprocess_keywords and writes unit_processes.csv with present/future status
    
3. Detect treatment processes in permits with LLM search
    - [npdes_permits/LLM_extraction/batch_query.py](npdes_permits/LLM_extraction/batch_query.py): runs LLM pipeline for all PDFs and writes llm_results/
    
    Uses llm_rag_loader.py helper for loading PDFs, extracting sections, and indexing text in Chroma

4. Post-process LLM output back to CWNS format

5. Compare NPDES text extraction vs CWNS survey data
    - [npdes_permits/step5a_compare_aggregate_results.py](npdes_permits/step5a_compare_aggregate_results.py): compares unit_processes.csv to unit_processes_by_facility.csv with bar chart comparisons
    - [npdes_permits/step5b_compare_facility_results.py](npdes_permits/step5b_compare_facility_results.py): compares unit_processes.csv to unit_processes_by_facility.csv (CA facilities matched by permit number) - facility-by-facility accuracy metrics (missed/hallucinated processes)

## How to Run
Executing from the repository root directory:

```bash
python npdes_permits/step0_build_cwns_table.py
python npdes_permits/step1_scrape_npdes.py
python npdes_permits/step2_search_npdes_text.py
python npdes_permits/LLM_extraction/batch_query.py
python npdes_permits/step4_postprocess_llm_output.py
python npdes_permits/step5a_compare_aggregate_results.py
python npdes_permits/step5b_compare_facility_results.py
```

## Known issues and limitations
- When first running permit_scrape.py, a Timeout Error may appear. Continue to rerun until the program successfully opens ChromeDrive **(Ensure that the "MM-DD-YYYY" Folder is deleted before rerunning)**.
- There are two distinct locations where permit_scrape.py is slow:
    1. After Region selection
    2. Selection of "ALL" Display range
- ...

## Contact 

Constance Rouffet - rouffetc@stanford.edu

Ashley Ramirez - ashlecr3@uci.edu

Daly Wettermark - dalyw@stanford.edu

Fletcher Chapin - fchapin@stanford.edu

## Acknowledgements

This work is funded in part by:
Stanford SURGE program
National Alliance for Water Innovation
