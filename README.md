# NPDES Permit Analysis Tools 
These tools are designed to access unit process data from the National Pollutant Discharge Elimination System (NPDES) permits for California.

## Description 
Researchers have utilized the CWNS to quantify greenhouse gas emissions. However, this data is infrequent, voluntary, and sparse. To address these limitations, we utilize NPDES permits. The following Python tools are used to collect its data:


1. Scrape permits and site metadata
    - [npdes_permits/scrape_npdes.py](npdes_permits/scrape_npdes.py): downloads NPDES permit PDFs and writes site_data.csv and matched_cwns_npdes_ca.csv
    
    Uses npdes_detection.py helpers to detect which files are actually NPDES

2. Detect treatment processes in permits with keyword search
    - [npdes_permits/search_npdes_text.py](npdes_permits/search_npdes_text.py): scans PDFs against unitprocess_keywords and writes unit_processes.csv with present/future status
    
    Uses text_extraction.py helpers to extract Facility Description and Planned Changes sections

3. Detect treatment processes in permits with LLM search
    - [npdes_permits/LLM_extraction/batch_query.py](npdes_permits/LLM_extraction/batch_query.py): runs LLM pipeline for all PDFs and writes llm_results/
    
    Uses llm_rag_loader.py helper for loading PDFs, extracting sections, and indexing text in Chroma

4. Build CWNS process tables
    - [npdes_permits/el_abbadi_tt_assignment.py](npdes_permits/el_abbadi_tt_assignment.py): creates unit_processes_by_facility.csv from CWNS 2004/2008/2012 data. 2022 doesn't include CA
    - [npdes_permits/create_cwns_process_df.py](npdes_permits/create_cwns_process_df.py): creates unit_processes_by_facility.csv from CWNS 2012 Unit_Process_Details (CA facilities with NPDES permits only)

5. Compare NPDES text extraction vs CWNS survey data
    - [npdes_permits/compare_processes.py](npdes_permits/compare_processes.py): compares unit_processes.csv to unit_processes_by_facility.csv with bar chart comparisons
    - [npdes_permits/compare_cwns_unitprocesses.py](npdes_permits/compare_cwns_unitprocesses.py): compares unit_processes.csv to unit_processes_by_facility.csv (CA facilities matched by permit number) - facility-by-facility accuracy metrics (missed/hallucinated processes)


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
