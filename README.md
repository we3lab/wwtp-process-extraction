# NPDES Permit Analysis Tools 
Our permit_scrape.py and upi_tool.py are designed to access unit process data from the National Pollutant Discharge Elimination System (NPDES) permits. 
## Description 
Researchers have utilized the CWNS to quantify greenhouse gas emissions. However, this data is infrequent, voluntary, and sparse. To address these limitations, we utilize NPDES permits. The following Python tools are used to collect its data: (i) a web scraping tool and (ii) a PDF text scanner to extract detailed facility-level information from NPDES permits. 
## Getting Started 

## Known issues and limitations
- When first running permit_scrape.py, a Timeout Error may appear. Continue to rerun until the program successfully opens ChromeDrive **(Ensure that the "MM-DD-YYYY" Folder is deleted before rerunning)**.
- There are two distinct locations where permit_scrape.py is slow:
    1. After Region selection
    2. Selection of "ALL" Display range
- ...

## Contact 
Ashley Ramirez - ashlecr3@uci.edu

Daly Wettermark - 

## Acknowledgements
