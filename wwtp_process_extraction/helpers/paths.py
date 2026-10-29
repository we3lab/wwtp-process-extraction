# Project paths and API settings, star-imported by the scripts (from helpers.paths import *)
from pathlib import Path

# Project paths, resolved from this file so scripts work from any directory
PACKAGE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PACKAGE_DIR / "data"
OUTPUT_DIR = PACKAGE_DIR / "output"
PERMITS_DIR = OUTPUT_DIR / "permits"
TXT_DIR = PERMITS_DIR / "text"
FIGURES_DIR = OUTPUT_DIR / "figures"
BY_CATEGORY_DIR = FIGURES_DIR / "by_category"
LLM_EXTRACTION_DIR = OUTPUT_DIR / "llm_extraction"
# The default ontology-based gpt-5-mini run, which holds the full CA set
MAIN_RUN_DIR = LLM_EXTRACTION_DIR / "ontology-based_gpt-5-mini"
WATERRAG_RETRIEVAL_DIR = OUTPUT_DIR / "waterrag_retrieval"
ONTOLOGY_TXT = DATA_DIR / "llm_extraction" / "input" / "ontology.txt"
CIWQS_TO_CWNS_CSV = DATA_DIR / "ciwqs_to_cwns.csv"
KEYWORDS_JSON = DATA_DIR / "unitprocess_keywords.json"
MANUAL_CSV = DATA_DIR / "unit_processes_by_facility_manual.csv"
SUPPLEMENTAL_CSV = DATA_DIR / "unit_processes_by_facility_supplemental_data.csv"
SUPPLEMENTAL_COMPARISON_CSV = OUTPUT_DIR / "supplemental_data_comparison_by_facility.csv"
SITE_DATA_ALL_CSV = OUTPUT_DIR / "site_data_all.csv"
SITE_DATA_RELEVANT_CSV = OUTPUT_DIR / "site_data_relevant.csv"
CWNS_TABLE_CSV = OUTPUT_DIR / "unit_processes_by_facility_cwns.csv"
PDF_KW_CSV = OUTPUT_DIR / "unit_processes_by_pdf_kw.csv"
FACILITY_KW_CSV = OUTPUT_DIR / "unit_processes_by_facility_kw.csv"
PDF_LLM_CSV = OUTPUT_DIR / "unit_processes_by_pdf_llm.csv"
FACILITY_LLM_CSV = OUTPUT_DIR / "unit_processes_by_facility_llm.csv"
# CWNS + permit rows merged by figure_3 (the SDR release CSV)
FACILITY_CSV = OUTPUT_DIR / "unit_processes_by_facility.csv"

# Stanford AI API gateway (step5, step5b)
API_URL = "https://aiapi-prod.stanford.edu/v1"
API_KEY_PATH = PACKAGE_DIR / "API_key.txt"
