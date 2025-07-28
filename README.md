# npdes-permits
This repository is currently a work-in-progress; however, it will aim to achieve the following: 
1. Access and scrape PDFs and other data tables from water boards.
2. Processing data and identifying unit processes.
3. Utilizing an LLM to process PDF text for data. 
4. Merge with the existing clean water needs assessment/surveys.

## Unit Process Keywords

The `unitprocess_keywords.json` file categorizes wastewater treatment processes into hierarchical categories:

### Categories
- **Primary**: Initial treatment processes (e.g., clarifiers)
- **Secondary**: Biological treatment processes (e.g., activated sludge, trickling filters)
- **Tertiary**: Tertiary treatment processes (e.g., UV)
- **Disinfection**: Disinfection processes (e.g., chlorine contact tank)
- **Advanced**: Advanced treatment processes (e.g., UV-AOP)
- **Solids**: Solids handling processes (e.g., digestion)
- **Cogeneration**: Energy recovery processes

### Category Flags
If a wastewater treatment plant (WWTP) has ANY processes present within a category, it receives a `True` flag for that category. This allows for high-level classification of treatment capabilities.

### Add-ons
Processes listed under "add-ons" are not unique and can exist alongside other unit processes within the same category. For example, "stepfeed" can be combined with other secondary treatment processes.

### Alternative Names and Case Sensitivity
- **alt_names**: Alternative terms or variations for the same unit process (e.g., "Primary Settling Tank" for clarifier)
- **alt_names_case_sensitive**: Controls whether text matching is case-sensitive ("N" = case-insensitive, "Y" = case-sensitive)
- **CWNS_2012_code**: How this process is listed in the 2008, 2010, and 2012 CWNS dataset
- **CWNS_2022_code**: How this process is listed in the 2022 CWNS dataset