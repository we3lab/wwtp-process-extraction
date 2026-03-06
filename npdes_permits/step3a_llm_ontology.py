import argparse
import requests
import json
import os
import shutil
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional
from helpers.npdes_text_extraction import extract_permit_sections
from helpers.ontology_to_txt import ontology_to_txt, output_file as ONTOLOGY_GENERATED_PATH

BASE_URL = "https://aiapi-prod.stanford.edu/v1"


def get_headers():
    api_key = Path("npdes_permits/API_key.txt").read_text(encoding="utf-8").strip()
    if not api_key:
        raise ValueError("API_key.txt is empty.")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

TEST_DATA = pd.read_csv("npdes_permits/data/test_data.csv")
PDF_DIR = "npdes_permits/output/2026-2-18/npdes"
OUTPUT_DIR = "npdes_permits/output/2026-2-18/llm_search_ontology"
MODEL = "gemini-2.0-flash-001"  # in claude-3-haiku, claude-4-5-sonnet, gemini-2.0-flash-001, gpt-5, gpt-5-mini, gemini-2.5-pro
DEFAULT_ONTOLOGY_PATH = "npdes_permits/data/ontology.txt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run wastewater treatment extraction with ontology-constrained LLM prompts."
    )
    parser.add_argument(
        "--init_ontology",
        action="store_true",
        help="Refresh ontology text file from source ontology repository and exit.",
    )
    parser.add_argument(
        "--model",
        default=MODEL,
        help=f"Model name for API calls in claude-3-haiku, claude-4-5-sonnet, gemini-2.0-flash-001, gpt-5, gpt-5-mini, gemini-2.5-pro (default: {MODEL}).",
    )
    parser.add_argument(
        "--pdf",
        default=PDF_DIR,
        help="Path to one PDF file or to a folder containing PDF files.",
    )
    parser.add_argument(
        "--ontology_path",
        default=DEFAULT_ONTOLOGY_PATH,
        help=f"Path to ontology text file (default: {DEFAULT_ONTOLOGY_PATH}).",
    )
    return parser.parse_args()


def build_pdf_jobs(pdf_arg: str):
    pdf_path = Path(pdf_arg)
    test_data_lookup = {
        str(row["PDF_File"]): str(row["Facility_Name"])
        for _, row in TEST_DATA.iterrows()
    }

    if pdf_path.is_file():
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Provided --pdf file is not a PDF: {pdf_path}")
        pdf_name = pdf_path.name
        facility_name = test_data_lookup.get(pdf_name, pdf_path.stem)
        return [(pdf_path, pdf_name, facility_name)]

    if pdf_path.is_dir():
        pdf_files = sorted([path for path in pdf_path.iterdir() if path.suffix.lower() == ".pdf"])
        jobs = []
        for path in pdf_files:
            pdf_name = path.name
            facility_name = test_data_lookup.get(pdf_name, path.stem)
            jobs.append((path, pdf_name, facility_name))
        return jobs

    raise ValueError(f"--pdf path does not exist: {pdf_path}")


def get_models():
    """GET available models"""
    url = f"{BASE_URL}/models"
    resp = requests.get(url, headers=get_headers())
    resp.raise_for_status()
    return resp.json()

def chat_completion_json(
    model: str,
    user_message: str,
    system_message: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 3000,
    max_completion_tokens: int = 8000,  # if you want to set max tokens for completion separately
    stop: Optional[Any] = None,
    retry_delay: float = 1.0,
    schema: Optional[Dict] = None,  # optional jsonschema to validate result
) -> Dict[str, Any]:
    """
    Request a JSON object from the model. Returns the parsed JSON (dict/list).
    Retries parsing/validation up to `retries` times if needed.
    """
    url = f"{BASE_URL}/chat/completions"

    messages = []
    if system_message is not None:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})

    effective_max_completion_tokens = min(max_tokens, max_completion_tokens)

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_completion_tokens": effective_max_completion_tokens,
        # Ask API to use JSON mode - guarantees valid JSON output (see docs).
        "response_format": {"type": "json_object"},
    }
    if stop is not None:
        payload["stop"] = stop

    try:
        resp = requests.post(url, headers=get_headers(), json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()

        # Extract assistant content string
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("No choices returned by the API.")
        first_choice = choices[0]
        finish_reason = first_choice.get("finish_reason")
        content = first_choice.get("message", {}).get("content")
        if content is None:
            raise ValueError("Assistant content is null.")
        usage = data.get("usage", {})
        completion_token = usage.get("completion_tokens", 0)
        prompt_token = usage.get("prompt_tokens", 0)
        total_token = usage.get("total_tokens", 0)
        completion_tokens_details = usage.get("completion_tokens_details", {})
        reasoning_tokens = completion_tokens_details.get("reasoning_tokens", 0)

        if finish_reason == "length":
            raise RuntimeError(
                "Generation stopped because token limit was reached "
                f"(finish_reason='length', completion_tokens={completion_token}, "
                f"max_tokens={max_tokens}, max_completion_tokens={effective_max_completion_tokens}). "
                "Increase limits or reduce prompt/output size."
            )

        if not content.strip():
            raise RuntimeError(
                "Model returned empty/whitespace content. "
                "Ensure the prompt explicitly requests JSON output and reduce requested output size if needed."
            )

        # content should be valid JSON string; try to parse
        parsed = json.loads(content)

        # Normalize optional list fields: convert [] to null for schema compatibility
        items = parsed.get("items") if isinstance(parsed, dict) else None
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                for field_name in ("Process", "Role", "Substance"):
                    if item.get(field_name) == []:
                        item[field_name] = None

        # Optional JSON Schema validation if scF_FILEhema provided and jsonschema is installed
        if schema is not None:
            try:
                import jsonschema
                jsonschema.validate(instance=parsed, schema=schema)
            except ImportError:
                # jsonschema not installed; warn but accept parsed JSON
                print("Warning: jsonschema not installed; skipping schema validation.")
            except jsonschema.ValidationError as ve:
                raise ValueError(f"JSON did not validate against schema: {ve}")

        return parsed, completion_token, prompt_token, total_token, reasoning_tokens

    except (requests.RequestException, ValueError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to get valid JSON. Last error: {e}")


if __name__ == "__main__":
    args = parse_args()
    ontology_path = Path(args.ontology_path)

    if args.init_ontology:
        print("Initializing ontology from source repository...")
        ontology_to_txt()
        generated_path = Path(ONTOLOGY_GENERATED_PATH)
        if generated_path.exists() and generated_path.resolve() != ontology_path.resolve():
            ontology_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(generated_path, ontology_path)
            print(f"Copied generated ontology file to {ontology_path}")
        print("Ontology initialization completed. Exiting without running LLM extraction.")
        raise SystemExit(0)

    if not ontology_path.exists():
        raise FileNotFoundError(
            f"Ontology file not found at {ontology_path}. "
            "Use --init_ontology to generate it or pass --ontology_path to an existing ontology.txt file."
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    token_usage_rows = []
    jobs = build_pdf_jobs(args.pdf)

    if not jobs:
        print(f"No PDF files found for --pdf {args.pdf}")
        raise SystemExit(0)

    ontology = open(ontology_path, encoding="utf-8").read()

    for pdf_path, pdf_file, facility_name in jobs:
        print("#" * 80)
        print(f"\nProcessing {pdf_file} for facility {facility_name}...")
        print("Extracting permit sections from PDF " f"{pdf_path}...")
        permit_section = extract_permit_sections(str(pdf_path))
        if permit_section is None:
            print("Failed to extract permit sections, aborting.")
            raise SystemExit(1)
        else:
            txt_section = permit_section.get("txt_section", "")
            txt_changes = permit_section.get("txt_changes", "")
            full_text = permit_section.get("full_text", "")
            combined_text = (txt_section + "\n\n" + txt_changes).strip()
            metadata = permit_section.get("metadata", {})
            print(f"Extracted text section (length {len(txt_section)}), changes section (length {len(txt_changes)}) and extract text (length {len(combined_text)}). Metadata: {metadata}")
            permit_extract = combined_text
            if len(permit_extract)>30000:
                print(f"Warning: Extracted text length ({len(permit_extract)}) exceeds typical token limits. Consider truncating or summarizing the text for better results.")
                continue

        system_msg = f"""           
        You are an expert in wastewater treatments. 

        Inputs:
        A wastewater treatment plant permit extract
        Ontology lists: equipments, processes and roles (with mandatory relationships)

        Task: 
        Using only the wastewater treatment plant’s permit extract, find all the planned, implemented or past treatments mentioned to be used in the {facility_name} facility. A treatment must be described by at least an equipment or a process from the given ontology lists.
        You may ignore irrelevant context.
        Think step by step:
        1. Extract explicitly from the permit text the treatments mentioned for the specific {facility_name} facility (if it is used in another facility or off-site it should be mentionned in the "location" field of the output JSON)
        2. Match each treatment find with an equipment and/or one or many processes from the ontology lists
        3. Complete with specification of the treatment with roles and substances if explicitly mentioned in the text
        4. Determine if the treatment is currently implemented and used in the facility (“present”), planned for the future (“planned”) or has been shut or removed (“past”)
        5. Structure the output as a JSON file as requested

            Output:
            Return only a JSON object with this exact shape:
            {{
                "items": [
                    {{
                        "Equipment": string|null,
                        "Process": string[]|null,
                        "Role": string[]|null,
                        "Substance": string[]|null,
                        "Implementation": "present"|"planned"|"past",
                        "Location": "on-site"|"off-site"|"third-party"|null,
                        "Score": number,
                        "Sentence": string
                    }}
                ]
            }}
            Do not add any other top-level fields.

            Field meaning and allowed cardinality:
            - Equipment: equipment name from ontology. Exactly one string or null.
            - Process: process names from ontology. Null or an array of 1+ strings.
            - Role: role names from ontology. Null or an array (can be empty).
            - Substance: substance names explicitly mentioned in text. Null or an array (can be empty).
            - Rule: at least one of Equipment or Process must be non-null.
            - Implementation: one of "present", "planned", "past".
            - Location: one of "on-site", "off-site", "third-party", or null if unspecified.
            - Score: confidence score between 0 and 1.
            - Sentence: direct supporting sentence/quote from permit extract.

        All equipment, process, role and substance names must be exactly as in the ontology lists.
        
        ########################################################################
        Ontology lists:
        ########################################################################
        {ontology}

        ########################################################################
        Examples of treatments mentioned in permit text and the corresponding JSON output:
        ########################################################################
        Example 1:
        Permit text extract: "Preliminary Treatment. Preliminary treatment consists of influent screens followed by grit removal. An iron salts dosing station doses ferric chloride at the Emergency Basin Overflow Structure for odor control.
            b. Primary Treatment. Following preliminary treatment, wastewater is pumped into rectangular primary clarifiers to remove floatable and settleable material.
            c. Biological Treatment. All wastewater receives biological treatment. A modified biological nutrient removal (BNR) process is employed that is designed to remove BOD and ammonia (NH3) in the same aeration basins. Each basin is divided into four sections referred to as “quads.” The first and third quads are operated under anoxic conditions, while the second and fourth quads are operated under aerobic conditions. This configuration achieves effective filament control and allows for some denitrification. Following biological treatment, wastewater is pumped to secondary clarifiers."
        JSON output:
        {{
            "items": [
                {{
                    "Equipment": "Screen",
                    "Process": ["Screening"],
                    "Role": ["Pretreatment"],
                    "Substance": null,
                    "Implementation": "present",
                    "Location": "on-site",
                    "Score": 1,
                    "Sentence": "Preliminary treatment consists of influent screens followed by grit removal."
                }},
                {{
                    "Equipment": "GritChamber",
                    "Process": ["Sedimentation"],
                    "Role": ["Pretreatment"],
                    "Substance": null,
                    "Implementation": "present",
                    "Location": "on-site",
                    "Score": 1,
                    "Sentence": "Preliminary treatment consists of influent screens followed by grit removal."
                }},
                {{
                    "Equipment": null,
                    "Process": ["ChemicalProcess"],
                    "Role": ["Pretreatment"],
                    "Substance": ["OdorControlAgent"],
                    "Implementation": "present",
                    "Location": "on-site",
                    "Score": 0.9,
                    "Sentence": "An iron salts dosing station doses ferric chloride at the Emergency Basin Overflow Structure for odor control."
                }},
                {{
                    "Equipment": "SedimentationTank",
                    "Process": ["Sedimentation"],
                    "Role": ["Primary"],
                    "Substance": null,
                    "Implementation": "present",
                    "Location": "on-site",
                    "Score": 0.9,
                    "Sentence": "...wastewater is pumped into rectangular primary clarifiers to remove floatable and settleable material."
                }},
                {{
                    "Equipment": "AerationBasin",
                    "Process": ["BiologicalProcess", "Nitrification", "Denitrification"],
                    "Role": ["NutrientRemoval", "NitrogenRemoval", "Anoxic", "Aerobic"],
                    "Substance": null,
                    "Implementation": "present",
                    "Location": "on-site",
                    "Score": 0.9,
                    "Sentence": "A modified biological nutrient removal (BNR) process is employed that is designed to remove BOD and ammonia (NH3) in the same aeration basins. Each basin is divided into four sections referred to as “quads.” The first and third quads are operated under anoxic conditions, while the second and fourth quads are operated under aerobic conditions. This configuration achieves effective filament control and allows for some denitrification"
                }},
                {{
                    "Equipment": "SedimentationTank",
                    "Process": ["Sedimentation"],
                    "Role": ["Secondary"],
                    "Substance": null,
                    "Implementation": "present",
                    "Location": "on-site",
                    "Score": 0.9,
                    "Sentence": " Following biological treatment, wastewater is pumped to secondary clarifiers."
                }}
            ]
        }}

        You shall use this example to understand how to extract the relevant information from the permit text and structure it in the required JSON format. The example is not exhaustive and you should adapt the extraction to the specific content of the permit extract provided for the {facility_name} facility. Always ensure that your output strictly follows the specified JSON structure and field requirements.
    """

        user_msg = f"""Find all the treatment processes explicitely used in the {facility_name} facility. Here is the permit extract: 
        {permit_extract}
        """

        # Optional JSON Schema (for extra validation)
        example_schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "Equipment": {"type": ["string", "null"]},
                            "Process": {
                                "anyOf": [
                                    {"type": "null"},
                                    {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "minItems": 1,
                                    },
                                ]
                            },
                            "Role": {
                                "anyOf": [
                                    {"type": "null"},
                                    {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                ]
                            },
                            "Substance": {
                                "anyOf": [
                                    {"type": "null"},
                                    {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                ]
                            },
                            "Implementation": {
                                "type": "string",
                                "enum": ["present", "planned", "past"],
                            },
                            "Location": {
                                "type": ["string", "null"],
                                "enum": ["on-site", "off-site", "third-party", None],
                            },
                            "Score": {"type": "number", "minimum": 0, "maximum": 1},
                            "Sentence": {"type": "string"},
                        },
                        "required": [
                            "Equipment",
                            "Process",
                            "Role",
                            "Substance",
                            "Implementation",
                            "Location",
                            "Score",
                            "Sentence",
                        ],
                        "not": {
                            "properties": {
                                "Equipment": {"type": "null"},
                                "Process": {"type": "null"}
                            },
                            "required": ["Equipment", "Process"]
                        },
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["items"],
            "additionalProperties": False,
        }

        try:
            result = chat_completion_json(
                model=args.model,
                system_message=system_msg,
                user_message=user_msg,
                temperature=0.0,
                max_tokens=4000,
                schema=example_schema,  # set to None to skip schema validation
            )
            parsed, completion_token, prompt_token, total_token, reasoning_tokens = result
            print("Parsed JSON result:")
            print(json.dumps(parsed, indent=2, ensure_ascii=False))
            # Save to file
            with open(f"{OUTPUT_DIR}/{pdf_file.replace('.pdf', '')}.json", "w", encoding="utf-8") as f:
                json.dump(parsed, f, ensure_ascii=False, indent=2)
            print("Saved assistant_output.json")
            print(f"Token usage: completion={completion_token}, prompt={prompt_token}, total={total_token}, reasoning={reasoning_tokens}")

            token_usage_rows.append(
                {
                    "facility_name": facility_name,
                    "pdf_file": pdf_file,
                    "completion_token": completion_token,
                    "prompt_toke": prompt_token,
                    "total_token": total_token,
                    "reasoning_token": reasoning_tokens,
                }
            )
        except Exception as exc:
            print("Error:", exc)

    token_usage_df = pd.DataFrame(
        token_usage_rows,
        columns=[
            "facility_name",
            "pdf_file",
            "completion_token",
            "prompt_toke",
            "total_token",
            "reasoning_token",
        ],
    )
    token_usage_csv_path = f"{OUTPUT_DIR}/token_usage_summary.csv"
    token_usage_df.to_csv(token_usage_csv_path, index=False)
    print(f"Saved token usage CSV: {token_usage_csv_path}")
