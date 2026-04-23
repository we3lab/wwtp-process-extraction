import argparse
import json
import os
import shutil
import re
from pathlib import Path

import pandas as pd

from helpers.npdes_text_extraction import extract_permit_sections
from helpers.ontology_to_txt import ontology_to_txt, output_file as ONTOLOGY_GENERATED_PATH
from helpers.api_llm_search import (
    build_pdf_jobs,
    chat_completion_json,
    load_icl_examples,
    init_unit_process_list_from_json,
    build_example_schema,
    get_method_paths,
)

PDF_DIR = "npdes_permits/output/2026-2-18/npdes"
MODEL = "gemini-2.0-flash-001"  # in claude-3-haiku, claude-4-5-sonnet, gemini-2.0-flash-001, gpt-5, gpt-5-mini, gemini-2.5-pro
DEFAULT_ONTOLOGY_PATH = "npdes_permits/data/llm_extraction/input/ontology.txt"
DEFAULT_FACILITIES_INFO_PATH = "npdes_permits/data/test_data.csv"
DEFAULT_UNITPROCESS_KEYWORDS_JSON = "npdes_permits/data/unitprocess_keywords.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run wastewater treatment extraction with configurable LLM prompt methods."
    )
    parser.add_argument(
        "--method",
        choices=["ontology-based", "list-based"],
        default="ontology-based",
        help="Prompting/extraction method to use (default: ontology-based).",
    )
    parser.add_argument(
        "--init_ontology",
        action="store_true",
        help="Refresh ontology text file from source ontology repository and exit.",
    )
    parser.add_argument(
        "--init_unit_process_list",
        action="store_true",
        help="Initialize unit process list text from unitprocess_keywords.json and exit.",
    )
    parser.add_argument(
        "--model",
        default=MODEL,
        help=f"Model name for API calls in claude-3-haiku, claude-4-5-sonnet, gemini-2.0-flash-001, gpt-5, gpt-5-mini, gemini-2.5-pro (default: {MODEL}).",
    )
    parser.add_argument(
        "--pdf_folder",
        default=PDF_DIR,
        help=f"Path to folder containing permit PDFs (default: {PDF_DIR}).",
    )
    parser.add_argument(
        "--facilities_information",
        default=DEFAULT_FACILITIES_INFO_PATH,
        help=(
            "Path to CSV with columns Facility_Name and PDF_File. "
            f"Each row is processed and saved separately, even when multiple facilities share the same PDF (default: {DEFAULT_FACILITIES_INFO_PATH})."
        ),
    )
    parser.add_argument(
        "--ontology_path",
        default=DEFAULT_ONTOLOGY_PATH,
        help=f"Path to ontology text file (default: {DEFAULT_ONTOLOGY_PATH}).",
    )
    parser.add_argument(
        "--unitprocess_keywords_json",
        default=DEFAULT_UNITPROCESS_KEYWORDS_JSON,
        help=(
            "Path to unitprocess_keywords.json used to initialize "
            "the list-based unit process text file."
        ),
    )
    parser.add_argument(
        "--unit_process_list_path",
        default=None,
        help="Optional override path for list-based unit process text file.",
    )
    parser.add_argument(
        "--system_prompt_path",
        default=None,
        help="Optional override path for system prompt template file.",
    )
    parser.add_argument(
        "--icl_examples_dir",
        default=None,
        help="Optional override folder for ICL example files named exampleN.txt.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional override output directory.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=1,
        help="Number of in-context learning examples to include (default: 1).",
    )
    parser.add_argument(
        "--skip_schema_validation",
        action="store_true",
        help="Skip local JSON schema validation of the model response.",
    )
    parser.add_argument(
        "--no_token_limit",
        action="store_true",
        help=(
            "Do not send max token limits to the API and do not skip long permit extracts. "
            "Server-side/model limits may still apply."
        ),
    )
    return parser.parse_args()


def render_system_message(
    template_text: str,
    facility_name: str,
    reference_text: str,
    prompt_examples: str,
    reference_placeholder: str,
) -> str:
    return (
        template_text
        .replace("__FACILITY_NAME__", facility_name)
        .replace(reference_placeholder, reference_text)
        .replace("__ONTOLOGY__", reference_text)
        .replace("__UNIT_PROCESS_LIST__", reference_text)
        .replace("__PROMPT_EXAMPLES__", prompt_examples)
    )


def slugify(text):
    slug = re.sub(r'[^A-Za-z0-9]+', '_', str(text or '').strip())
    slug = re.sub(r'_+', '_', slug).strip('_')
    return slug or 'facility'


def resolve_method_paths(args):
    method_paths = get_method_paths(args.method)

    if args.method == "ontology-based":
        reference_path = Path(args.ontology_path)
    else:
        reference_path = Path(args.unit_process_list_path or method_paths["reference_path"])

    prompt_path = Path(args.system_prompt_path or method_paths["prompt_path"])
    examples_dir = Path(args.icl_examples_dir or method_paths["examples_dir"])
    output_dir = Path(args.output_dir or method_paths["output_dir"])

    return method_paths, reference_path, prompt_path, examples_dir, output_dir


if __name__ == "__main__":
    args = parse_args()
    method_paths, reference_path, prompt_path, examples_dir, output_dir = resolve_method_paths(args)

    if args.init_ontology:
        print("Initializing ontology from source repository...")
        ontology_to_txt()
        generated_path = Path(ONTOLOGY_GENERATED_PATH)
        ontology_target = Path(args.ontology_path)
        if generated_path.exists() and generated_path.resolve() != ontology_target.resolve():
            ontology_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(generated_path, ontology_target)
            print(f"Copied generated ontology file to {ontology_target}")
        print("Ontology initialization completed. Exiting without running LLM extraction.")
        raise SystemExit(0)

    if args.method == "list-based" and (
        args.init_unit_process_list or not reference_path.exists()
    ):
        generated_list_path = init_unit_process_list_from_json(
            keywords_json_path=args.unitprocess_keywords_json,
            output_txt_path=str(reference_path),
        )
        print(f"Initialized unit process list file: {generated_list_path}")
        if args.init_unit_process_list:
            print("Unit process list initialization completed. Exiting.")
            raise SystemExit(0)

    if not reference_path.exists() or not reference_path.is_file():
        raise FileNotFoundError(f"Reference file not found: {reference_path}")

    if not prompt_path.exists() or not prompt_path.is_file():
        raise FileNotFoundError(f"System prompt template not found: {prompt_path}")

    os.makedirs(output_dir, exist_ok=True)
    token_usage_rows = []
    manifest_rows = []
    jobs = build_pdf_jobs(args.pdf_folder, args.facilities_information)

    if not jobs:
        print(
            "No facilities were processed. "
            f"Check --facilities_information ({args.facilities_information}) and --pdf_folder ({args.pdf_folder})."
        )
        raise SystemExit(0)

    reference_text = reference_path.read_text(encoding="utf-8")
    prompt_examples = load_icl_examples(
        num_examples=args.num_examples,
        examples_dir=str(examples_dir),
    )
    system_prompt_template = prompt_path.read_text(encoding="utf-8")
    example_schema = None if args.skip_schema_validation else build_example_schema(args.method)

    facilities_source_df = pd.read_csv(args.facilities_information, dtype=str).fillna('')

    for row_idx, pdf_path, pdf_file, facility_name in jobs:
        print("#" * 80)
        print(f"\nProcessing {pdf_file} for facility {facility_name}...")
        print("Extracting permit sections from PDF " f"{pdf_path}...")
        permit_section = extract_permit_sections(str(pdf_path))
        if permit_section is None:
            print("Failed to extract permit sections, aborting.")
            raise SystemExit(1)

        txt_section = permit_section.get("txt_section", "")
        txt_changes = permit_section.get("txt_changes", "")
        combined_text = (txt_section + "\n\n" + txt_changes).strip()
        metadata = permit_section.get("metadata", {})
        print(
            f"Extracted text section (length {len(txt_section)}), "
            f"changes section (length {len(txt_changes)}) and extract text "
            f"(length {len(combined_text)}). Metadata: {metadata}"
        )

        permit_extract = combined_text
        if not args.no_token_limit and len(permit_extract) > 30000:
            print(
                f"Warning: Extracted text length ({len(permit_extract)}) exceeds typical token limits. "
                "Consider truncating or summarizing the text for better results."
            )
            continue

        system_msg = render_system_message(
            template_text=system_prompt_template,
            facility_name=facility_name,
            reference_text=reference_text,
            prompt_examples=prompt_examples,
            reference_placeholder=method_paths["reference_placeholder"],
        )

        user_msg = f"""Find all the treatment processes explicitely used in the {facility_name} facility. Here is the permit extract:
        {permit_extract}
        """

        try:
            result = chat_completion_json(
                model=args.model,
                system_message=system_msg,
                user_message=user_msg,
                temperature=0.0,
                max_tokens=None if args.no_token_limit else 10000,
                max_completion_tokens=None if args.no_token_limit else 20000,
                schema=example_schema,
            )
            parsed, completion_token, prompt_token, total_token, reasoning_tokens = result
            print("Parsed JSON result:")
            print(json.dumps(parsed, indent=2, ensure_ascii=False))

            facility_slug = slugify(facility_name)
            pdf_stem = Path(pdf_file).stem
            extraction_file_name = f"{pdf_stem}__{row_idx + 1:04d}__{facility_slug}.json"
            output_json_path = output_dir / extraction_file_name
            with open(output_json_path, "w", encoding="utf-8") as output_file:
                json.dump(parsed, output_file, ensure_ascii=False, indent=2)

            print(f"Saved {output_json_path}")
            print(
                f"Token usage: completion={completion_token}, prompt={prompt_token}, "
                f"total={total_token}, reasoning={reasoning_tokens}"
            )

            token_usage_rows.append(
                {
                    "facility_name": facility_name,
                    "pdf_file": pdf_file,
                    "extraction_file": extraction_file_name,
                    "completion_token": completion_token,
                    "prompt_toke": prompt_token,
                    "total_token": total_token,
                    "reasoning_token": reasoning_tokens,
                }
            )

            if 0 <= row_idx < len(facilities_source_df):
                facility_row = facilities_source_df.iloc[row_idx].to_dict()
            else:
                facility_row = {}
            facility_row["pdf_file"] = pdf_file
            facility_row["extraction_file"] = extraction_file_name
            manifest_rows.append(facility_row)
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
    token_usage_csv_path = output_dir / "token_usage_summary.csv"
    token_usage_df.to_csv(token_usage_csv_path, index=False)
    print(f"Saved token usage CSV: {token_usage_csv_path}")

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_csv_path = output_dir / "facility_extraction_manifest.csv"
    manifest_df.to_csv(manifest_csv_path, index=False)
    print(f"Saved facility manifest CSV: {manifest_csv_path}")
