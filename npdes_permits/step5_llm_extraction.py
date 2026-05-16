import argparse
import json
import os
import shutil
import re
import subprocess
from pathlib import Path

import pandas as pd

from helpers.ontology_to_txt import ontology_to_txt, output_file as ONTOLOGY_GENERATED_PATH
from helpers.api_llm_search import (
    build_pdf_jobs,
    chat_completion_json,
    load_icl_examples,
    init_unit_process_list_from_json,
    build_example_schema,
    get_method_paths,
)

TXT_DIR = "npdes_permits/output/2026-4-26/npdes/text"
MODEL = "gpt-5-mini"  # in claude-3-haiku, claude-4-5-sonnet, gemini-2.0-flash-001, gpt-5, gpt-5-mini, gemini-2.5-pro
ONTOLOGY_PATH = "npdes_permits/data/llm_extraction/input/ontology.txt"
FACILITIES_INFO_PATH = "npdes_permits/output/2026-4-26/site_data.csv"
UNITPROCESS_KEYWORDS_JSON = "npdes_permits/data/unitprocess_keywords.json"
NUM_ICL_EXAMPLES = 1


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
        "--model",
        default=MODEL,
        help=f"Model name for API calls in claude-3-haiku, claude-4-5-sonnet, gemini-2.0-flash-001, gpt-5, gpt-5-mini, gemini-2.5-pro (default: {MODEL}).",
    )
    parser.add_argument(
        "--txt_folder",
        default=TXT_DIR,
        help=f"Path to folder containing permit TXT files (default: {TXT_DIR}).",
    )
    parser.add_argument(
        "--facilities_information",
        "--facilities_info_path",
        default=FACILITIES_INFO_PATH,
        help=(
            "Path to CSV with columns Facility Name and PDF_File. "
            f"Each row is processed and saved separately, even when multiple facilities share the same PDF (default: {FACILITIES_INFO_PATH})."
        ),
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
    parser.add_argument(
        "--web_search",
        action="store_true",
        help=(
            "Use Claude Code CLI (claude -p) with WebSearch/WebFetch tools instead of the "
            "Stanford proxy. Tracks cost_usd instead of token counts."
        ),
    )
    parser.add_argument(
        "--max_facilities",
        type=int,
        default=None,
        help="Limit number of facilities processed (useful for testing).",
    )
    return parser.parse_args()


def resolve_output_dir(method, model, web_search, txt_folder, facilities_information):
    if "model_comparison" in str(facilities_information):
        suffix = f"{method}_{model}" + ("-web" if web_search else "")
        return Path("npdes_permits/output/llm_model_comparison") / suffix

    # Derive date from txt_folder path (e.g. output/2026-4-26/npdes/text → 2026-4-26)
    date_segment = next(
        (p for p in Path(txt_folder).parts if re.match(r'\d{4}-\d{1,2}-\d{1,2}', p)),
        None,
    )
    if date_segment:
        return Path("npdes_permits/output") / date_segment / "llm_extraction"

    return Path("npdes_permits/output/llm_extraction")


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


_WEB_SYSTEM_SUFFIX = """
For each extracted item, set the Source field:
- "permit_text": evidence is only in the permit extract
- "web_search": evidence found via web search
- "both": evidence supported by both sources

For items with Source "web_search" or "both", also set the Website field to the URL or website name where you found the information.
If from multiple sites, pick the most relevant one. Set Website to null if the source is only the permit text.

Search the web for additional information about this facility when the permit text is unclear or incomplete.
"""


def chat_completion_web(
    model: str,
    system_message: str,
    user_message: str,
    schema: dict,
) -> tuple:
    """Call claude CLI with WebSearch/WebFetch. Returns (parsed_json, cost_usd)."""
    cmd = [
        "claude", "-p", user_message,
        "--system-prompt", system_message,
        "--allowedTools", "WebSearch,WebFetch",
        "--json-schema", json.dumps(schema),
        "--output-format", "json",
        "--model", model,
        "--no-session-persistence",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"CLI failed: {result.stderr[:200]}")
    if not result.stdout.strip():
        raise RuntimeError("empty output from CLI")
    return _parse_claude_json(result.stdout, schema)


def _parse_claude_json(stdout: str, schema: dict) -> tuple:
    """Parse JSON result from claude CLI output, handling multiple objects and markdown wrapping."""
    output = None
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(stdout):
        try:
            obj, end_idx = decoder.raw_decode(stdout, idx)
            if isinstance(obj, dict) and obj.get("type") == "result":
                output = obj
                break
            idx = end_idx
            while idx < len(stdout) and stdout[idx].isspace():
                idx += 1
        except json.JSONDecodeError:
            break
    if not output:
        raise RuntimeError("No result found in CLI output")
    if output.get("is_error"):
        raise RuntimeError(f"API error: {output.get('api_error_status')}")
    raw = output["result"]
    if isinstance(raw, dict):
        parsed = raw
    elif isinstance(raw, str) and raw.strip().startswith("{"):
        parsed = json.loads(raw.strip())
    else:
        # Extract JSON from markdown wrapper or empty result
        if not raw or not raw.strip():
            raise RuntimeError("Model returned empty result (possibly hit timeout during web search)")
        import re
        match = re.search(r'\{.*\}', str(raw), re.DOTALL)
        if not match:
            raise RuntimeError(f"No JSON found in result: {str(raw)[:100]}")
        parsed = json.loads(match.group())
    return parsed, float(output.get("total_cost_usd") or 0.0)


if __name__ == "__main__":
    args = parse_args()
    method_paths = get_method_paths(args.method)
    reference_path = Path(method_paths["reference_path"])
    prompt_path = Path(method_paths["prompt_path"])
    examples_dir = Path(method_paths["examples_dir"])
    output_dir = resolve_output_dir(
        args.method, args.model, args.web_search, args.txt_folder, args.facilities_information
    )

    if args.method == "ontology-based":
        print("Initializing ontology from source repository...")
        ontology_to_txt()
        generated_path = Path(ONTOLOGY_GENERATED_PATH)
        ontology_target = Path(ONTOLOGY_PATH)
        if generated_path.exists() and generated_path.resolve() != ontology_target.resolve():
            ontology_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(generated_path, ontology_target)
            print(f"Copied generated ontology file to {ontology_target}")

    if args.method == "list-based":
        generated_list_path = init_unit_process_list_from_json(
            keywords_json_path=UNITPROCESS_KEYWORDS_JSON,
            output_txt_path=str(reference_path),
        )
        print(f"Initialized unit process list file: {generated_list_path}")

    os.makedirs(output_dir, exist_ok=True)
    token_usage_rows = []
    manifest_rows = []
    jobs = build_pdf_jobs(args.txt_folder, args.facilities_information)

    if not jobs:
        print(
            "No facilities were processed. "
            f"Check --facilities_information ({args.facilities_information}) and --txt_folder ({args.txt_folder})."
        )
        raise SystemExit(0)

    reference_text = reference_path.read_text(encoding="utf-8")
    prompt_examples = load_icl_examples(
        num_examples=NUM_ICL_EXAMPLES,
        examples_dir=str(examples_dir),
    )
    system_prompt_template = prompt_path.read_text(encoding="utf-8")
    example_schema = None if args.skip_schema_validation else build_example_schema(args.method, web=args.web_search)

    facilities_source_df = pd.read_csv(args.facilities_information, dtype=str).fillna('')

    if args.max_facilities is not None:
        jobs = jobs[:args.max_facilities]

    for row_idx, txt_path_candidate, txt_file, facility_name in jobs:
        print("#" * 80)
        print(f"\nProcessing {txt_file} for facility {facility_name}...")

        txt_path = Path(txt_path_candidate)
        if not txt_path.exists():
            print(f"TXT file not found: {txt_path}. Skipping.")
            continue

        print(f"Reading text from {txt_path}...")
        permit_extract = txt_path.read_text(encoding="utf-8")
        print(f"Read text extract (length {len(permit_extract)})")

        if not args.no_token_limit and len(permit_extract) > 30000:
            print(
                f"Warning: Extracted text length ({len(permit_extract)}) exceeds typical token limits. "
                "Consider truncating or summarizing the text for better results."
            )
            continue

        txt_stem = txt_path.stem
        extraction_file_name = f"{txt_stem}_{slugify(facility_name)}.json"
        output_json_path = output_dir / extraction_file_name

        if output_json_path.exists():
            print(f"Already processed: {output_json_path.name}, skipping.")
            continue

        system_msg = render_system_message(
            template_text=system_prompt_template,
            facility_name=facility_name,
            reference_text=reference_text,
            prompt_examples=prompt_examples,
            reference_placeholder=method_paths["reference_placeholder"],
        )
        if args.web_search:
            system_msg = system_msg + _WEB_SYSTEM_SUFFIX

        user_msg = f"""Find all the treatment processes explicitely used in the {facility_name} facility. Here is the permit extract:
        {permit_extract}
        """
        if args.web_search:
            user_msg = (
                f"Search the web for information about {facility_name} wastewater treatment "
                f"facility to supplement the permit extract below.\n\n"
                + user_msg
                + "\n\nIMPORTANT: Your entire response must be ONLY a valid JSON object matching the schema. No other text, no explanation, no markdown."
            )

        try:
            if args.web_search:
                parsed, cost_usd = chat_completion_web(
                    model=args.model,
                    system_message=system_msg,
                    user_message=user_msg,
                    schema=example_schema,
                )
                completion_token = prompt_token = total_token = reasoning_tokens = 0
                print("Parsed JSON result:")
                print(json.dumps(parsed, indent=2, ensure_ascii=False))
                print(f"Cost: ${cost_usd:.6f}")
            else:
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
                cost_usd = None
                print("Parsed JSON result:")
                print(json.dumps(parsed, indent=2, ensure_ascii=False))
                print(
                    f"Token usage: completion={completion_token}, prompt={prompt_token}, "
                    f"total={total_token}, reasoning={reasoning_tokens}"
                )

            with open(output_json_path, "w", encoding="utf-8") as output_file:
                json.dump(parsed, output_file, ensure_ascii=False, indent=2)

            print(f"Saved {output_json_path}")

            token_usage_rows.append(
                {
                    "facility_name": facility_name,
                    "txt_file": txt_file,
                    "extraction_file": extraction_file_name,
                    "completion_token": completion_token,
                    "prompt_token": prompt_token,
                    "total_token": total_token,
                    "reasoning_token": reasoning_tokens,
                    "cost_usd": cost_usd,
                }
            )

            if 0 <= row_idx < len(facilities_source_df):
                facility_row = facilities_source_df.iloc[row_idx].to_dict()
            else:
                facility_row = {}
            facility_row["txt_file"] = txt_file
            facility_row["extraction_file"] = extraction_file_name
            manifest_rows.append(facility_row)
        except Exception as exc:
            print("Error:", exc)

    token_usage_df = pd.DataFrame(
        token_usage_rows,
        columns=[
            "facility_name",
            "txt_file",
            "extraction_file",
            "completion_token",
            "prompt_token",
            "total_token",
            "reasoning_token",
            "cost_usd",
        ],
    )
    token_usage_csv_path = output_dir / "token_usage_summary.csv"
    token_usage_df.to_csv(token_usage_csv_path, index=False)
    print(f"Saved token usage CSV: {token_usage_csv_path}")

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_csv_path = output_dir / "facility_extraction_manifest.csv"
    manifest_df.to_csv(manifest_csv_path, index=False)
    print(f"Saved facility manifest CSV: {manifest_csv_path}")

# python npdes_permits/step3_llm_extraction_new.py \
#   --method ontology-based \
#   --model claude-sonnet-4-5 \
#   --web_search \
#   --facilities_information npdes_permits/data/model_comparison_facilities.csv

# python npdes_permits/step3_llm_extraction_new.py \
#   --method list-based \
#   --model claude-sonnet-4-5 \
#   --web_search \
#   --facilities_information npdes_permits/data/model_comparison_facilities.csv
