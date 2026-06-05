import argparse
import json
import os
import shutil
import re
import subprocess
from pathlib import Path

import pandas as pd

from helpers.ontology_to_txt import ontology_to_txt, output_file as ONTOLOGY_GENERATED_PATH
from helpers.utils import build_txt_jobs, SEP
from helpers.api_llm_search import (
    chat_completion_json,
    load_icl_examples,
    init_unit_process_list_from_json,
    build_example_schema,
    get_method_paths,
)

TXT_DIR = f"wwtp_process_extraction/output/npdes/text"
MODEL = "gpt-5-mini"  # in claude-3-haiku, claude-4-5-sonnet, gpt-5, gpt-5-mini, gemini-2.5-pro
ONTOLOGY_PATH = "wwtp_process_extraction/data/llm_extraction/input/ontology.txt"
FACILITIES_INFO_PATH = f"wwtp_process_extraction/output/site_data_relevant.csv"
UNITPROCESS_KEYWORDS_JSON = "wwtp_process_extraction/data/unitprocess_keywords.json"
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
        help=f"Model name for API calls in claude-3-haiku, claude-4-5-sonnet, gpt-5, gpt-5-mini, gemini-2.5-pro (default: {MODEL}).",
    )
    parser.add_argument(
        "--txt_folder",
        default=TXT_DIR,
        help=f"Path to folder containing permit TXT files (default: {TXT_DIR}).",
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
    parser.add_argument(
        "--all_models",
        action="store_true",
        help="Loop over all models and methods (model_comparison mode). Ignores --model and --method.",
    )
    return parser.parse_args()


def resolve_output_dir(method, model, web_search, txt_folder):
    if "model_comparison" in str(FACILITIES_INFO_PATH):
        suffix = f"{method}_{model}" + ("-web" if web_search else "")
        return Path("wwtp_process_extraction/output/llm_model_comparison") / suffix

    # Derive date from txt_folder path (e.g. output/2026-5-25/npdes/text → 2026-5-25)
    date_segment = next(
        (p for p in Path(txt_folder).parts if re.match(r'\d{4}-\d{1,2}-\d{1,2}', p)),
        None,
    )
    if date_segment:
        return Path("wwtp_process_extraction/output") / date_segment / "llm_extraction"

    return Path("wwtp_process_extraction/output/llm_extraction")


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
            diag = {k: output.get(k) for k in ("subtype", "num_turns", "duration_ms", "total_cost_usd", "stop_reason")}
            raise RuntimeError(f"Model returned empty result (no final JSON emitted). CLI result meta: {diag}")
        import re
        match = re.search(r'\{.*\}', str(raw), re.DOTALL)
        if not match:
            raise RuntimeError(f"No JSON found in result: {str(raw)[:100]}")
        parsed = json.loads(match.group())
    return parsed, float(output.get("total_cost_usd") or 0.0)


ALL_MODELS = ["claude-3-haiku", "claude-4-5-sonnet", "gpt-5", "gpt-5-mini", "gemini-2.5-pro"]
DEFAULT_MAX_TOKENS = 10000
DEFAULT_MAX_COMPLETION_TOKENS = 20000
# Per-model completion-token limit. Reasoning models (gpt-5, etc.) spend most of
# their completion budget on hidden reasoning before emitting JSON, so they need a
# higher ceiling than chat models; claude-3-haiku has a hard 4096 output cap.
MAX_TOKENS_BY_MODEL = {"claude-3-haiku": 4096, "gpt-5": 32000, "gpt-5-mini": 32000}
ALL_METHODS = ["ontology-based", "list-based"]


def run_extraction(args):
    method_paths = get_method_paths(args.method)
    reference_path = Path(method_paths["reference_path"])
    prompt_path = Path(method_paths["prompt_path"])
    examples_dir = Path(method_paths["examples_dir"])
    output_dir = resolve_output_dir(
        args.method, args.model, args.web_search, args.txt_folder
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
    jobs = build_txt_jobs(args.txt_folder, FACILITIES_INFO_PATH)

    if not jobs:
        print(
            "No facilities were processed. "
            f"Check --facilities_information ({FACILITIES_INFO_PATH}) and --txt_folder ({args.txt_folder})."
        )
        raise SystemExit(0)

    reference_text = reference_path.read_text(encoding="utf-8")
    prompt_examples = load_icl_examples(
        num_examples=NUM_ICL_EXAMPLES,
        examples_dir=str(examples_dir),
    )
    system_prompt_template = prompt_path.read_text(encoding="utf-8")
    example_schema = None if args.skip_schema_validation else build_example_schema(args.method, web=args.web_search)

    facilities_source_df = pd.read_csv(FACILITIES_INFO_PATH, dtype=str).fillna('')

    # Write token usage and manifest one row at a time so interrupting mid-model
    # doesn't lose progress for facilities already completed in this run.
    token_usage_csv_path = output_dir / "token_usage_summary.csv"
    manifest_csv_path = output_dir / "facility_extraction_manifest.csv"
    token_usage_cols = [
        "facility_name", "txt_file", "extraction_file",
        "completion_token", "prompt_token", "total_token", "reasoning_token", "cost_usd",
    ]
    manifest_cols = list(facilities_source_df.columns) + ["txt_file", "extraction_file", "structured_output"]

    def append_row_csv(path, row, columns):
        pd.DataFrame([row], columns=columns).to_csv(
            path, mode="a", header=not path.exists(), index=False
        )

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
        if not permit_extract.split(SEP, 1)[0].strip():
            print(f"Empty description section, skipping.")
            continue
        print(f"Read text extract (length {len(permit_extract)})")

        if not args.no_token_limit and len(permit_extract) > 30000:
            print(
                f"Warning: Extracted text length ({len(permit_extract)}) exceeds typical token limits. "
                "Consider truncating or summarizing the text for better results."
            )
            continue

        txt_stem = txt_path.stem
        place_id = str(facilities_source_df.iloc[row_idx].get("Place ID", "")).strip() if 0 <= row_idx < len(facilities_source_df) else ""
        file_id = place_id or slugify(facility_name)
        extraction_file_name = f"{txt_stem}_{file_id}.json"
        output_json_path = output_dir / extraction_file_name

        existing = next(output_dir.glob(f"{txt_stem}_{file_id}.json"), None)
        if existing:
            print(f"Already processed: {existing.name}, skipping.")
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
                structured_output = None
                print("Parsed JSON result:")
                print(json.dumps(parsed, indent=2, ensure_ascii=False))
                print(f"Cost: ${cost_usd:.6f}")
            else:
                model_max = MAX_TOKENS_BY_MODEL.get(args.model)
                result = chat_completion_json(
                    model=args.model,
                    system_message=system_msg,
                    user_message=user_msg,
                    temperature=0.0,
                    max_tokens=None if args.no_token_limit else (model_max or DEFAULT_MAX_TOKENS),
                    max_completion_tokens=None if args.no_token_limit else (model_max or DEFAULT_MAX_COMPLETION_TOKENS),
                    schema=example_schema,
                )
                parsed, completion_token, prompt_token, total_token, reasoning_tokens, structured_output = result
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

            append_row_csv(
                token_usage_csv_path,
                {
                    "facility_name": facility_name,
                    "txt_file": txt_file,
                    "extraction_file": extraction_file_name,
                    "completion_token": completion_token,
                    "prompt_token": prompt_token,
                    "total_token": total_token,
                    "reasoning_token": reasoning_tokens,
                    "cost_usd": cost_usd,
                },
                token_usage_cols,
            )

            if 0 <= row_idx < len(facilities_source_df):
                facility_row = facilities_source_df.iloc[row_idx].to_dict()
                facility_row["txt_file"] = txt_file
                facility_row["extraction_file"] = extraction_file_name
                facility_row["structured_output"] = structured_output
                append_row_csv(manifest_csv_path, facility_row, manifest_cols)
        except Exception as exc:
            print("Error:", exc)

    if token_usage_csv_path.exists():
        print(f"Token usage CSV: {token_usage_csv_path}")
    if manifest_csv_path.exists():
        print(f"Facility manifest CSV: {manifest_csv_path}")


if __name__ == "__main__":
    args = parse_args()
    if args.all_models:
        for method in ALL_METHODS:
            for model in ALL_MODELS:
                print(f"\n{'='*80}")
                print(f"Running method={method} model={model}")
                print(f"{'='*80}\n")
                args.method = method
                args.model = model
                run_extraction(args)
    else:
        run_extraction(args)

# Terminal commands to run (uncommented)

# 5-FACILITY COMPARISON
# python wwtp_process_extraction/step5_llm_extraction.py \
#   --all_models \

# python wwtp_process_extraction/step5_llm_extraction.py \
#   --method ontology-based \
#   --model claude-sonnet-4-6 \
#   --web_search \

# python wwtp_process_extraction/step5_llm_extraction.py \
#   --method list-based \
#   --model claude-sonnet-4-6 \
#   --web_search \

# ALL FACILITIES, ONTOLOGY-BASED, GPT-5-MINI
# python wwtp_process_extraction/step5_llm_extraction.py