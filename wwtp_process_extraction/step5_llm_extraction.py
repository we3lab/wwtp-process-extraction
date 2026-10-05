import argparse
import json
import re
import subprocess

import pandas as pd

from helpers.ontology_to_txt import ontology_to_txt
from helpers.utils import (
    build_txt_jobs,
    SEP,
    DATA_DIR,
    TXT_DIR,
    MANUAL_CSV,
    SITE_DATA_RELEVANT_CSV,
    KEYWORDS_JSON,
    WATERRAG_RETRIEVAL_DIR,
    LLM_EXTRACTION_DIR,
)
from helpers.api_llm_search import (
    chat_completion_json,
    load_icl_examples,
    init_unit_process_list_from_json,
    build_example_schema,
    require_api_key,
)

ALL_MODELS = ["claude-3-haiku", "claude-4-5-sonnet", "gpt-5", "gpt-5-mini", "gemini-2.5-pro"]
ALL_METHODS = ["ontology-based", "list-based"]
MODEL = "gpt-5-mini"
NUM_ICL_EXAMPLES = 1
DEFAULT_MAX_TOKENS = 10000
# Per-model completion-token limit. Reasoning models (gpt-5, etc.) spend most of
# their completion budget on hidden reasoning before emitting JSON, so they need a
# higher ceiling than chat models; claude-3-haiku has a hard 4096 output cap.
MAX_TOKENS_BY_MODEL = {"claude-3-haiku": 4096, "gpt-5": 32000, "gpt-5-mini": 32000}

LLM_DATA_DIR = DATA_DIR / "llm_extraction"
METHOD_PATHS = {
    "list-based": {
        "reference_path": LLM_DATA_DIR / "input" / "unit_process_list.txt",
        "examples_dir": LLM_DATA_DIR / "icl_examples" / "list_based",
        "prompt_path": LLM_DATA_DIR / "prompt" / "list_based_prompt.txt",
    },
    "ontology-based": {
        "reference_path": LLM_DATA_DIR / "input" / "ontology.txt",
        "examples_dir": LLM_DATA_DIR / "icl_examples" / "ontology_based",
        "prompt_path": LLM_DATA_DIR / "prompt" / "ontology_based_prompt.txt",
    },
}
WEB_PROMPT_SUFFIX_PATH = LLM_DATA_DIR / "prompt" / "web_search_suffix.txt"


TOKEN_USAGE_COLUMNS = [
    "facility_name", "place_id", "extraction_file", "structured_output",
    "completion_token", "prompt_token", "total_token", "reasoning_token", "cost_usd",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run wastewater treatment extraction with configurable LLM prompt methods."
    )
    parser.add_argument(
        "--method",
        choices=ALL_METHODS,
        default="ontology-based",
        help="Prompting/extraction method to use (default: ontology-based).",
    )
    parser.add_argument(
        "--model",
        default=MODEL,
        help=f"Model name for API calls in {', '.join(ALL_MODELS)} (default: {MODEL}).",
    )
    parser.add_argument(
        "--all_methods",
        action="store_true",
        help="Loop over both methods instead of --method.",
    )
    parser.add_argument(
        "--all_models",
        action="store_true",
        help="Loop over ALL_MODELS instead of --model.",
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
        "--waterrag_context",
        action="store_true",
        help=(
            "Append wastewater-literature context retrieved by the WaterRAG index to the user "
            "message. Requires step5b_waterrag_retrieval.py to have cached the context first."
        ),
    )
    parser.add_argument(
        "--all_facilities",
        action="store_true",
        help=(
            "Run the full CA set (site_data_relevant.csv) instead of the manually-read facilities. "
            "Writes to the same output/llm_extraction/<method>_<model> folder as any other run."
        ),
    )
    parser.add_argument(
        "--repeat_runs",
        type=int,
        default=None,
        help=(
            "Run the benchmark facilities N extra times with the default model/method "
            "(for F1 run-to-run variance), each into output/llm_extraction/ontology-based_gpt-5-mini"
            "[-waterrag]/additional_runs/run_<k>/. Ignores --model/--method/--all_models/--all_methods/--web_search/"
            "--all_facilities."
        ),
    )
    return parser.parse_args()


def resolve_output_dir(method, model, web_search, waterrag_context):
    # Every run (full CA or model comparison) writes to output/llm_extraction/<method>_<model>.
    # The default ontology-based_gpt-5-mini folder accumulates the full CA set; the benchmark
    # facilities are a subset of it, so model-comparison runs for that config are reused.
    suffix = f"{method}_{model}" + ("-web" if web_search else "") + ("-waterrag" if waterrag_context else "")
    return LLM_EXTRACTION_DIR / suffix


def raise_with_raw_output(message, raw_output):
    err = RuntimeError(message)
    err.raw_output = raw_output
    raise err


def chat_completion_web(
    model: str,
    system_message: str,
    user_message: str,
    schema: dict,
) -> tuple:
    """Call claude CLI with WebSearch/WebFetch. Returns (parsed_json, cost_usd).
    Parses the JSON result from the CLI output, handling multiple objects and markdown wrapping."""
    cmd = [
        "claude", "-p", user_message,
        "--system-prompt", system_message,
        "--allowedTools", "WebSearch,WebFetch",
        "--output-format", "json",
        "--model", model,
        "--effort", "medium",
        "--no-session-persistence",
        "--json-schema", json.dumps(schema),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        err = RuntimeError(f"CLI failed: {result.stderr[:200]}")
        if result.stdout.strip():
            err.raw_output = result.stdout
        raise err
    stdout = result.stdout
    if not stdout.strip():
        raise RuntimeError("empty output from CLI")

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
        raise_with_raw_output("No result found in CLI output", stdout)
    if output.get("is_error"):
        raise_with_raw_output(f"API error: {output.get('api_error_status')}", stdout)
    # --json-schema puts the validated result in structured_output; result is empty in that case
    if isinstance(output.get("structured_output"), dict):
        return output["structured_output"], float(output.get("total_cost_usd") or 0.0)
    raw = output["result"]
    if raw.strip().startswith("{"):
        parsed = json.loads(raw.strip())
    else:
        if not raw.strip():
            diag = {k: output.get(k) for k in ("subtype", "num_turns", "duration_ms", "total_cost_usd", "stop_reason")}
            raise_with_raw_output(f"Model returned empty result (no final JSON emitted). CLI result meta: {diag}", stdout)
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            raise_with_raw_output(f"No JSON found in result: {raw[:100]}", stdout)
        parsed = json.loads(match.group())
    return parsed, float(output.get("total_cost_usd") or 0.0)


def append_row_csv(path, row):
    df = pd.DataFrame([row], columns=TOKEN_USAGE_COLUMNS)
    if path.exists():
        df = pd.concat([pd.read_csv(path).reindex(columns=TOKEN_USAGE_COLUMNS), df], ignore_index=True)
    # keep one row per extraction: prefer the latest non-FAILED row, else the latest row
    key = df["extraction_file"].fillna(df["facility_name"])
    failed = df["structured_output"].astype(str).str.upper() == "FAILED"
    ordered_keys = pd.concat([key[failed], key[~failed]])
    keep_index = ordered_keys[~ordered_keys.duplicated(keep="last")].index
    df.loc[sorted(keep_index)].to_csv(path, index=False)


def run_extraction(args, output_dir_override=None):
    method_paths = METHOD_PATHS[args.method]
    reference_path = method_paths["reference_path"]
    # Default: the manually-read facilities (model comparison). --all_facilities switches to the full CA set.
    facilities_info = SITE_DATA_RELEVANT_CSV if args.all_facilities else MANUAL_CSV
    output_dir = output_dir_override or resolve_output_dir(
        args.method, args.model, args.web_search, args.waterrag_context
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_txt_jobs(facilities_info)

    if not jobs:
        print(
            "No facilities were processed. "
            f"Check the facilities file ({facilities_info}) and {TXT_DIR}."
        )
        return

    reference_text = reference_path.read_text(encoding="utf-8")
    prompt_examples = load_icl_examples(
        num_examples=NUM_ICL_EXAMPLES,
        examples_dir=method_paths["examples_dir"],
    )
    system_prompt_template = method_paths["prompt_path"].read_text(encoding="utf-8")
    example_schema = build_example_schema(args.method, web=args.web_search)

    # Single per-dir summary, written one row at a time so interrupting mid-model doesn't
    # lose progress. This is the only file step6/table_1 need: step6 derives Place ID from
    # each JSON filename ({txt_stem}_{place_id}.json), and the unit-process data lives in
    # the JSON results, so the txt_file/extraction_file names aren't stored here.
    token_usage_csv_path = output_dir / "token_usage_summary.csv"

    for _, txt_path, facility_name, place_id in jobs:
        print("#" * 80)
        print(f"\nProcessing {txt_path.name} for facility {facility_name}...")

        permit_extract = txt_path.read_text(encoding="utf-8")
        if not permit_extract.split(SEP, 1)[0].strip():
            print("Empty description section, skipping.")
            continue
        print(f"Read text extract (length {len(permit_extract)})")

        txt_stem = txt_path.stem
        extraction_file_name = f"{txt_stem}_{place_id}.json"
        output_json_path = output_dir / extraction_file_name

        if output_json_path.exists():
            print(f"Already processed: {extraction_file_name}, skipping.")
            continue

        system_msg = (
            system_prompt_template
            .replace("__FACILITY_NAME__", facility_name)
            .replace("__ONTOLOGY__", reference_text)
            .replace("__UNIT_PROCESS_LIST__", reference_text)
            .replace("__PROMPT_EXAMPLES__", prompt_examples)
        )
        if args.web_search:
            system_msg = system_msg + WEB_PROMPT_SUFFIX_PATH.read_text(encoding="utf-8")

        user_msg = (
            f"Find all the treatment processes explicitly used in the {facility_name} facility. "
            f"Here is the permit extract:\n{permit_extract}\n"
        )
        if args.waterrag_context:
            # Appended to the user message, never the system message, so the prompt template,
            # ontology dump and ICL example stay byte-identical to the no-retrieval control.
            context_path = WATERRAG_RETRIEVAL_DIR / extraction_file_name
            if not context_path.exists():
                print(f"No cached WaterRAG context ({context_path.name}), skipping. Run step5b first.")
                continue
            chunks = json.loads(context_path.read_text(encoding="utf-8")).get("chunks", [])
            if not chunks:
                print(f"Cached WaterRAG context is empty ({context_path.name}), skipping.")
                continue
            rendered = "\n\n".join(
                f"[{i}] {chunk.get('citation') or 'No citation'}\n{chunk['text']}"
                for i, chunk in enumerate(chunks, 1)
            )
            user_msg += (
                "\n\nReference literature on wastewater unit processes, retrieved for background "
                "only. Use it to interpret terminology; extract ONLY what the permit extract above "
                f"states about the {facility_name} facility, and keep every Sentence field quoted "
                f"from the permit, never from this literature.\n\n{rendered}\n"
            )

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
                structured_output = True
                print(f"Cost: ${cost_usd:.6f}")
            else:
                parsed, completion_token, prompt_token, total_token, reasoning_tokens, structured_output = chat_completion_json(
                    model=args.model,
                    system_message=system_msg,
                    user_message=user_msg,
                    max_tokens=MAX_TOKENS_BY_MODEL.get(args.model, DEFAULT_MAX_TOKENS),
                    schema=example_schema,
                )
                cost_usd = None
                print(
                    f"Token usage: completion={completion_token}, prompt={prompt_token}, "
                    f"total={total_token}, reasoning={reasoning_tokens}"
                )

            # Only dump the parsed JSON for non-structured (schema-nonconforming) outputs.
            if structured_output is False:
                print("Parsed JSON result (non-structured):")
                print(json.dumps(parsed, indent=2, ensure_ascii=False))

            with open(output_json_path, "w", encoding="utf-8") as output_file:
                json.dump(parsed, output_file, ensure_ascii=False, indent=2)

            append_row_csv(
                token_usage_csv_path,
                {
                    "facility_name": facility_name,
                    "place_id": place_id,
                    "extraction_file": extraction_file_name,
                    "structured_output": structured_output,
                    "completion_token": completion_token,
                    "prompt_token": prompt_token,
                    "total_token": total_token,
                    "reasoning_token": reasoning_tokens,
                    "cost_usd": cost_usd,
                },
            )
        except Exception as exc:
            print("Error:", exc)
            raw = getattr(exc, "raw_output", None)
            if raw:
                failed_path = output_dir / f"{txt_stem}_{place_id}_FAILED.json"
                failed_path.write_text(raw, encoding="utf-8")
                print(f"Saved raw output to {failed_path.name}")
            append_row_csv(
                token_usage_csv_path,
                {
                    "facility_name": facility_name,
                    "place_id": place_id,
                    "extraction_file": extraction_file_name,
                    "structured_output": "FAILED",
                    "completion_token": 0,
                    "prompt_token": 0,
                    "total_token": 0,
                    "reasoning_token": 0,
                    "cost_usd": None,
                },
            )

    if token_usage_csv_path.exists():
        print(f"Token usage CSV: {token_usage_csv_path}")


if __name__ == "__main__":
    args = parse_args()
    if args.repeat_runs or not args.web_search:
        require_api_key()
    ontology_to_txt()  # once per invocation, from the pinned Zenodo release
    if args.repeat_runs:
        # Repeated runs of the benchmark facilities with the default model/method (same config as
        # the full-CA run) to measure F1 run-to-run variance. Each run gets its own folder so all
        # facilities run every time (no cross-run skipping).
        args.method, args.model = "ontology-based", MODEL
        args.all_facilities = args.web_search = False
        base = resolve_output_dir(args.method, args.model, False, args.waterrag_context) / "additional_runs"
        for k in range(1, args.repeat_runs + 1):
            print(f"\n{'='*80}\nRepeat run {k}/{args.repeat_runs}\n{'='*80}\n")
            run_extraction(args, output_dir_override=base / f"run_{k}")
        raise SystemExit(0)

    methods = ALL_METHODS if args.all_methods else [args.method]
    models = ALL_MODELS if args.all_models else [args.model]

    if "list-based" in methods:
        init_unit_process_list_from_json(
            keywords_json_path=KEYWORDS_JSON,
            output_txt_path=METHOD_PATHS["list-based"]["reference_path"],
        )
    for method in methods:
        for model in models:
            print(f"\n{'='*80}\nRunning method={method} model={model}\n{'='*80}\n")
            args.method, args.model = method, model
            run_extraction(args)
