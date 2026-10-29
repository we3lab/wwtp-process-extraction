import argparse
import json
import subprocess

import jsonschema
import pandas as pd
import requests

from helpers.ontology_to_txt import ontology_to_txt
from helpers.paths import *
from helpers.utils import (
    build_txt_jobs,
    leaves,
    SEP,
)

ALL_MODELS = ["claude-3-haiku", "claude-4-5-sonnet", "gpt-5", "gpt-5-mini", "gemini-2.5-pro"]
ALL_METHODS = ["ontology-based", "list-based"]
MODEL = "gpt-5-mini"
WEB_SEARCH_MODEL = "claude-sonnet-4-6"
NUM_ICL_EXAMPLES = 1
DEFAULT_MAX_TOKENS = 10000
# Reasoning models spend most of their budget on hidden reasoning; claude-3-haiku caps output at 4096
MAX_TOKENS_BY_MODEL = {"claude-3-haiku": 4096, "gpt-5": 32000, "gpt-5-mini": 32000}

LLM_DATA_DIR = DATA_DIR / "llm_extraction"
METHOD_PATHS = {
    "list-based": {
        "reference_path": LLM_DATA_DIR / "input" / "unit_process_list.txt",
        "examples_dir": LLM_DATA_DIR / "icl_examples" / "list_based",
        "prompt_path": LLM_DATA_DIR / "prompt" / "list_based_prompt.txt",
    },
    "ontology-based": {
        "reference_path": ONTOLOGY_TXT,
        "examples_dir": LLM_DATA_DIR / "icl_examples" / "ontology_based",
        "prompt_path": LLM_DATA_DIR / "prompt" / "ontology_based_prompt.txt",
    },
}
WEB_PROMPT_SUFFIX_PATH = LLM_DATA_DIR / "prompt" / "web_search_suffix.txt"

TOKEN_USAGE_COLUMNS = [
    "facility_name", "place_id", "extraction_file", "structured_output",
    "completion_token", "prompt_token", "total_token", "reasoning_token", "cost_usd",
]
STRING_LIST = {"type": "array", "items": {"type": "string"}}
ITEM_FIELDS = {"Equipment", "Process", "Role", "Substance", "Implementation", "Location", "Sentence"}


def output_dir_for(args):
    suffix = ("-web" if args.web_search else "") + ("-waterrag" if args.waterrag_context else "")
    return LLM_EXTRACTION_DIR / f"{args.method}_{args.model}{suffix}"


def raise_with_raw_output(message, raw_output):
    err = RuntimeError(message)
    err.raw_output = raw_output
    raise err


def build_example_schema(method, web=False):
    """JSON schema for the {"items": [...]} extraction output of one method."""
    if method == "list-based":
        props = {"Process": {**STRING_LIST, "minItems": 1}}
        required = ["Process"]
        method_constraints = {}
    else:
        props = {
            "Equipment": {"type": ["string", "null"]},
            "Process": {"anyOf": [{"type": "null"}, {**STRING_LIST, "minItems": 1}]},
            "Role": {"anyOf": [{"type": "null"}, STRING_LIST]},
            "Substance": {"anyOf": [{"type": "null"}, STRING_LIST]},
        }
        required = ["Equipment", "Process", "Role", "Substance"]
        # Equipment and Process can't both be null
        both_null = {"Equipment": {"type": "null"}, "Process": {"type": "null"}}
        method_constraints = {"not": {"properties": both_null, "required": ["Equipment", "Process"]}}

    props["Implementation"] = {"type": "string", "enum": ["present", "future", "past"]}
    props["Location"] = {"type": ["string", "null"], "enum": ["on-site", "off-site", None]}
    props["Sentence"] = {"type": "string"}
    required += ["Implementation", "Location", "Sentence"]
    if web:
        props["Source"] = {"type": "string", "enum": ["permit_text", "web_search", "both"]}
        props["Website"] = {
            "type": ["string", "null"],
            "description": "Website or URL source if Source includes web_search",
        }
        required.append("Source")
    item_schema = {
        "type": "object",
        "properties": props,
        "required": required,
        **method_constraints,
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"items": {"type": "array", "items": item_schema}},
        "required": ["items"],
        "additionalProperties": False,
    }


def chat_completion_json(model, system_message, user_message, max_tokens, schema):
    """Request a JSON object from the Stanford proxy. Returns (parsed_json, completion_tokens,
    prompt_tokens, total_tokens, reasoning_tokens, structured_output)."""
    api_key = API_KEY_PATH.read_text(encoding="utf-8").strip()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
        "max_completion_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(f"{API_URL}/chat/completions", headers=headers, json=payload, timeout=600)
    if not resp.ok:
        print(f"API error {resp.status_code}: {resp.text[:500]}")
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    content = choice["message"]["content"]
    usage = data["usage"]
    completion_token = usage["completion_tokens"]
    prompt_token = usage["prompt_tokens"]
    total_token = usage["total_tokens"]
    # completion_tokens_details may be absent (every claude row has 0 reasoning tokens)
    reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)

    try:
        if choice.get("finish_reason") == "length":
            raise ValueError(
                f"Generation stopped (finish_reason='length', completion_tokens={completion_token}, "
                f"max_tokens={max_tokens}, max_completion_tokens={max_tokens}). "
            )
        if content is None or not content.strip():
            raise ValueError("Model returned null/empty content.")
        parsed = json.loads(content)
        # whether the raw output matched the schema before coercion ("fraction structured output")
        structured_output = jsonschema.Draft202012Validator(schema).is_valid(parsed)

        # Weaker models drop the {"items": [...]} wrapper or return scalar fields as lists; fix those
        if isinstance(parsed, list):
            parsed = {"items": parsed}
        if not isinstance(parsed, dict):
            raise ValueError("JSON output is not an object")
        if not isinstance(parsed.get("items"), list):
            for key in ("output", "result", "results", "data", "extractions"):
                inner = parsed.get(key)
                if isinstance(inner, dict) and isinstance(inner.get("items"), list):
                    parsed = inner
                    break
                if isinstance(inner, list):
                    parsed = {"items": inner}
                    break
            else:
                # a single item returned without the items wrapper
                if ITEM_FIELDS & set(parsed):
                    parsed = {"items": [parsed]}
        if isinstance(parsed.get("items"), dict):
            parsed["items"] = [parsed["items"]]
        # Items missing fields are kept; structured_output already records the raw mismatch
        if not isinstance(parsed.get("items"), list):
            raise ValueError("JSON output had no recoverable items list")
        for item in parsed["items"]:
            for field_name in ("Process", "Role", "Substance"):
                if item.get(field_name) == []:
                    item[field_name] = None
            for field_name in ("Equipment", "Implementation", "Location"):
                val = item.get(field_name)
                if isinstance(val, list):
                    item[field_name] = val[0] if val else None
    except ValueError as exc:
        raise_with_raw_output(f"Failed to get valid JSON. Last error: {exc}", content)
    return parsed, completion_token, prompt_token, total_token, reasoning_tokens, structured_output


def chat_completion_web(model, system_message, user_message, schema):
    """Call the claude CLI with WebSearch/WebFetch. Returns (parsed_json, cost_usd)."""
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
    output = json.loads(result.stdout)
    if output["is_error"]:
        raise_with_raw_output(f"API error: {output.get('api_error_status')}", result.stdout)
    cost_usd = float(output["total_cost_usd"])
    # --json-schema puts the validated result in structured_output; result is empty in that case
    if isinstance(output.get("structured_output"), dict):
        return output["structured_output"], cost_usd
    raw = output["result"].strip()
    if not raw:
        diag_keys = ("subtype", "num_turns", "duration_ms", "total_cost_usd", "stop_reason")
        diag = {k: output.get(k) for k in diag_keys}
        message = f"Model returned empty result (no final JSON emitted). CLI result meta: {diag}"
        raise_with_raw_output(message, result.stdout)
    if raw.startswith("{"):
        return json.loads(raw), cost_usd
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end < start:
        raise_with_raw_output(f"No JSON found in result: {raw[:100]}", result.stdout)
    return json.loads(raw[start:end + 1]), cost_usd


def append_row_csv(path, row):
    df = pd.DataFrame([row], columns=TOKEN_USAGE_COLUMNS)
    if path.exists():
        df = pd.concat([pd.read_csv(path).reindex(columns=TOKEN_USAGE_COLUMNS), df], ignore_index=True)
    # keep one row per extraction: prefer the latest non-FAILED row, else the latest row
    key = df["extraction_file"].fillna(df["facility_name"])
    failed = df["structured_output"].astype(str) == "FAILED"
    ordered_keys = pd.concat([key[failed], key[~failed]])
    keep_index = ordered_keys[~ordered_keys.duplicated(keep="last")].index
    df.loc[sorted(keep_index)].to_csv(path, index=False)


def run_extraction(args, output_dir):
    method_paths = METHOD_PATHS[args.method]
    facilities_info = SITE_DATA_RELEVANT_CSV if args.all_facilities else MANUAL_CSV
    jobs = build_txt_jobs(facilities_info)
    if not jobs:
        raise SystemExit(f"No facilities found. Check {facilities_info} and {TXT_DIR}.")
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_text = method_paths["reference_path"].read_text(encoding="utf-8")
    prompt_examples = "\n\n".join(
        f"Example {idx}:\n"
        + (method_paths["examples_dir"] / f"example{idx}.txt").read_text(encoding="utf-8").strip()
        for idx in range(1, NUM_ICL_EXAMPLES + 1)
    )
    system_prompt_template = method_paths["prompt_path"].read_text(encoding="utf-8")
    example_schema = build_example_schema(args.method, web=args.web_search)

    # One row per extraction, appended as we go so an interrupted run keeps its progress
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
        usage_row = {
            "facility_name": facility_name, "place_id": place_id, "extraction_file": extraction_file_name,
        }

        system_msg = (
            system_prompt_template
            .replace("__FACILITY_NAME__", facility_name)
            .replace("__ONTOLOGY__", reference_text)
            .replace("__UNIT_PROCESS_LIST__", reference_text)
            .replace("__PROMPT_EXAMPLES__", prompt_examples)
        )
        if args.web_search:
            system_msg += WEB_PROMPT_SUFFIX_PATH.read_text(encoding="utf-8")

        user_msg = (
            f"Find all the treatment processes explicitly used in the {facility_name} facility. "
            f"Here is the permit extract:\n{permit_extract}\n"
        )
        if args.waterrag_context:
            # Appended to the user message so the system prompt stays identical to the no-retrieval control
            context_path = WATERRAG_RETRIEVAL_DIR / extraction_file_name
            chunks = json.loads(context_path.read_text(encoding="utf-8"))["chunks"]
            rendered = "\n\n".join(
                f"[{i}] {chunk['citation'] or 'No citation'}\n{chunk['text']}"
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
                + "\n\nIMPORTANT: Your entire response must be ONLY a valid JSON object matching the schema. "
                "No other text, no explanation, no markdown."
            )

        try:
            if args.web_search:
                parsed, cost_usd = chat_completion_web(args.model, system_msg, user_msg, example_schema)
                completion_token = prompt_token = total_token = reasoning_tokens = 0
                structured_output = True
                print(f"Cost: ${cost_usd:.6f}")
            else:
                max_tokens = MAX_TOKENS_BY_MODEL.get(args.model, DEFAULT_MAX_TOKENS)
                parsed, completion_token, prompt_token, total_token, reasoning_tokens, structured_output = (
                    chat_completion_json(args.model, system_msg, user_msg, max_tokens, example_schema)
                )
                cost_usd = None
                print(
                    f"Token usage: completion={completion_token}, prompt={prompt_token}, "
                    f"total={total_token}, reasoning={reasoning_tokens}"
                )

            if structured_output is False:
                print("Parsed JSON result (non-structured):")
                print(json.dumps(parsed, indent=2, ensure_ascii=False))
            output_json_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
            append_row_csv(token_usage_csv_path, {
                **usage_row, "structured_output": structured_output, "completion_token": completion_token,
                "prompt_token": prompt_token, "total_token": total_token, "reasoning_token": reasoning_tokens,
                "cost_usd": cost_usd,
            })
        except Exception as exc:
            print("Error:", exc)
            raw = getattr(exc, "raw_output", None)
            if raw:
                failed_path = output_dir / f"{txt_stem}_{place_id}_FAILED.json"
                failed_path.write_text(raw, encoding="utf-8")
                print(f"Saved raw output to {failed_path.name}")
            append_row_csv(token_usage_csv_path, {
                **usage_row, "structured_output": "FAILED", "completion_token": 0, "prompt_token": 0,
                "total_token": 0, "reasoning_token": 0, "cost_usd": None,
            })

    print(f"Token usage CSV: {token_usage_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM unit-process extraction on permit text.")
    parser.add_argument("--all_methods", action="store_true",
                        help="Run both methods (default: ontology-based).")
    parser.add_argument("--all_models", action="store_true", help=f"Loop over ALL_MODELS (default: {MODEL}).")
    parser.add_argument("--web_search", action="store_true",
                        help=f"Use the claude CLI with WebSearch/WebFetch and {WEB_SEARCH_MODEL}.")
    parser.add_argument("--waterrag_context", action="store_true",
                        help="Add the WaterRAG literature context cached by step5b to the user message.")
    parser.add_argument("--all_facilities", action="store_true",
                        help="Run the full CA set instead of the manually-read facilities.")
    parser.add_argument("--repeat_runs", type=int,
                        help="Rerun the benchmark N times with the default config, for F1 variance.")
    args = parser.parse_args()

    if (args.repeat_runs or not args.web_search) and not API_KEY_PATH.read_text(encoding="utf-8").strip():
        raise SystemExit(f"{API_KEY_PATH} is empty.")
    ontology_to_txt()  # once per invocation, from the pinned Zenodo release
    if args.repeat_runs:
        # Default config in its own folder per run, so no facility is skipped as already processed
        args.method, args.model = "ontology-based", MODEL
        args.all_facilities = args.web_search = False
        base = output_dir_for(args) / "additional_runs"
        for k in range(1, args.repeat_runs + 1):
            print(f"\n{'='*80}\nRepeat run {k}/{args.repeat_runs}\n{'='*80}\n")
            run_extraction(args, base / f"run_{k}")
        raise SystemExit(0)

    methods = ALL_METHODS if args.all_methods else ["ontology-based"]
    models = ALL_MODELS if args.all_models else [WEB_SEARCH_MODEL if args.web_search else MODEL]

    if "list-based" in methods:
        # list-based prompt reference: one "name: alt names" line per leaf process
        lines = []
        for name, details, _ in leaves:
            alt_names = [a.strip() for a in details["alt_names"] + details["alt_names_case_sensitive"]]
            lines.append(f"{name}: {', '.join(dict.fromkeys(alt_names))}")
        METHOD_PATHS["list-based"]["reference_path"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    for method in methods:
        for model in models:
            print(f"\n{'='*80}\nRunning method={method} model={model}\n{'='*80}\n")
            args.method, args.model = method, model
            run_extraction(args, output_dir_for(args))
