import json
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import pandas as pd
import requests

BASE_URL = "https://aiapi-prod.stanford.edu/v1"


def get_headers():
    api_key = Path("npdes_permits/API_key.txt").read_text(encoding="utf-8").strip()
    if not api_key:
        raise ValueError("API_key.txt is empty.")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def build_txt_jobs(txt_folder: str, facilities_information: str):
    txt_folder_path = Path(txt_folder)
    if not txt_folder_path.exists() or not txt_folder_path.is_dir():
        raise ValueError(f"--txt_folder must be an existing directory: {txt_folder_path}")

    facilities_path = Path(facilities_information)
    if not facilities_path.exists() or not facilities_path.is_file():
        raise ValueError(
            f"--facilities_information must be an existing CSV file: {facilities_path}"
        )

    facilities_df = pd.read_csv(facilities_path, dtype=str).fillna("")
    required_columns = {"Facility Name", "PDF_File"}
    missing_columns = required_columns.difference(set(facilities_df.columns))
    if missing_columns:
        raise ValueError(
            "--facilities_information is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    jobs = []
    for row_idx, row in facilities_df.iterrows():
        facility_name = str(row["Facility Name"]).strip()
        pdf_file_value = str(row["PDF_File"]).strip()

        if not facility_name or facility_name.lower() == "nan":
            continue
        if not pdf_file_value or pdf_file_value.lower() == "nan":
            continue

        txt_path = Path(pdf_file_value)
        if not txt_path.is_absolute():
            txt_path = txt_folder_path / txt_file_value_to_txt_name(pdf_file_value)

        if txt_path.suffix.lower() != ".txt":
            raise ValueError(
                f"PDF_File value is not mapped to a .txt for facility '{facility_name}': {pdf_file_value}"
            )
        if not txt_path.exists() or not txt_path.is_file():
            raise FileNotFoundError(
                f"TXT not found for facility '{facility_name}': {txt_path}"
            )

        jobs.append((row_idx, txt_path, txt_path.name, facility_name))

    return jobs


def txt_file_value_to_txt_name(file_value: str) -> str:
    path_value = Path(file_value)
    return path_value.with_suffix(".txt").name


def build_pdf_jobs(pdf_folder: str, facilities_information: str):
    return build_txt_jobs(pdf_folder, facilities_information)


def get_models():
    """GET available models"""
    url = f"{BASE_URL}/models"
    resp = requests.get(url, headers=get_headers())
    resp.raise_for_status()
    return resp.json()


def load_icl_examples(num_examples: int, examples_dir: str) -> str:
    if num_examples < 0:
        raise ValueError("--num_examples must be >= 0.")
    if num_examples == 0:
        return ""

    examples_root = Path(examples_dir)
    examples = []
    for idx in range(1, num_examples + 1):
        example_path = examples_root / f"example{idx}.txt"
        if not example_path.exists() or not example_path.is_file():
            raise FileNotFoundError(
                f"ICL example file not found: {example_path}. "
                "Create the file or reduce --num_examples."
            )
        example_content = example_path.read_text(encoding="utf-8").strip()
        examples.append(f"Example {idx}:\n{example_content}")

    return "\n\n".join(examples)


def _collect_process_entries(
    data: Dict[str, Any],
    entries: Optional[List[Tuple[str, List[str]]]] = None,
) -> List[Tuple[str, List[str]]]:
    if entries is None:
        entries = []

    for process_name, value in data.items():
        if not isinstance(value, dict):
            continue

        if "alt_names" in value or "alt_names_case_sensitive" in value:
            alt_names = value.get("alt_names", [])
            alt_names_case_sensitive = value.get("alt_names_case_sensitive", [])
            all_alt_names = []
            if isinstance(alt_names, list):
                all_alt_names.extend([str(item).strip() for item in alt_names if str(item).strip()])
            if isinstance(alt_names_case_sensitive, list):
                all_alt_names.extend(
                    [str(item).strip() for item in alt_names_case_sensitive if str(item).strip()]
                )

            dedup_alt_names = list(dict.fromkeys(all_alt_names))
            entries.append((str(process_name), dedup_alt_names))

        _collect_process_entries(value, entries)

    return entries


def init_unit_process_list_from_json(keywords_json_path: str, output_txt_path: str) -> Path:
    source_path = Path(keywords_json_path)
    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError(f"unitprocess keywords JSON not found: {source_path}")

    data = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("unitprocess keywords JSON must contain a top-level object.")

    entries = _collect_process_entries(data)
    if not entries:
        raise ValueError("No process entries with alt names found in unitprocess keywords JSON.")

    lines = []
    for process_name, alt_names in entries:
        if alt_names:
            lines.append(f"{process_name}: {', '.join(alt_names)}")
        else:
            lines.append(f"{process_name}:")

    output_path = Path(output_txt_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


_SOURCE_FIELD = {
    "type": "string",
    "enum": ["permit_text", "web_search", "both"],
}

_WEBSITE_FIELD = {
    "type": ["string", "null"],
    "description": "Website or URL source if Source includes web_search",
}


def build_example_schema(method: str, web: bool = False) -> Dict[str, Any]:
    if method == "list-based":
        props = {
            "Process": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            },
            "Implementation": {
                "type": "string",
                "enum": ["present", "planned", "past"],
            },
            "Location": {
                "type": ["string", "null"],
                "enum": ["on-site", "off-site", None],
            },
            "Score": {"type": "number", "minimum": 0, "maximum": 1},
            "Sentence": {"type": "string"},
        }
        required = ["Process", "Implementation", "Location", "Score", "Sentence"]
        if web:
            props["Source"] = _SOURCE_FIELD
            props["Website"] = _WEBSITE_FIELD
            required.append("Source")
        return {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": props,
                        "required": required,
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["items"],
            "additionalProperties": False,
        }

    props = {
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
            "enum": ["on-site", "off-site", None],
        },
        "Score": {"type": "number", "minimum": 0, "maximum": 1},
        "Sentence": {"type": "string"},
    }
    required = [
        "Equipment", "Process", "Role", "Substance",
        "Implementation", "Location", "Score", "Sentence",
    ]
    if web:
        props["Source"] = _SOURCE_FIELD
        props["Website"] = _WEBSITE_FIELD
        required.append("Source")
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                    "not": {
                        "properties": {
                            "Equipment": {"type": "null"},
                            "Process": {"type": "null"},
                        },
                        "required": ["Equipment", "Process"],
                    },
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }


def get_method_paths(method: str) -> Dict[str, str]:
    if method == "list-based":
        return {
            "reference_path": "npdes_permits/data/llm_extraction/input/unit_process_list.txt",
            "examples_dir": "npdes_permits/data/llm_extraction/icl_examples/list_based",
            "prompt_path": "npdes_permits/data/llm_extraction/prompt/list_based_prompt.txt",
            "output_dir": "npdes_permits/output/2026-2-18/llm_search_list",
            "reference_placeholder": "__UNIT_PROCESS_LIST__",
        }

    return {
        "reference_path": "npdes_permits/data/llm_extraction/input/ontology.txt",
        "examples_dir": "npdes_permits/data/llm_extraction/icl_examples/ontology_based",
        "prompt_path": "npdes_permits/data/llm_extraction/prompt/ontology_based_prompt.txt",
        "output_dir": "npdes_permits/output/2026-2-18/llm_search_ontology",
        "reference_placeholder": "__ONTOLOGY__",
    }


def chat_completion_json(
    model: str,
    user_message: str,
    system_message: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = 3000,
    max_completion_tokens: Optional[int] = 8000,
    stop: Optional[Any] = None,
    retry_delay: float = 1.0,
    schema: Optional[Dict] = None,
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

    effective_max_completion_tokens = None
    if max_tokens is not None and max_completion_tokens is not None:
        effective_max_completion_tokens = min(max_tokens, max_completion_tokens)
    elif max_completion_tokens is not None:
        effective_max_completion_tokens = max_completion_tokens
    elif max_tokens is not None:
        effective_max_completion_tokens = max_tokens

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if effective_max_completion_tokens is not None:
        payload["max_completion_tokens"] = effective_max_completion_tokens
    if stop is not None:
        payload["stop"] = stop

    try:
        resp = requests.post(url, headers=get_headers(), json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()

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

        parsed = json.loads(content)

        items = parsed.get("items") if isinstance(parsed, dict) else None
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                for field_name in ("Process", "Role", "Substance"):
                    if item.get(field_name) == []:
                        item[field_name] = None

        if schema is not None:
            try:
                import jsonschema

                jsonschema.validate(instance=parsed, schema=schema)
            except ImportError:
                print("Warning: jsonschema not installed; skipping schema validation.")
            except jsonschema.ValidationError as ve:
                raise ValueError(f"JSON did not validate against schema: {ve}")

        return parsed, completion_token, prompt_token, total_token, reasoning_tokens

    except (requests.RequestException, ValueError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to get valid JSON. Last error: {e}")
