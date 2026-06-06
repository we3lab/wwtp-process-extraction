import json
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import requests
import jsonschema

BASE_URL = "https://aiapi-prod.stanford.edu/v1"


def get_headers():
    api_key = Path("wwtp_process_extraction/API_key.txt").read_text(encoding="utf-8").strip()
    if not api_key:
        raise ValueError("API_key.txt is empty.")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


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
                "enum": ["present", "future", "past"],
            },
            "Location": {
                "type": ["string", "null"],
                "enum": ["on-site", "off-site", None],
            },
            "Sentence": {"type": "string"},
        }
        required = ["Process", "Implementation", "Location", "Sentence"]
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
            "enum": ["present", "future", "past"],
        },
        "Location": {
            "type": ["string", "null"],
            "enum": ["on-site", "off-site", None],
        },
        "Sentence": {"type": "string"},
    }
    required = [
        "Equipment", "Process", "Role", "Substance",
        "Implementation", "Location", "Sentence",
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
            "reference_path": "wwtp_process_extraction/data/llm_extraction/input/unit_process_list.txt",
            "examples_dir": "wwtp_process_extraction/data/llm_extraction/icl_examples/list_based",
            "prompt_path": "wwtp_process_extraction/data/llm_extraction/prompt/list_based_prompt.txt",
            "output_dir": "wwtp_process_extraction/output/2026-2-18/llm_search_list",
            "reference_placeholder": "__UNIT_PROCESS_LIST__",
        }

    return {
        "reference_path": "wwtp_process_extraction/data/llm_extraction/input/ontology.txt",
        "examples_dir": "wwtp_process_extraction/data/llm_extraction/icl_examples/ontology_based",
        "prompt_path": "wwtp_process_extraction/data/llm_extraction/prompt/ontology_based_prompt.txt",
        "output_dir": "wwtp_process_extraction/output/2026-2-18/llm_search_ontology",
        "reference_placeholder": "__ONTOLOGY__",
    }


def _coerce_extraction_json(parsed):
    """Best-effort reshape of model output so it matches the extraction schema.

    Weaker models sometimes drop the {"items": [...]} wrapper or return single
    scalar fields as one-element lists. Fix those here before schema validation
    rather than failing the whole extraction.
    """
    item_keys = {"Equipment", "Process", "Role", "Substance",
                 "Implementation", "Location", "Sentence"}

    # Unwrap to a top-level {"items": [...]} object
    if isinstance(parsed, list):
        parsed = {"items": parsed}
    elif isinstance(parsed, dict) and not isinstance(parsed.get("items"), list):
        for key in ("output", "result", "results", "data", "extractions"):
            inner = parsed.get(key)
            if isinstance(inner, dict) and isinstance(inner.get("items"), list):
                parsed = inner
                break
            if isinstance(inner, list):
                parsed = {"items": inner}
                break
        else:
            # A single item object returned without the items wrapper
            if item_keys & set(parsed.keys()):
                parsed = {"items": [parsed]}

    # "items" given as a single dict instead of a list
    if isinstance(parsed, dict) and isinstance(parsed.get("items"), dict):
        parsed["items"] = [parsed["items"]]

    items = parsed.get("items") if isinstance(parsed, dict) else None
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            # Empty arrays → null for nullable array fields
            for field_name in ("Process", "Role", "Substance"):
                if item.get(field_name) == []:
                    item[field_name] = None
            # Single scalar fields returned as lists → unwrap to the scalar
            for field_name in ("Equipment", "Implementation", "Location"):
                val = item.get(field_name)
                if isinstance(val, list):
                    item[field_name] = val[0] if val else None
    return parsed


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

    content = None
    try:
        resp = requests.post(url, headers=get_headers(), json=payload, timeout=600)
        if not resp.ok:
            print(f"API error {resp.status_code}: {resp.text[:500]}")
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

        # Record whether the raw model output already matched the desired schema,
        # before any coercion. This feeds the "fraction structured output" metric.
        structured_output = None
        if schema is not None:
            try:
                jsonschema.validate(instance=parsed, schema=schema)
                structured_output = True
            except jsonschema.ValidationError:
                structured_output = False

        parsed = _coerce_extraction_json(parsed)

        # Proceed as long as we recovered an items list. Individual items may be
        # missing optional fields; those are left blank rather than dropped, and
        # structured_output already records that the raw output was non-conforming.
        if schema is not None and not isinstance(parsed.get("items"), list):
            raise ValueError("JSON output had no recoverable items list")

        return parsed, completion_token, prompt_token, total_token, reasoning_tokens, structured_output

    except (requests.RequestException, ValueError, json.JSONDecodeError) as e:
        err = RuntimeError(f"Failed to get valid JSON. Last error: {e}")
        if content:
            err.raw_output = content
        raise err
