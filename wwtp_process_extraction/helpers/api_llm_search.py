import json
from pathlib import Path

import requests
import jsonschema

BASE_URL = "https://aiapi-prod.stanford.edu/v1"
API_KEY_PATH = Path("wwtp_process_extraction/API_key.txt")


def require_api_key():
    """Fail once up front rather than once per facility. Without this a missing key
    produces an 'Error:' line and a FAILED usage row for every facility in the run."""
    if not API_KEY_PATH.exists():
        raise SystemExit(
            f"{API_KEY_PATH} not found. Create it with your Stanford AI API key "
            "(run from the repo root)."
        )
    if not API_KEY_PATH.read_text(encoding="utf-8").strip():
        raise SystemExit(f"{API_KEY_PATH} is empty.")


def get_headers():
    api_key = API_KEY_PATH.read_text(encoding="utf-8").strip()
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def build_example_schema(method, web=False):
    if method == "list-based":
        props = {
            "Process": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            },
        }
        required = ["Process"]
        method_constraints = {}
    else:
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
        }
        required = ["Equipment", "Process", "Role", "Substance"]
        method_constraints = {
            "not": {
                "properties": {
                    "Equipment": {"type": "null"},
                    "Process": {"type": "null"},
                },
                "required": ["Equipment", "Process"],
            },
        }

    props["Implementation"] = {
        "type": "string",
        "enum": ["present", "future", "past"],
    }
    props["Location"] = {
        "type": ["string", "null"],
        "enum": ["on-site", "off-site", None],
    }
    props["Sentence"] = {"type": "string"}
    required += ["Implementation", "Location", "Sentence"]
    if web:
        props["Source"] = {
            "type": "string",
            "enum": ["permit_text", "web_search", "both"],
        }
        props["Website"] = {
            "type": ["string", "null"],
            "description": "Website or URL source if Source includes web_search",
        }
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
                    **method_constraints,
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }


def coerce_extraction_json(parsed):
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


def chat_completion_json(model, system_message, user_message, max_tokens, schema):
    """
    Request a JSON object from the model. Returns (parsed_json, completion_tokens,
    prompt_tokens, total_tokens, reasoning_tokens, structured_output).
    """
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

    content = None
    try:
        resp = requests.post(f"{BASE_URL}/chat/completions", headers=get_headers(), json=payload, timeout=600)
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
        reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)

        if finish_reason == "length":
            raise ValueError(
                f"Generation stopped (finish_reason='length', completion_tokens={completion_token}, "
                f"max_tokens={max_tokens}, max_completion_tokens={max_tokens}). "
            )

        if not content.strip():
            raise ValueError("Model returned empty/whitespace content.")

        parsed = json.loads(content)

        # Record whether the raw model output already matched the desired schema,
        # before any coercion. This feeds the "fraction structured output" metric.
        try:
            jsonschema.validate(instance=parsed, schema=schema)
            structured_output = True
        except jsonschema.ValidationError:
            structured_output = False

        parsed = coerce_extraction_json(parsed)

        # Proceed as long as we recovered an items list. Individual items may be
        # missing optional fields; those are left blank rather than dropped, and
        # structured_output already records that the raw output was non-conforming.
        if not isinstance(parsed.get("items"), list):
            raise ValueError("JSON output had no recoverable items list")

        return parsed, completion_token, prompt_token, total_token, reasoning_tokens, structured_output

    except (requests.RequestException, ValueError, AttributeError) as e:
        err = RuntimeError(f"Failed to get valid JSON. Last error: {e}")
        if content:
            err.raw_output = content
        raise err
