# parsing the evaluator's JSON response

import json
import re

# extract (correct, speech) from an evaluator response
def parse_evaluator_response(raw: str) -> tuple[bool, str]:
    text = (raw or "").strip()
    if not text:
        return False, "I couldn't judge that one. Let's move on."

    # strip code fences if present, then try to parse
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    for candidate in (text, cleaned):
        parsed = try_json(candidate)
        if parsed is not None:
            return parsed

    # pull out the first JSON object embedded in surrounding prose
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        parsed = try_json(match.group(0))
        if parsed is not None:
            return parsed

    # no usable JSON - fall back to the old keyword convention so the
    # turn still completes, and speak whatever the model produced
    upper = cleaned.upper()
    is_correct = upper.startswith("CORRECT") or "IS CORRECT" in upper
    return is_correct, cleaned

# parse one candidate string, return None if it's not usable
def try_json(candidate: str) -> tuple[bool, str] | None:
    try:
        data = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(data, dict) or "correct" not in data:
        return None

    correct = data["correct"]
    # some models return the boolean as a string despite the instruction
    if isinstance(correct, str):
        correct = correct.strip().lower() in ("true", "yes", "correct")

    speech = str(data.get("speech", "")).strip()
    if not speech:
        speech = "Correct!" if correct else "Not this time."

    return bool(correct), speech