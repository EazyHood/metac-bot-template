"""Send one tiny request to each candidate free model and report which ones answer.

Run by .github/workflows/probe_free_models.yaml. Costs one request per model
against the OpenRouter free quota (50/day). Never prints the API key.
"""

import json
import os
import time
import urllib.error
import urllib.request

CANDIDATES = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "thinkingmachines/inkling:free",
    "thinkingmachines/inkling-small:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "poolside/laguna-s-2.1:free",
    "dots-studio/dots-3-note-preview:free",
]


def post(model: str, key: str) -> tuple[int, dict | str, float]:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with the single word OK."}],
            "max_tokens": 20,
        }
    ).encode()
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.load(response), time.perf_counter() - start
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode(errors="replace")[:300], time.perf_counter() - start
    except Exception as error:  # timeouts, DNS
        return 0, repr(error)[:300], time.perf_counter() - start


def main() -> None:
    key = os.environ["OPENROUTER_API_KEY"]
    for model in CANDIDATES:
        status, payload, seconds = post(model, key)
        # OpenRouter can answer HTTP 200 with an {"error": ...} body and no choices.
        if status == 200 and isinstance(payload, dict) and payload.get("choices"):
            provider = payload.get("provider", "?")
            text = (payload["choices"][0]["message"].get("content") or "").strip()[:40]
            print(f"OK   {model} | {seconds:5.1f} s | {provider} | {text!r}")
        else:
            print(f"FAIL {model} | {seconds:5.1f} s | HTTP {status} | {payload}")
        time.sleep(4)  # stay well under 20 requests/minute


if __name__ == "__main__":
    main()
