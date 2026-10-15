"""Check which no-cost model providers this bot can reach without new keys.

Run by .github/workflows/probe_free_models.yaml. One tiny request per target:
- the Metaculus LLM proxy (OpenAI and Anthropic routes), with METACULUS_TOKEN
- GitHub Models, with the workflow's GITHUB_TOKEN (needs `models: read`)
Never prints a token.
"""

import json
import os
import time
import urllib.error
import urllib.request

PROMPT = "Reply with the single word OK."


def call(url: str, headers: dict, body: dict | None = None) -> tuple[int, str, float]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, headers=headers)
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, response.read().decode(errors="replace"), time.perf_counter() - start
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode(errors="replace"), time.perf_counter() - start
    except Exception as error:  # timeouts, DNS
        return 0, repr(error), time.perf_counter() - start


def report(label: str, status: int, text: str, seconds: float) -> None:
    verdict = "OK  " if status == 200 else "FAIL"
    print(f"{verdict} {label} | {seconds:5.1f} s | HTTP {status} | {text[:300]!r}")


def probe_metaculus_proxy(token: str) -> None:
    auth = {"Authorization": f"Token {token}", "Content-Type": "application/json"}
    for model in ["gpt-4o", "gpt-4o-mini"]:
        status, text, seconds = call(
            "https://llm-proxy.metaculus.com/proxy/openai/v1/chat/completions",
            auth,
            {"model": model, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 5},
        )
        report(f"metaculus-proxy openai/{model}", status, text, seconds)
        time.sleep(3)
    status, text, seconds = call(
        "https://llm-proxy.metaculus.com/proxy/anthropic/v1/messages",
        {**auth, "anthropic-version": "2023-06-01"},
        {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 5,
            "messages": [{"role": "user", "content": PROMPT}],
        },
    )
    report("metaculus-proxy anthropic/claude-sonnet-4", status, text, seconds)


def probe_github_models(token: str) -> None:
    auth = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
    }
    status, text, seconds = call("https://models.github.ai/catalog/models", auth)
    try:
        catalog = json.loads(text) if status == 200 else None
    except json.JSONDecodeError:
        catalog = None
    if isinstance(catalog, list):
        print(f"OK   github-models catalog | {len(catalog)} models")
        for model in catalog:
            print(f"     {model.get('id')} | tier={model.get('rate_limit_tier')} | {model.get('publisher')}")
    else:
        report("github-models catalog", status, text, seconds)
    for model in ["openai/gpt-4.1", "openai/gpt-5", "deepseek/deepseek-r1"]:
        time.sleep(3)
        status, text, seconds = call(
            "https://models.github.ai/inference/chat/completions",
            auth,
            {"model": model, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 20},
        )
        report(f"github-models {model}", status, text, seconds)


def main() -> None:
    probe_metaculus_proxy(os.environ["METACULUS_TOKEN"])
    probe_github_models(os.environ["GITHUB_TOKEN"])


if __name__ == "__main__":
    main()
