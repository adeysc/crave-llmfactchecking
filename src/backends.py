"""Build a `prompt -> completion` callable for a named judge (see config/judges.py).

  openai : POST to the OpenAI chat-completions API (BYOK via OPENAI_API_KEY).
"""
from __future__ import annotations

import os
import json
import time

from config.judges import JUDGES


def _openai_backend(cfg):
    """Return a callable that sends one prompt to the OpenAI chat API and returns text."""
    import requests
    try:
        import tiktoken
        _enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _enc = None

    api_key = os.environ.get(cfg["env_key"], "")
    if not api_key:
        raise RuntimeError(f"{cfg['env_key']} is not set (the openai judge is BYOK).")
    model = cfg["model"]
    temperature = cfg.get("temperature", 0)
    max_tok = cfg.get("max_input_tokens", 16000)
    url = "https://api.openai.com/v1/chat/completions"

    def complete(prompt):
        if _enc is not None:                         # truncate over-long prompts
            toks = _enc.encode(prompt)
            if len(toks) > max_tok:
                prompt = _enc.decode(toks[:max_tok])
                print("TOKEN OVERLOAD >> LIMITED")
        payload = {"model": model, "temperature": temperature,
                   "messages": [{"role": "user", "content": prompt}]}
        headers = {"Authorization": f"Bearer {api_key}"}
        data = json.loads(requests.post(url, headers=headers, json=payload).text)
        while "error" in data:                       # wait out rate limits
            print(data["error"], "\nwaiting 20s for ratelimit")
            time.sleep(20)
            data = json.loads(requests.post(url, headers=headers, json=payload).text)
        return data["choices"][0]["message"]["content"]

    return complete


def make_backend(name):
    """Return a `prompt -> str` callable for a judge name in config/judges.py."""
    if name not in JUDGES:
        raise ValueError(f"Unknown judge {name!r}. Options: {sorted(JUDGES)}")
    cfg = JUDGES[name]
    if cfg["kind"] == "openai":
        return _openai_backend(cfg)
    raise ValueError(f"Unknown backend kind {cfg['kind']!r} for judge {name!r}")
