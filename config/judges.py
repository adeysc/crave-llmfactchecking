"""Judge configuration: the openai (BYOK) backends.

The judge PROMPT (src/judge.py) is identical regardless of backend; this file only
declares HOW to reach each model. `src/backends.py` reads it to build the
`prompt -> completion` callable.

  openai      : the paper's judge (GPT-4o). Bring your own key: export OPENAI_API_KEY=...
                This is the path that reproduces the paper's numbers.
  openai-mini : cost-controlled gpt-4o-mini, input capped at 4000 tokens.
"""
from __future__ import annotations

DEFAULT_JUDGE = "openai"

JUDGES = {
    "openai": {
        "kind": "openai",
        "model": "gpt-4o",
        "env_key": "OPENAI_API_KEY",
        "temperature": 0,
        "max_input_tokens": 16000,   # truncate over-long prompts (cl100k) before sending
    },
    # Cost-controlled mini judge: same openai path, gpt-4o-mini only, input capped at 4000 tokens.
    "openai-mini": {
        "kind": "openai",
        "model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
        "temperature": 0,
        "max_input_tokens": 4000,
    },
}
