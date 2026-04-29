"""
LLM Provider Registry
======================
Stateless, provider-isolated LLM calls.

Each provider is fully independent:
  - Only the selected provider's import / key is touched
  - No shared client state between providers
  - Switching providers creates a fresh client every time

Public API
----------
  validate_provider_config(provider, api_key, model) -> (ok: bool, error: str)
  call_llm(provider, model, api_key, system_prompt, user_prompt,
            temperature, max_tokens) -> str
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

# ── Provider constants ────────────────────────────────────────────────────────

PROVIDER_OPENAI    = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_LOCAL     = "local"
PROVIDER_GROQ      = "groq"

ALL_PROVIDERS = (PROVIDER_OPENAI, PROVIDER_ANTHROPIC, PROVIDER_LOCAL, PROVIDER_GROQ)

DEFAULT_MODELS: dict[str, str] = {
    PROVIDER_OPENAI:    "gpt-4o-mini",
    PROVIDER_ANTHROPIC: "claude-haiku-4-5-20251001",
    PROVIDER_LOCAL:     "mistralai/Mistral-7B-Instruct-v0.3",
    PROVIDER_GROQ:      "llama-3.3-70b-versatile",
}

# ENV variable names per provider (used only when UI key is blank)
ENV_KEYS: dict[str, str] = {
    PROVIDER_OPENAI:    "OPENAI_API_KEY",
    PROVIDER_ANTHROPIC: "ANTHROPIC_API_KEY",
    PROVIDER_LOCAL:     "HF_TOKEN",          # optional for public models
    PROVIDER_GROQ:      "GROQ_API_KEY",
}

# Models available in the UI per provider
# HuggingFace models that support chat_completion via the Inference API.
# These are free-tier serverless endpoints — a HF token unlocks higher limits.
# The first entry is the default.
HF_CHAT_MODELS: list[str] = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "HuggingFaceH4/zephyr-7b-beta",
    "microsoft/Phi-3-mini-4k-instruct",
    "google/gemma-2-2b-it",
    "Qwen/Qwen2.5-7B-Instruct",
]

# Groq-hosted models (fast inference, free tier available)
# Full list: https://console.groq.com/docs/models
GROQ_MODELS: list[str] = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

AVAILABLE_MODELS: dict[str, list[str]] = {
    PROVIDER_ANTHROPIC: [
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-6",
        "claude-opus-4-6",
    ],
    PROVIDER_OPENAI: [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-3.5-turbo",
    ],
    PROVIDER_LOCAL: HF_CHAT_MODELS,
    PROVIDER_GROQ:  GROQ_MODELS,
}


# ── Key resolution ────────────────────────────────────────────────────────────

def _resolve_key(provider: str, ui_key: str) -> str:
    """
    Return the API key to use for *provider*.

    Priority: UI field → environment variable.
    For LOCAL provider an empty string is fine (public HF models need no token).
    """
    if ui_key and ui_key.strip():
        return ui_key.strip()
    env_var = ENV_KEYS.get(provider, "")
    return os.environ.get(env_var, "")


# ── Validation ────────────────────────────────────────────────────────────────

def validate_provider_config(
    provider: str,
    api_key: str = "",
    model: str = "",
) -> Tuple[bool, str]:
    """
    Validate that the selected provider can be used.

    Returns (True, "") on success or (False, human_readable_error) on failure.
    Does NOT make a live API call — only checks imports and key presence.
    """
    if provider not in ALL_PROVIDERS:
        return False, f"Unknown provider '{provider}'. Choose from: {ALL_PROVIDERS}"

    if provider == PROVIDER_OPENAI:
        try:
            import openai  # noqa: F401
        except ImportError:
            return False, "openai package not installed. Run: pip install openai"
        key = _resolve_key(provider, api_key)
        if not key:
            return False, (
                "OpenAI API key not found. "
                "Enter it in the sidebar or set the OPENAI_API_KEY environment variable."
            )

    elif provider == PROVIDER_ANTHROPIC:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False, "anthropic package not installed. Run: pip install anthropic"
        key = _resolve_key(provider, api_key)
        if not key:
            return False, (
                "Anthropic API key not found. "
                "Enter it in the sidebar or set the ANTHROPIC_API_KEY environment variable."
            )

    elif provider == PROVIDER_LOCAL:
        try:
            from huggingface_hub import InferenceClient  # noqa: F401
        except ImportError:
            return False, (
                "huggingface_hub package not installed. "
                "Run: pip install huggingface_hub"
            )
        # HF token is optional for public/serverless models; no hard error here.
        # If the model requires auth the API call itself will return a clear error.

    elif provider == PROVIDER_GROQ:
        try:
            from groq import Groq  # noqa: F401
        except ImportError:
            return False, "groq package not installed. Run: pip install groq"
        key = _resolve_key(provider, api_key)
        if not key:
            return False, (
                "Groq API key not found. "
                "Enter it in the sidebar or set the GROQ_API_KEY environment variable. "
                "Get a free key at https://console.groq.com/keys"
            )

    return True, ""


# ── Provider-specific callers ─────────────────────────────────────────────────

def _call_openai(
    model: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    import openai
    key = _resolve_key(PROVIDER_OPENAI, api_key)
    if not key:
        raise ValueError(
            "OpenAI API key not found. "
            "Enter it in the sidebar or set OPENAI_API_KEY."
        )
    client = openai.OpenAI(api_key=key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def _call_anthropic(
    model: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    import anthropic
    key = _resolve_key(PROVIDER_ANTHROPIC, api_key)
    if not key:
        raise ValueError(
            "Anthropic API key not found. "
            "Enter it in the sidebar or set ANTHROPIC_API_KEY."
        )
    client = anthropic.Anthropic(api_key=key)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text.strip()


def _call_local(
    model: str,
    api_key: str,          # HF token — optional for public serverless models
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """
    Run inference via the HuggingFace Inference API (serverless endpoints).

    Uses huggingface_hub.InferenceClient.chat_completion — no PyTorch or local
    model download required.  A HF token is optional for public models but
    strongly recommended to avoid rate-limiting on free-tier endpoints.
    """
    from huggingface_hub import InferenceClient

    hf_token = _resolve_key(PROVIDER_LOCAL, api_key) or None

    client = InferenceClient(
        model=model,
        token=hf_token,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=max(temperature, 0.01),  # some models reject 0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as chat_err:
        # Fallback: text_generation for models that don't support the chat endpoint
        combined_prompt = (
            "<s>[INST] <<SYS>>\n"
            + system_prompt
            + "\n<</SYS>>\n\n"
            + user_prompt
            + " [/INST]"
        )
        try:
            raw = client.text_generation(
                prompt=combined_prompt,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                return_full_text=False,
            )
            return raw.strip()
        except Exception as tg_err:
            raise RuntimeError(
                "HuggingFace Inference API failed.\n"
                f"  chat_completion error : {chat_err}\n"
                f"  text_generation error : {tg_err}\n"
                f"Possible causes: model '{model}' may not be available on the "
                "free serverless tier, or a HuggingFace token is required. "
                "Get a free token at https://huggingface.co/settings/tokens"
            ) from tg_err


def _call_groq(
    model: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """
    Call Groq's lightning-fast inference API.

    Groq uses an OpenAI-compatible chat completions interface.
    Free tier available at https://console.groq.com — no credit card required.
    """
    from groq import Groq

    key = _resolve_key(PROVIDER_GROQ, api_key)
    if not key:
        raise ValueError(
            "Groq API key not found. "
            "Enter it in the sidebar or set GROQ_API_KEY. "
            "Get a free key at https://console.groq.com/keys"
        )

    client = Groq(api_key=key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


# ── Unified entry point ───────────────────────────────────────────────────────

def call_llm(
    provider: str,
    model: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> str:
    """
    Dispatch an LLM call to the selected provider.

    Only the logic for *provider* is executed; other providers are never
    imported or queried.

    Raises ValueError / ImportError with a clear, provider-specific message
    on misconfiguration.
    """
    if provider == PROVIDER_OPENAI:
        return _call_openai(model, api_key, system_prompt, user_prompt, temperature, max_tokens)

    elif provider == PROVIDER_ANTHROPIC:
        return _call_anthropic(model, api_key, system_prompt, user_prompt, temperature, max_tokens)

    elif provider == PROVIDER_LOCAL:
        return _call_local(model, api_key, system_prompt, user_prompt, max_tokens, temperature)

    elif provider == PROVIDER_GROQ:
        return _call_groq(model, api_key, system_prompt, user_prompt, temperature, max_tokens)

    else:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Valid options: {', '.join(ALL_PROVIDERS)}"
        )
