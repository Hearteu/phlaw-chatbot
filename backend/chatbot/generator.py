# import torch

# torch.set_num_threads(1)  # Prevent CPU contention

# from transformers import AutoModelForCausalLM, AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("AdaptLLM/law-chat")
# model = AutoModelForCausalLM.from_pretrained(
#     "AdaptLLM/law-chat",
#     torch_dtype=torch.float16  # if you have GPU!
# )
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# def generate_response(prompt, max_new_tokens=64):
#     prompt = prompt[:1500]
#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=False,  # for speed/determinism
#             pad_token_id=tokenizer.eos_token_id
#         )
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response.replace(prompt, "").strip()


import os
import re
from typing import Any, Dict, List, Optional

import httpx
from llama_cpp import Llama

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "law-chat.Q5_K_M.gguf")

# Lazy-init local model to avoid loading if Rev21 API is configured and healthy
_llm: Optional[Llama] = None

REV21_BASE_URL = os.getenv("REV21_BASE_URL", "https://ai-tools.rev21labs.com/api/v1")
REV21_API_KEY = os.getenv("REV21_API_KEY", "MTc0NTI4NTItYzNkYS00NmQ0LWI0MTktMDc2MmVhYjc2OWE3")
REV21_ENDPOINT_PATH = os.getenv("REV21_ENDPOINT_PATH", "/chat/completions")
REV21_ENABLED = os.getenv("REV21_ENABLED", "true").lower() not in {"0", "false", "no"}


def _ensure_llm() -> Llama:
    global _llm
    if _llm is None:
        _llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=4096,
            n_gpu_layers=-1,
            n_threads=8,
        )
    return _llm


def _parse_rev21_response(data: Dict[str, Any]) -> Optional[str]:
    # Common shapes: OpenAI-like {choices: [{text|message:{content}}]}, or flat {text|completion|output}
    try:
        if isinstance(data, dict):
            if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                first = data["choices"][0]
                if isinstance(first, dict):
                    if "message" in first and isinstance(first["message"], dict):
                        content = first["message"].get("content")
                        if isinstance(content, str) and content.strip():
                            return content
                    text = first.get("text")
                    if isinstance(text, str) and text.strip():
                        return text
            for key in ("text", "completion", "output", "result"):
                val = data.get(key)
                if isinstance(val, str) and val.strip():
                    return val
            if "data" in data and isinstance(data["data"], dict):
                inner = data["data"]
                for key in ("text", "completion", "output", "result"):
                    val = inner.get(key)
                    if isinstance(val, str) and val.strip():
                        return val
    except Exception:
        return None
    return None


def _call_rev21_prompt(prompt: str) -> Optional[str]:
    if not REV21_ENABLED:
        return None
    if not REV21_API_KEY:
        return None

    url_candidates = [
        f"{REV21_BASE_URL.rstrip('/')}{REV21_ENDPOINT_PATH}",
        f"{REV21_BASE_URL.rstrip('/')}/chat",
        f"{REV21_BASE_URL.rstrip('/')}/generate",
    ]
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "x-api-key": REV21_API_KEY,
    }
    payload: Dict[str, Any] = {
        "prompt": prompt,
        "max_tokens": 900,
        "temperature": 0.4,
        "top_p": 0.9,
    }

    with httpx.Client(timeout=30.0) as client:
        for url in url_candidates:
            try:
                resp = client.post(url, headers=headers, json=payload)
                if resp.status_code >= 400:
                    continue
                data = resp.json()
                text = _parse_rev21_response(data)
                if isinstance(text, str) and text.strip():
                    text = re.sub(r"\s*\[/?INST\]\s*", " ", text)
                    return text.strip()
            except Exception:
                continue
    return None


def _call_rev21_messages(messages: List[Dict[str, str]]) -> Optional[str]:
    if not REV21_ENABLED or not REV21_API_KEY:
        return None

    url = f"{REV21_BASE_URL.rstrip('/')}{REV21_ENDPOINT_PATH}"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "x-api-key": REV21_API_KEY,
    }
    payload: Dict[str, Any] = {
        "messages": messages,
        "max_tokens": 900,
        "temperature": 0.4,
        "top_p": 0.9,
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            if resp.status_code >= 400:
                return None
            data = resp.json()
            text = _parse_rev21_response(data)
            if isinstance(text, str) and text.strip():
                text = re.sub(r"\s*\[/?INST\]\s*", " ", text)
                return text.strip()
    except Exception:
        return None
    return None


def generate_response(prompt: str) -> str:
    # Try Rev21 Labs API first
    remote = _call_rev21_prompt(prompt)
    if isinstance(remote, str) and remote.strip():
        return remote

    # Fallback to local llama model
    llm = _ensure_llm()
    output = llm(
        prompt,
        max_tokens=900,
        temperature=0.4,
        top_p=0.9,
        repeat_penalty=1.1,
        stop=["User:", "Sources:", "Question:", "\n\n"],
    )
    text = output["choices"][0]["text"]
    text = re.sub(r"\s*\[/?INST\]\s*", " ", text)
    return text.strip()


def generate_response_from_messages(messages: List[Dict[str, str]]) -> str:
    # Prefer Rev21 chat-style endpoint
    remote = _call_rev21_messages(messages)
    if isinstance(remote, str) and remote.strip():
        return remote

    # Fallback: stitch messages into a single prompt for local model
    stitched = []
    for m in messages[-12:]:
        role = (m.get("role") or "user").lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        stitched.append(f"{role.capitalize()}: {content}")
    prompt = "\n".join(stitched)
    return generate_response(prompt)
