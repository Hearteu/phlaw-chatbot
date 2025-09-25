import json
import os
from typing import Dict, List


def _load_dataset() -> List[Dict[str, str]]:
    samples: List[Dict[str, str]] = [
        {
            "prompt": "G.R. No. 149311",
            "reference": (
                "Case Digest: ... Dispositive: WHEREFORE ... SO ORDERED."
            ),
        },
        {
            "prompt": "Perjury",
            "reference": (
                "Here are the possible cases:\n1. ..."
            ),
        },
    ]

    # Optional external dataset
    external_path = os.path.join(os.path.dirname(__file__), "data", "samples", "bleu_eval.jsonl")
    if os.path.exists(external_path):
        with open(external_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                prompt = (row.get("prompt") or row.get("query") or "").strip()
                ref = (row.get("reference") or row.get("ref") or "").strip()
                if prompt and ref:
                    samples.append({"prompt": prompt, "reference": ref})
    return samples


def _call_bot(prompt: str) -> str:
    # Call the in-process chat engine directly
    from chatbot.chat_engine import chat_with_law_bot

    try:
        out = chat_with_law_bot(prompt, history=None)
        if not isinstance(out, str):
            out = str(out)
        return out.strip()
    except Exception as e:
        return f"[ERROR] {e}".strip()


def main() -> None:
    try:
        import sacrebleu
    except Exception:
        print("sacrebleu is not installed. Please run: pip install -r requirements.txt")
        return

    data = _load_dataset()
    sys_outputs: List[str] = []
    references: List[str] = []

    print(f"Running BLEU on {len(data)} samples...\n")
    for idx, ex in enumerate(data, 1):
        prompt = ex["prompt"]
        reference = ex["reference"]
        output = _call_bot(prompt)
        sys_outputs.append(output)
        references.append(reference)
        sys_snip = output[:200].replace("\n", " ")
        if len(output) > 200:
            sys_snip += "..."
        ref_snip = reference[:200].replace("\n", " ")
        if len(reference) > 200:
            ref_snip += "..."
        print(f"[{idx}] Prompt: {prompt}")
        print(f"System: {sys_snip}")
        print(f"Reference: {ref_snip}")
        print()

    # sacrebleu expects list of system outputs and list of list of references
    bleu = sacrebleu.corpus_bleu(sys_outputs, [references])
    print("Aggregate BLEU:")
    print(bleu)


if __name__ == "__main__":
    main()



