# chat_engine.py
import re

RULING_REGEX = re.compile(r"(WHEREFORE.*?SO ORDERED\.?|ACCORDINGLY.*?SO ORDERED\.?)",
                          re.IGNORECASE | re.DOTALL)

retriever = None

def format_source_line(doc: dict) -> str:
    title = doc.get("title") or doc.get("gr_number") or doc.get("filename") or "Untitled case"
    url   = doc.get("source_url") or "N/A"
    return f"Source: {title} [{url}]"

def _ensure_section(doc: dict) -> str:
    sec = doc.get("section")
    if sec:
        return sec
    text = (doc.get("text") or "")
    return "ruling" if RULING_REGEX.search(text) else "body"

def chat_with_law_bot(query):
    global retriever
    if retriever is None:
        from .retriever import LegalRetriever
        retriever = LegalRetriever()

    docs = retriever.retrieve(query, k=6)
    if not docs:
        return "No relevant jurisprudence found."

    # ensure every doc has a section label
    for d in docs:
        d["section"] = _ensure_section(d)

    wants_ruling = "ruling" in query.lower()
    ruling_doc = next((d for d in docs if d.get("section") == "ruling"), None)

    # Always run the generator:
    # If the user mentions "ruling", put the ruling doc first in the context (no early return)
    if wants_ruling and ruling_doc:
        ordered = [ruling_doc] + [d for d in docs if d is not ruling_doc]
    else:
        ordered = sorted(docs, key=lambda d: {"ruling": 0, "header": 1}.get(d.get("section", "body"), 2))

    # Build compact, source-labeled context
    context = "\n\n".join(
        f"{format_source_line(d)} [{d.get('section','body')}]"
        f"\n{(d.get('text') or '')[:1200]}"
        for d in ordered[:4]
    )

    print("Context for query:", context)

    prompt = (
        "You are a legal assistant for PH jurisprudence.\n"
        "If the question asks for the ruling, quote the ruling verbatim first, then briefly explain it.\n"
        "Answer using the sources below.\n\n"
        f"Sources:\n{context}\n\nQuestion: {query}\nAnswer:"
    )

    from .generator import generate_response
    return generate_response(prompt)
