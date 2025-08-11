# chat_engine.py
retriever = None

def format_source_line(doc: dict) -> str:
    title = doc.get("title") or doc.get("gr_number") or doc.get("filename") or "Untitled case"
    url   = doc.get("source_url") or "N/A"
    return f"Source: {title} [{url}]"

def chat_with_law_bot(query):
    global retriever
    if retriever is None:
        from .retriever import LegalRetriever
        retriever = LegalRetriever()

    docs = retriever.retrieve(query, k=6)
    if not docs:
        return "No relevant jurisprudence found."

    # If the question mentions "ruling" and we have a ruling section, answer verbatim.
    wants_ruling = "ruling" in query.lower()
    ruling_doc = next((d for d in docs if d.get("section") == "ruling"), None)

    if wants_ruling and ruling_doc:
        # return the ruling text directly (cleaned)
        return ruling_doc.get("text", "").strip()

    # Otherwise, build a tight context (favor ruling/header first)
    ordered = sorted(docs, key=lambda d: {"ruling":0, "header":1}.get(d.get("section","body"),2))
    context = "\n\n".join(
        f"Source: {d.get('filename')} [{d.get('section')}]"
        f"\n{d.get('text')[:1200]}"
        for d in ordered[:4]
    )
    print("Context for query:", context)

    prompt = (
        "You are a legal assistant for PH jurisprudence.\n"
        # "Answer strictly from the provided sources; if unknown, say you don’t know.\n\n"
        "Quote directly for holdings/rulings; keep the answer faithful to the text.\n\n"
        f"Sources:\n{context}\n\nQuestion: {query}\nAnswer:"
    )

    from .generator import generate_response
    return generate_response(prompt)
