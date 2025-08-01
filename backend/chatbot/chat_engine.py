# chat_engine.py (memory-optimized)

retriever = None

def truncate_text(text, max_chars=1000):
    """Safely truncate a string to max_chars."""
    if not text:
        return ""
    return text[:max_chars]

def chat_with_law_bot(query, 
                      max_doc_chars=1000, 
                      max_total_chars=4096, 
                      min_docs=1):
    """
    Memory-safe chat engine that limits doc size and overall prompt length.
    """
    global retriever
    if retriever is None:
        from .retriever import LegalRetriever
        retriever = LegalRetriever()

    # Get top-k documents from retriever
    docs = retriever.retrieve(query)
    if not docs or len(docs) < min_docs:
        return "No relevant jurisprudence found."

    # Prepare context: only include truncated excerpts, and limit total prompt size
    context_parts = []
    total_chars = 0

    for doc in docs:
        excerpt = truncate_text(doc.get('text'), max_doc_chars)
        chunk = f"Source: {doc.get('filename')}\n{excerpt}"
        if total_chars + len(chunk) > max_total_chars:
            # Stop if adding this chunk would exceed limit
            break
        context_parts.append(chunk)
        total_chars += len(chunk)

    if not context_parts:
        return "Context could not be constructed due to document size limits."

    context = "\n\n".join(context_parts)

    prompt = (
        "Answer the question using Philippine jurisprudence context:\n\n"
        f"{context}\n\nQuestion: {query}\nAnswer:"
    )

    # Call your generator to get the final answer
    from .generator import generate_response
    return generate_response(prompt)
