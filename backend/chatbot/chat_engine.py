retriever = None

def chat_with_law_bot(query):
    global retriever
    if retriever is None:
        from .retriever import LegalRetriever
        retriever = LegalRetriever()

    docs = retriever.retrieve(query)
    context = "\n\n".join(docs)
    from .generator import generate_response
    prompt = f"Answer the question using Philippine jurisprudence context:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    return generate_response(prompt)
