retriever = None

def chat_with_law_bot(query):
    """
    Simple chat engine that gives all retrieved context to the generator.
    """
    global retriever
    if retriever is None:
        from .retriever import LegalRetriever
        retriever = LegalRetriever()

    # Get top-k documents from retriever
    docs = retriever.retrieve(query)
    if not docs:
        return "No relevant jurisprudence found."

    # Prepare context: use full text of each doc
    context = "\n\n".join(
        f"Source: {doc.get('filename')}\n{doc.get('text')}" for doc in docs
    )

    prompt = (
        f"You are a friendly legal assistant. "
        f"Answer the user's question using the Philippine jurisprudence sources below. "
        f"Explain clearly, summarize where possible, and use conversational language. "
        # f"Do not rephrase issues, just provide the answer in same as the source.\n\n"
        f"Sources:\n{context}\n\nUser Question: {query}\nConversational Answer:"
    )

    # Call your generator to get the final answer
    from .generator import generate_response
    return generate_response(prompt)
