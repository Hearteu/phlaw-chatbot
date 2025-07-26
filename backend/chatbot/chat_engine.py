retriever = None

def chat_with_law_bot(query):
    global retriever
    if retriever is None:
        from .retriever import LegalRetriever
        retriever = LegalRetriever()

    # Get top-k documents from retriever
    docs = retriever.retrieve(query)
    
    # Build context from the text field only
    context = "\n\n".join(f"Source: {doc['filename']}\n{doc['text']}" for doc in docs)

    # Create the prompt for the language model
    prompt = f"Answer the question using Philippine jurisprudence context:\n\n{context}\n\nQuestion: {query}\nAnswer:"

    # Call your generator to get the final answer
    from .generator import generate_response
    return generate_response(prompt)
