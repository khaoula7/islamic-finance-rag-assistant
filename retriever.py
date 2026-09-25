SUPPORTED_CONTRACTS = ("murabaha", "ijara", "sukuk")

def detect_contract(query: str) -> str | None:
    """
    Detect which supported Islamic finance contract
    is explicitly mentioned in the user's question.
    """
    normalized_query = query.lower()

    for contract in SUPPORTED_CONTRACTS:
        if contract in normalized_query:
            return contract

    return None


def retrieve_context(query: str, collection, embedder, top_k: int = 2, contract: str | None = None,) -> dict:
    """
    Retrieve the most relevant chunks for a question.

    Returns the retrieved documents, metadata, and distances.
    """
    if not query.strip():
        raise ValueError("The retrieval query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    query_embedding = embedder.encode([query]).tolist()

    query_arguments = {
        "query_embeddings": query_embedding,
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }

    if contract is not None:
        query_arguments["where"] = {"contract": contract}

    results = collection.query(**query_arguments)

    return {
        "documents": results["documents"][0],
        "metadatas": results["metadatas"][0],
        "distances": results["distances"][0],
    }