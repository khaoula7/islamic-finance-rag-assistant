from pathlib import Path

import streamlit as st

from generator import (
    FALLBACK_ANSWER,
    generate_answer,
    load_generator_model,
)
from ingest import initialize_knowledge_base
from retriever import detect_contract, retrieve_context


DATA_DIR = Path("data")
TOP_K = 2


st.set_page_config(
    page_title="Islamic Finance RAG Assistant",
    page_icon="🤖",
)

st.title("🤖 Islamic Finance RAG Assistant")
st.write(
    "Ask questions about Murabaha, Ijara, or Sukuk "
    "based on the provided context."
)

@st.cache_resource
def load_rag_system():
    """
    Initialize and cache the knowledge base.
    """
    return initialize_knowledge_base(DATA_DIR)


@st.cache_resource
def load_llm():
    """
    Load and cache the language model.
    """
    return load_generator_model()


collection, embedder = load_rag_system()
tokenizer, model = load_llm()

query = st.text_input("Ask a question:")

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a question.")

    else:
        contract = detect_contract(query)

        st.subheader("Answer")

        if contract is None:
            st.write(FALLBACK_ANSWER)

        else:
            retrieval_result = retrieve_context(
                query=query,
                collection=collection,
                embedder=embedder,
                top_k=TOP_K,
                contract=contract,
            )

            retrieved_docs = retrieval_result["documents"]
            retrieved_metadatas = retrieval_result["metadatas"]

            answer = generate_answer(
                query=query,
                retrieved_docs=retrieved_docs,
                tokenizer=tokenizer,
                model=model,
            )

            st.write(answer)

            with st.expander("See Retrieved Context"):
                for index, document in enumerate(retrieved_docs):
                    st.markdown(f"**Chunk {index + 1}:**")
                    st.write(
                        "Metadata:",
                        retrieved_metadatas[index],
                    )
                    st.write(document)
                    st.divider()