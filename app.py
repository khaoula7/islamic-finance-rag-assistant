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
if "answer" not in st.session_state:
    st.session_state.answer = None

if "retrieval_result" not in st.session_state:
    st.session_state.retrieval_result = None

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

        if contract is None:
            st.session_state.answer = FALLBACK_ANSWER
            st.session_state.retrieval_result = None

        else:
            retrieval_result = retrieve_context(
                query=query,
                collection=collection,
                embedder=embedder,
                top_k=TOP_K,
                contract=contract,
            )

            retrieved_docs = retrieval_result["documents"]

            answer = generate_answer(
                query=query,
                retrieved_docs=retrieved_docs,
                tokenizer=tokenizer,
                model=model,
            )

            st.session_state.answer = answer
            st.session_state.retrieval_result = retrieval_result

if st.session_state.answer is not None:
    st.subheader("Answer")
    st.write(st.session_state.answer)    
    
if st.session_state.retrieval_result is not None:
    retrieval_result = st.session_state.retrieval_result

    retrieved_docs = retrieval_result["documents"]
    retrieved_metadatas = retrieval_result["metadatas"]
    retrieved_distances = retrieval_result["distances"]

    with st.expander("View supporting evidence"):
        show_technical_details = st.checkbox(
            "Show technical retrieval details"
        )

        for index, document in enumerate(retrieved_docs):
            metadata = retrieved_metadatas[index]
            distance = retrieved_distances[index]

            source_name = (
                Path(metadata["source"]).stem.replace("_", " ").title()
            )

            section_name = metadata["section"]

            st.markdown(f"### {source_name} — {section_name}")
            st.write(document)

            if show_technical_details:
                st.caption(
                    f"Retrieval rank: {index + 1} | "
                    f"Chunk ID: {metadata['chunk_id']} | "
                    f"Distance: {distance:.4f} "
                    f"(lower means semantically closer)"
                )

            st.divider()
    
    
   
