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
    "Ask questions about **Murabaha**, **Ijara**, or **Sukuk**. "
    "Answers are generated only from the project's knowledge base."
)

st.caption(
    "Educational tool only — responses should not be treated as "
    "financial advice or a final Shariah ruling."
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

def clear_results():
    """
    Clear the current question, answer, and retrieval evidence.
    """
    st.session_state.query_input = ""
    st.session_state.answer = None
    st.session_state.retrieval_result = None


collection, embedder = load_rag_system()
tokenizer, model = load_llm()

query = st.text_input(
    "Your question",
    placeholder="For example: Must the seller own the asset in Murabaha?",
    key="query_input",
)

# The first column is three times wider than the second,
answer_column, clear_column = st.columns([3, 1])

with answer_column:
    get_answer_clicked = st.button(
        "Get Answer",
        type="primary",
        use_container_width=True,
    )

with clear_column:
    st.button(
        "Clear",
        use_container_width=True,
        on_click=clear_results,
    )

if get_answer_clicked:
    if not query.strip():
        st.warning("Please enter a question.")

    else:
        contract = detect_contract(query)

        if contract is None:
            st.session_state.answer = FALLBACK_ANSWER
            st.session_state.retrieval_result = None

        else:
            with st.spinner("Searching the knowledge base and generating an answer..."):
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

    st.markdown("#### Sources used")

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
            
            if(index < TOP_K -1):
                st.divider()
            else:
                st.markdown("\n")
    
    
   
