import streamlit as st
from ingest import (
    load_text_files,
    chunk_text,
    build_vector_store,
    detect_contract,
    generate_answer_eval,
)
from pathlib import Path

DATA_DIR = Path("data")

st.set_page_config(page_title="Islamic Finance RAG Assistant", page_icon="🤖")

st.title("🤖 Islamic Finance RAG Assistant")
st.write("Ask questions about Murabaha, Ijara, or Sukuk based on the provided context.")

@st.cache_resource
def load_rag_system():
    texts = load_text_files(DATA_DIR)

    all_chunks = []
    for name, content in texts.items():
        file_chunks = chunk_text(content, name)
        all_chunks.extend(file_chunks)

    collection, embedder = build_vector_store(all_chunks)
    return collection, embedder

collection, embedder = load_rag_system()

query = st.text_input("Ask a question:")

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        contract = detect_contract(query)

        if contract is None:
            st.subheader("Answer")
            st.write("I don't know based on the provided context.")
        else:
            q_emb = embedder.encode([query]).tolist()

            results = collection.query(
                query_embeddings=q_emb,
                n_results=2,
                where={"contract": contract},
            )

            retrieved_docs = results["documents"][0]
            retrieved_metadatas = results["metadatas"][0]

            retrieved_docs = results["documents"][0]
            answer = generate_answer_eval(query, retrieved_docs)

            st.subheader("Answer")
            st.write(answer)

            with st.expander("See Retrieved Context"):
                for i, doc in enumerate(retrieved_docs):
                    st.markdown(f"**Chunk {i+1}:**")
                    if retrieved_metadatas:
                        st.write("Metadata:", retrieved_metadatas[i])
                    st.write(doc)
                    st.write("---")