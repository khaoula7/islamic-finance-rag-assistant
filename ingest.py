from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "islamic_finance_rag"


def load_text_files(data_dir: Path) -> dict[str, str]:
    """
    Load all .txt files from the data directory.

    Returns:
        A dictionary in the form:
        {
            "murabaha.txt": "document content...",
            "ijara.txt": "document content..."
        }
    """
    texts: dict[str, str] = {}

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data folder not found: {data_dir.resolve()}"
        )

    txt_files = sorted(data_dir.glob("*.txt"))

    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in: {data_dir.resolve()}"
        )

    for file_path in txt_files:
        texts[file_path.name] = file_path.read_text( encoding="utf-8").strip()

    return texts

def split_document_sections(document_text: str) -> list[tuple[str, str]]:
    """
    Split a document into sections using blank lines.
    Returns:
        A list of:
        [
            ("MURABAHA Definition", "Murabaha is..."),
            ("MURABAHA Conditions", "The asset must...")
        ]
    """
    blocks = document_text.split("\n\n")
    sections = []

    for block in blocks:
        block = block.strip()

        if not block:
            continue

        lines = block.splitlines()

        # Ignore a title that has no body text.
        if len(lines) == 1:
            continue

        section_name = lines[0].rstrip(":").strip()
        section_content = "\n".join(lines[1:]).strip()

        if section_content:
            sections.append(
                (section_name, section_content)
            )

    return sections

def chunk_text(document_text: str, source_name: str) -> list[dict]:
    """
    Split a document section by section and attach
    explainable metadata to every chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=40,
        length_function=len,
    )

    contract_type = Path(source_name).stem.lower()
    document_sections = split_document_sections(document_text)

    chunks = []
    chunk_id = 0

    for section_name, section_content in document_sections:
        section_text = (f"{section_name}:\n{section_content}")

        section_chunks = splitter.split_text(section_text)

        for text in section_chunks:
            chunks.append(
                {
                    "text": text,
                    "source": source_name,
                    "contract": contract_type,
                    "section": section_name,
                    "chunk_id": chunk_id,
                }
            )

            chunk_id += 1

    return chunks

def create_chunks(texts: dict[str, str]) -> list[dict]:
    """
    Convert all loaded documents into structured chunks.
    """
    all_chunks = []

    for source_name, document_text in texts.items():
        document_chunks = chunk_text(
            document_text=document_text,
            source_name=source_name,
        )
        all_chunks.extend(document_chunks)

    return all_chunks


def build_vector_store(chunks: list[dict]):
    """
    Create embeddings and store the chunks in ChromaDB.

    Returns:
        collection: The ChromaDB collection.
        embedder: The embedding model used for indexing.
    """
    if not chunks:
        raise ValueError("Cannot build a vector store without chunks.")

    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]

    print("Creating document embeddings...")
    embeddings = embedder.encode(texts).tolist()

    print("Initializing ChromaDB...")
    client = chromadb.Client()
    collection = client.create_collection(
        name=COLLECTION_NAME
    )

    ids = [
        f"{chunk['contract']}_{chunk['chunk_id']}"
        for chunk in chunks
    ]

    metadatas = [
        {
            "source": chunk["source"],
            "contract": chunk["contract"],
            "section": chunk["section"],
            "chunk_id": chunk["chunk_id"],
        }
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return collection, embedder


def initialize_knowledge_base(data_dir: Path):
    """
    Run the complete ingestion pipeline.

    Files -> chunks -> embeddings -> ChromaDB
    """
    texts = load_text_files(data_dir)
    chunks = create_chunks(texts)
    collection, embedder = build_vector_store(chunks)

    return collection, embedder