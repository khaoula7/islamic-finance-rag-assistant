from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from test_cases import test_cases

DATA_DIR = Path("data")

def load_text_files(data_dir: Path) -> dict[str, str]:
    """
    Loads all .txt files in data_dir and returns a dict:
    {filename: file_text}
    """
    texts: dict[str, str] = {}

    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_dir.resolve()}")

    txt_files = sorted(data_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in: {data_dir.resolve()}")

    for fp in txt_files:
        texts[fp.name] = fp.read_text(encoding="utf-8").strip()

    return texts

def chunk_text(combined_text: str, source_name: str):
    """
    Split text into overlapping chunks with metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=40,
        length_function=len,
    )

    raw_chunks = splitter.split_text(combined_text)

    chunks = []
    for i, text in enumerate(raw_chunks):
        contract_type = source_name.replace(".txt", "").lower()

        chunks.append({
            "text": text,
            "source": source_name,
            "contract": contract_type,
            "chunk_id": i,
        })

    return chunks

def build_vector_store(chunks: list[str]):
    """
    Create embeddings and store them in Chroma.
    """
    print("\n🔄 Loading embedding model...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("🔄 Creating embeddings...")
    texts_only = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts_only).tolist()

    print("\n🔬 Embedding shape check:")
    print("Number of chunks:", len(chunks))
    print("Embedding length of first chunk:", len(embeddings[0]))

    print("🔄 Initializing Chroma...")
    client = chromadb.Client()
    collection = client.create_collection(name="islamic_finance_week1")

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        documents=texts_only,
        metadatas=chunks,
        embeddings=embeddings,
        ids=ids,
    )

    return collection, embedder

def detect_contract(query: str) -> str | None:
    """
    Very simple intent detection.
    Returns: 'murabaha', 'ijara', 'sukuk', or None if not detected.
    """
    q = query.lower()
    for contract in ["murabaha", "ijara", "sukuk"]:
        if contract in q:
            return contract
    return None

def test_query(collection, embedder):
    """
    Ask a test question and retrieve top chunks.
    """
    query = input("\nAsk a question: ").strip()

    contract = detect_contract(query)

    q_emb = embedder.encode([query]).tolist()
    
    k = 3 #number of chunks given to the model
    print(f"\n🔎 Retrieving Top {k} chunks")

    query_kwargs = {
        "query_embeddings": q_emb,
        "n_results": k,
    }

    if contract:
        query_kwargs["where"] = {"contract": contract}
        print(f"\n🎯 Detected contract: {contract} (filter applied)")
    else:
        print("\nℹ️ No contract detected (no filter applied)")

    results = collection.query(**query_kwargs)

    print("\n✅ Top retrieved chunks:\n")
    for i, doc in enumerate(results["documents"][0]):
        print(f"{i+1}) {doc[:220]}...\n")

    generate_answer(query, results["documents"][0])

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def generate_answer(query: str, retrieved_docs: list[str]):
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    context = "\n\n".join(retrieved_docs)
    prompt = f"""
    You are an Islamic finance assistant.
    Answer the question using ONLY the context below.
    If the answer cannot be found in the context, reply exactly with:
    I don't know based on the provided context.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        min_new_tokens=20,      # prevents 1–3 word answers
        num_beams=4,            # improves quality
        do_sample=False,        # deterministic
        early_stopping=True,
        no_repeat_ngram_size=3,#reduces weird loops
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    # remove accidental newlines
    answer = answer.replace("\n", " ")

    print("\n✅ FINAL ANSWER:\n")
    print(answer)
 
def evaluate_system(collection, embedder):
    print("\n📊 Running Evaluation...\n")

    for i, test in enumerate(test_cases):
        question = test["question"]
        expected = test["expected"]

        print(f"\nTest {i+1}")
        print(f"Q: {question}")

        contract = detect_contract(question)

        if contract is None:
            answer = "I don't know based on the provided context."
        else:
            q_emb = embedder.encode([question]).tolist()

            results = collection.query(
                query_embeddings=q_emb,
                n_results=2,
                where={"contract": contract}
            )

            retrieved_docs = results["documents"][0]
            answer = generate_answer_eval(question, retrieved_docs)

        print(f"Expected: {expected}")
        print(f"Got: {answer}")
        print("-" * 40)   

def generate_answer_eval(query: str, retrieved_docs: list[str]):
    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    context = "\n\n".join(retrieved_docs)

    prompt = f"""
    You are an Islamic finance assistant.
    Answer using ONLY the context.

    If the answer is not in the context, say:
    I don't know based on the provided context.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        min_new_tokens=20,
        num_beams=4,
        do_sample=False
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()


def main():
    texts = load_text_files(DATA_DIR)
    all_chunks = []

    for name, content in texts.items():
        file_chunks = chunk_text(content, name)
        all_chunks.extend(file_chunks)

    print("\n📦 Total structured chunks:", len(all_chunks))

    print("✅ Loaded files:")
    for name, content in texts.items():
        print(f" - {name} ({len(content)} chars)")

    combined = "\n\n".join([f"[SOURCE: {name}]\n{content}" for name, content in texts.items()])

       # --- Chunk the text ---


    # print("\n📦 Total chunks created:", len(chunks))
    # print("\n--- Sample chunk ---")
    # print(chunks[0][:500])
    # print("\nChunk length:", len(chunks[0]))
    # --- Build vector store ---
    collection, embedder = build_vector_store(all_chunks)

    # --- Test retrieval ---
    # Interactive mode
    # test_query(collection, embedder)

    # Evaluation mode
    evaluate_system(collection, embedder)




if __name__ == "__main__":
    main()