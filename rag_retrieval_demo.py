from sentence_transformers import SentenceTransformer
import chromadb

# 1) Tiny knowledge base (Week 1 demo)
docs = [
    "Murabaha is a cost-plus sale. The seller discloses cost and profit margin.",
    "In murabaha, the seller should own/possess the asset before selling it.",
    "Riba refers to prohibited interest/usury and must be avoided.",
    "Sukuk represent ownership in an underlying asset, usufruct, or project."
]

# 2) Create embeddings
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = embedder.encode(docs).tolist()

# 3) Store in Chroma (vector database)
client = chromadb.Client()
collection = client.create_collection(name="islamic_finance_week1")

collection.add(
    documents=docs,
    embeddings=embeddings,
    ids=[f"doc{i}" for i in range(len(docs))]
)

# 4) Query → retrieve top chunks
query = "What condition must be met before selling an asset in murabaha?"
q_emb = embedder.encode([query]).tolist()

results = collection.query(
    query_embeddings=q_emb,
    n_results=2
)

print("QUERY:", query)
print("\nTOP RESULTS:")
for i, doc in enumerate(results["documents"][0]):
    print(f"{i+1}) {doc}")
