import chromadb
from sentence_transformers import SentenceTransformer
import ollama

def retrieve_chunks(question, n_results=3, collection_name="brsr_docs"):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name=collection_name)

    question_embedding = embedder.encode([question]).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=n_results
    )
    return results["documents"][0]  # list of retrieved chunk texts

def generate_answer(question, retrieved_chunks):
    context = "\n\n---\n\n".join(retrieved_chunks)

    prompt = f"""You are a compliance assistant answering questions about a company's BRSR/ESG report.
Use ONLY the context below to answer. If the answer isn't in the context, say so — do not make things up.

Context:
{context}

Question: {question}

Answer:"""

    response = ollama.chat(
        model="llama3:latest",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

if __name__ == "__main__":
    question = input("Ask a question about the report: ")

    print("\nRetrieving relevant chunks...")
    chunks = retrieve_chunks(question)

    print("Generating answer...\n")
    answer = generate_answer(question, chunks)

    print("=" * 50)
    print("ANSWER:")
    print(answer)
    print("=" * 50)