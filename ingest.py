import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

# ---- Step 1: Load the PDF and extract text ----
def load_pdf_text(filepath):
    reader = PdfReader(filepath)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text

# ---- Step 2: Chunk the text into small overlapping pieces ----
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # overlap keeps context continuity between chunks
    return chunks

# ---- Step 3: Embed chunks and store in ChromaDB ----
def build_vector_store(chunks, collection_name="brsr_docs"):
    print("Loading embedding model (first run downloads it, ~90MB)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name=collection_name)

    print(f"Embedding {len(chunks)} chunks...")
    embeddings = embedder.encode(chunks).tolist()

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB.")

if __name__ == "__main__":
    pdf_path = "data/sample.pdf"
    text = load_pdf_text(pdf_path)
    print(f"Extracted {len(text)} characters from PDF.")

    chunks = chunk_text(text)
    print(f"Split into {len(chunks)} chunks.")

    build_vector_store(chunks)