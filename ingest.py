import os
from pypdf import PdfReader
from embeddings import embed_texts
from vectorstore import store_chunks

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
        start += chunk_size - overlap
    return chunks

# ---- Step 3: Embed chunks and store in Qdrant ----
def build_vector_store(chunks, collection_name="brsr_docs"):
    print(f"Embedding {len(chunks)} chunks...")
    embeddings = embed_texts(chunks)

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    store_chunks(ids, chunks, embeddings, collection_name)
    print(f"Stored {len(chunks)} chunks in Qdrant.")

if __name__ == "__main__":
    pdf_path = "data/sample.pdf"
    text = load_pdf_text(pdf_path)
    print(f"Extracted {len(text)} characters from PDF.")

    chunks = chunk_text(text)
    print(f"Split into {len(chunks)} chunks.")

    build_vector_store(chunks)