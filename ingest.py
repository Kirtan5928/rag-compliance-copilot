from pypdf import PdfReader
from embeddings import embed_texts
from vectorstore import store_chunks

def extract_pdf_pages(filepath):
    reader = PdfReader(filepath)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((page_number, text))
    return pages

def load_pdf_text(filepath):
    return "\n".join(text for _, text in extract_pdf_pages(filepath))

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    step = max(1, chunk_size - overlap)
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_size]))
        start += step
    return chunks

def chunk_pages(pages, chunk_size=500, overlap=50):
    chunks = []
    for page_number, text in pages:
        for index, chunk in enumerate(chunk_text(text, chunk_size, overlap)):
            chunks.append({"text": chunk, "page_start": page_number, "page_end": page_number, "chunk_index": index})
    return chunks

def build_vector_store(chunks, document_id, document_name, collection_name="brsr_docs"):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts)
    store_chunks(chunks, embeddings, document_id, document_name, collection_name)
    print(f"Stored {len(chunks)} chunks in Qdrant.")

if __name__ == "__main__":
    pages = extract_pdf_pages("data/sample.pdf")
    build_vector_store(chunk_pages(pages), "sample", "sample.pdf")
