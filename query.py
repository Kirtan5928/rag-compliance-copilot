import os
from dotenv import load_dotenv
from hybrid_retrieval import hybrid_search
import ollama

load_dotenv()

# Controls which LLM backend generates answers.
# "ollama" = local, free, no internet needed, used for development.
# "groq"   = hosted, free-tier API, used for deployment where Ollama can't run.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

def generate_answer(question, retrieved):
    # Build context with labeled sources so the LLM can reference them
    labeled_context = "\n\n".join(
        [f"[Source: {chunk_id}]\n{text}" for chunk_id, text, _ in retrieved]
    )

    prompt = f"""You are a compliance assistant answering questions about a company's BRSR/ESG report.
Use ONLY the context below to answer. Do not use outside knowledge, do not guess, and do not estimate.
If the context does not contain the answer, respond exactly with: "This information is not available in the provided report context."
Do not include source labels or citations in your answer -- the verified source will be shown separately.

Context:
{labeled_context}

Question: {question}

Answer:"""

    if LLM_PROVIDER == "groq":
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content

    else:  # default: local Ollama
        response = ollama.chat(
            model="llama3:latest",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0}
        )
        return response["message"]["content"]

if __name__ == "__main__":
    question = input("Ask a question about the report: ")

    print("\nRunning hybrid retrieval (vector + BM25)...")
    all_retrieved = hybrid_search(question, n_results=5, alpha=0.5)

    # Filter out weakly-relevant chunks instead of forcing a fixed count
    hybrid_threshold = 0.8
    retrieved = [r for r in all_retrieved if r[2] >= hybrid_threshold]
    if not retrieved:
        retrieved = all_retrieved[:1]  # fallback: at least the single best match

    print("Generating answer...\n")
    answer = generate_answer(question, retrieved)

    print("=" * 50)
    print("ANSWER:")
    print(answer)
    print("=" * 50)
    print("\nRETRIEVED SOURCES (hybrid score, for verification):")
    for chunk_id, text, score in retrieved:
        print(f"\n[{chunk_id}] (hybrid score: {score:.3f})")
        print(text[:200] + "..." if len(text) > 200 else text)
    print("=" * 50)