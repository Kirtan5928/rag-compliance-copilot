import os
from dotenv import load_dotenv
import ollama

load_dotenv()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")

def generate_answer(question, retrieved):
    if not retrieved:
        return "This information is not available in the provided report context."

    context = "\n\n".join(
        f"[Source: {item['document_name']}, page {item['page_start']}]\n{item['text']}"
        for item in retrieved
    )
    prompt = f"""You are a compliance assistant answering questions about company BRSR/ESG reports.
Use ONLY the context below. Do not use outside knowledge, do not guess, and do not estimate.
If the context does not contain the answer, respond exactly with:
"This information is not available in the provided report context."
Answer concisely and factually. Do not invent citations.

Context:
{context}

Question: {question}

Answer:"""

    if LLM_PROVIDER == "groq":
        from groq import Groq
        response = Groq(api_key=os.getenv("GROQ_API_KEY")).chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    response = ollama.chat(
        model="llama3:latest",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    return response["message"]["content"].strip()
