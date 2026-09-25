import json
import os
from dotenv import load_dotenv
import ollama

load_dotenv()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")

UNAVAILABLE = "This information is not available in the provided report context."

SYSTEM_PROMPT = """You are a compliance assistant answering questions about company BRSR/ESG reports.

Use ONLY the supplied report context. Never use outside knowledge, guess, estimate, round, abbreviate, or infer a value that is not explicitly supported by the context.

NUMERICAL ACCURACY IS CRITICAL:
- Preserve numbers exactly as written in the source, including every digit, comma, decimal point, sign, percentage, and unit.
- For table questions, identify the exact row/parameter and the correct financial year before answering.
- Never shorten a large number. For example, do not turn 46,312,39,459 into 46,312.
- Do not perform arithmetic unless the question explicitly asks for a calculation.
- If the context does not explicitly support the requested answer, mark the answer as unavailable.

Return a concise factual answer. Do not invent citations."""
 
def generate_answer(question, retrieved):
    if not retrieved:
        return UNAVAILABLE

    context = "\n\n".join(
        f"[Source: {item['document_name']}, page {item['page_start']}]\n{item['text']}"
        for item in retrieved
    )

    user_prompt = f"""Report context:
{context}

Question: {question}

Answer the question using only the report context."""

    if LLM_PROVIDER == "groq":
        from groq import Groq

        response = Groq(api_key=os.getenv("GROQ_API_KEY")).chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "compliance_answer",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answer": {
                                "type": "string",
                                "description": "Concise answer. Preserve every supported number exactly as written in the context."
                            },
                            "available": {
                                "type": "boolean",
                                "description": "True only when the context explicitly supports the answer."
                            }
                        },
                        "required": ["answer", "available"],
                        "additionalProperties": False,
                    },
                },
            },
        )

        data = json.loads(response.choices[0].message.content or "{}")
        if not data.get("available", False):
            return UNAVAILABLE
        return data.get("answer", UNAVAILABLE).strip() or UNAVAILABLE

    response = ollama.chat(
        model="llama3:latest",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0},
    )
    return response["message"]["content"].strip()
