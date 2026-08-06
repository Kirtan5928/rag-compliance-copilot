"""
Evaluation harness for the RAG pipeline.
Runs a mixed batch of questions -- answerable, broad/multi-hop, and
deliberately OUT OF SCOPE (not in the document) -- and logs how the
system behaves on each category. This is what proves the system isn't
just tuned to easy cases: we WANT to see it fail gracefully on the
out-of-scope questions (say "not found") rather than fabricate an answer.
"""

from hybrid_retrieval import hybrid_search
from query import generate_answer

# Label each question by type so results are easy to read and reason about.
TEST_QUESTIONS = [
    ("ANSWERABLE - numeric", "What is the company's total energy consumption?"),
    ("ANSWERABLE - numeric", "What is the total number of employees and workers in the company?"),
    ("ANSWERABLE - qualitative", "Does the company have a policy on employee health and safety?"),
    ("ANSWERABLE - broad/multi-hop", "Summarize the company's environmental sustainability initiatives."),
    ("ANSWERABLE - broad/multi-hop", "What steps has the company taken to reduce its carbon footprint?"),

    # These should NOT have a good answer in a BRSR/ESG report.
    # A grounded system should say "not found in context" -- NOT fabricate a plausible-sounding answer.
    ("OUT OF SCOPE - unrelated", "What is the company's stock price target for next year?"),
    ("OUT OF SCOPE - unrelated", "Who is the company's Chief Marketing Officer?"),
    ("OUT OF SCOPE - fabrication trap", "What is the company's total revenue from cryptocurrency mining operations?"),
]

def run_evaluation(hybrid_threshold=0.8):
    for category, question in TEST_QUESTIONS:
        print("\n" + "=" * 70)
        print(f"[{category}]")
        print(f"Q: {question}")
        print("-" * 70)

        all_retrieved = hybrid_search(question, n_results=5, alpha=0.5)
        retrieved = [r for r in all_retrieved if r[2] >= hybrid_threshold]

        # Show the FULL score spread, not just what passed -- this is what
        # makes the eval honest instead of hiding the threshold's effect.
        print("Top-5 raw scores:", [round(r[2], 3) for r in all_retrieved])
        print(f"Chunks passing threshold ({hybrid_threshold}): {len(retrieved)}")

        if not retrieved:
            retrieved = all_retrieved[:1]
            print("-> No chunk cleared threshold, using fallback top-1.")

        answer = generate_answer(question, retrieved)
        print(f"\nANSWER:\n{answer}")

if __name__ == "__main__":
    run_evaluation()