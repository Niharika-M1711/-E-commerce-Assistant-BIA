"""
retriever.py
Loads FAQ.md and Policies.md, splits them into small Q&A / section chunks,
and finds the most relevant chunk(s) for a user question using TF-IDF cosine similarity.

Why TF-IDF instead of embeddings?
- The knowledge base is tiny (~20 chunks). A full vector DB / embedding API is overkill,
  adds cost + complexity, and TF-IDF matches keyword-heavy support questions just as well.
"""

import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def chunk_faq(text: str):
    """Split FAQ.md into one chunk per Q&A pair."""
    chunks = []
    blocks = re.split(r"\n(?=\*\*Q\d+\.\d+:)", text)
    for block in blocks:
        block = block.strip()
        if block.startswith("**Q"):
            # Strip anything after a "---" divider or "## " next-heading that
            # bleeds into the last Q&A of a section
            block = re.split(r"\n-{3,}|\n## ", block)[0]
            clean = block.replace("**", "")
            chunks.append(clean.strip())
    return chunks


def chunk_policies(text: str):
    """Split Policies.md into one chunk per numbered section."""
    chunks = []
    blocks = re.split(r"\n(?=## \d+\.)", text)
    for block in blocks:
        block = block.strip()
        if block.startswith("## "):
            clean = block.replace("#", "").strip()
            chunks.append(clean)
    return chunks


class KnowledgeBase:
    def __init__(self, data_dir="data"):
        with open(os.path.join(data_dir, "FAQ.md"), encoding="utf-8") as f:
            faq_text = f.read()
        with open(os.path.join(data_dir, "Policies.md"), encoding="utf-8") as f:
            policy_text = f.read()

        self.chunks = chunk_faq(faq_text) + chunk_policies(policy_text)

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.chunks)

    def search(self, query: str, top_k: int = 2, min_score: float = 0.08):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        ranked_idx = scores.argsort()[::-1][:top_k]
        results = [
            (self.chunks[i], float(scores[i]))
            for i in ranked_idx
            if scores[i] >= min_score
        ]
        return results


if __name__ == "__main__":
    kb = KnowledgeBase(data_dir="data")
    print(f"Loaded {len(kb.chunks)} knowledge chunks total.\n")

    test_queries = [
        "How do I return a product?",
        "My payment failed but money was deducted",
        "Do you charge for shipping?",
    ]

    for q in test_queries:
        print(f"QUERY: {q}")
        results = kb.search(q)
        for chunk, score in results:
            print(f"  [score={score:.3f}] {chunk[:120]}...")
        print()