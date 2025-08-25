
"""Benchmark weighted-average vs summary embedding search."""
import numpy as np


def embed(text: str, dims: int = 64) -> np.ndarray:
    tokens = text.lower().split()
    vec = np.zeros(dims, dtype=np.float32)
    for tok in tokens:
        vec[hash(tok) % dims] += 1.0
    return vec

SENTENCES = [
    "The cat sat on the mat and purred softly.",
    "A recipe for apple pie includes apples, sugar, and crust.",
    "Machine learning enables computers to learn from data.",
]
QUERIES = [
    "Where does a cat rest?",
    "How do you bake a pie?",
    "What allows computers to learn from examples?",
]

def benchmark():
    e_chunks = [embed(s) for s in SENTENCES]
    e_summary = [embed(s) for s in SENTENCES]
    e_weighted = [e for e in e_chunks]  # single chunk per sentence
    correct_summary = 0
    correct_weighted = 0
    for q_idx, query in enumerate(QUERIES):
        e_q = embed(query)
        sims_summary = [float(np.dot(e_q, e_s)) for e_s in e_summary]
        sims_weighted = [float(np.dot(e_q, e_w)) for e_w in e_weighted]
        if int(np.argmax(sims_summary)) == q_idx:
            correct_summary += 1
        if int(np.argmax(sims_weighted)) == q_idx:
            correct_weighted += 1
    print('summary correct', correct_summary)
    print('weighted correct', correct_weighted)

if __name__ == '__main__':
    benchmark()
