from __future__ import annotations
import random
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
def tokenize(sentence: str) -> List[str]:
    return sentence.lower().split()

def build_vocabulary(
    sentences: Sequence[str],
    min_count: int = 1,
    stopwords: Sequence[str] = (),
    fixed_words: Optional[Sequence[str]] = None,
) -> Dict[str, int]:
    if fixed_words is not None:
        return {w: i for i, w in enumerate(fixed_words)}
    stop = set(stopwords)
    counts = Counter(w for s in sentences for w in tokenize(s) if w not in stop)
    words = sorted(w for w, c in counts.items() if c >= min_count)
    return {w: i for i, w in enumerate(words)}

def build_cooccurrence_matrix(
    sentences: Sequence[str],
    vocabulary: Dict[str, int],
    window: int = 1,
) -> np.ndarray:
    n = len(vocabulary)
    X = np.zeros((n, n), dtype=np.float64)
    for sentence in sentences:
        tokens = tokenize(sentence)
        for i, w in enumerate(tokens):
            if w not in vocabulary:
                continue
            lo, hi = max(0, i - window), min(len(tokens), i + window + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                c = tokens[j]
                if c in vocabulary:
                    X[vocabulary[w], vocabulary[c]] += 1
    return X

def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))

def most_similar(
    word: str,
    matrix: np.ndarray,
    vocabulary: Dict[str, int],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    if word not in vocabulary:
        raise KeyError(f"'{word}' not in vocab")
    inv = {i: w for w, i in vocabulary.items()}
    target = matrix[vocabulary[word]]
    scores = [
        (inv[i], cosine_similarity(target, matrix[i]))
        for i in range(matrix.shape[0])
        if i != vocabulary[word]
    ]
    scores.sort(key=lambda x: (-x[1], x[0]))
    return scores[:top_k]

def matrix_stats(matrix: np.ndarray) -> dict:
    nnz = int(np.count_nonzero(matrix))
    return {
        "vocab_size": matrix.shape[0],
        "matrix_shape": f"{matrix.shape[0]}x{matrix.shape[1]}",
        "n_entries": int(matrix.size),
        "nonzero": nnz,
        "density": nnz / matrix.size,
    }

def cbow_pairs(tokens: Sequence[str], window: int = 1):
    out = []
    for i, target in enumerate(tokens):
        ctx = [tokens[j] for j in range(max(0, i - window), min(len(tokens), i + window + 1)) if j != i]
        out.append((ctx, target))
    return out

def skipgram_pairs(tokens: Sequence[str], window: int = 1):
    out = []
    for i, target in enumerate(tokens):
        for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
            if j != i:
                out.append((target, tokens[j]))
    return out

def ppmi(matrix: np.ndarray) -> np.ndarray:
    total = matrix.sum()
    pw = matrix.sum(axis=1, keepdims=True) / total
    pc = matrix.sum(axis=0, keepdims=True) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((matrix / total) / (pw * pc))
    pmi[~np.isfinite(pmi)] = 0.0
    return np.maximum(pmi, 0.0)

def svd_embeddings(matrix: np.ndarray, dim: int = 20) -> np.ndarray:
    U, S, _ = np.linalg.svd(ppmi(matrix), full_matrices=False)
    return U[:, :dim] * np.sqrt(S[:dim])
    print("most_similar('doctor'):", most_similar("doctor", X, vocab, top_k=5))
