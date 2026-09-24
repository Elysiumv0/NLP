from __future__ import annotations
import math
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple
import numpy as np
def tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE)

def build_vocabulary(documents: Sequence[str]) -> Dict[str, int]:
    terms = set()
    for document in documents:
        terms.update(tokenize(document))
    return {term: i for i, term in enumerate(sorted(terms))}

def compute_counts(documents: Sequence[str], vocabulary: Dict[str, int]) -> np.ndarray:
    counts = np.zeros((len(documents), len(vocabulary)), dtype=np.float64)
    for document_index, document in enumerate(documents):
        for term, count in Counter(tokenize(document)).items():
            column_index = vocabulary.get(term)
            if column_index is not None:
                counts[document_index, column_index] = float(count)
    return counts

def compute_tf(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError("counts must be 2D")
    row_sums = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums != 0)

def compute_idf(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError("counts must be 2D")
    n_documents = counts.shape[0]
    if n_documents == 0:
        raise ValueError("empty corpus")
    df = np.count_nonzero(counts > 0, axis=0)
    if np.any(df == 0):
        raise ValueError("vocabulary contains a term with df=0")
    return np.log(n_documents / df)

def compute_tfidf(tf: np.ndarray, idf: np.ndarray) -> np.ndarray:
    tf = np.asarray(tf, dtype=np.float64)
    idf = np.asarray(idf, dtype=np.float64)
    if tf.ndim != 2 or idf.ndim != 1:
        raise ValueError("invalid dimensions")
    if tf.shape[1] != idf.shape[0]:
        raise ValueError("TF width and IDF length differ")
    return tf * idf

def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D")
    if x.shape != y.shape:
        raise ValueError("shape mismatch")
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    if x_norm == 0.0 or y_norm == 0.0:
        return 0.0
    return float(np.dot(x, y) / (x_norm * y_norm))

def transform_query(query: str, vocabulary: Dict[str, int], idf: np.ndarray) -> np.ndarray:
    counts = np.zeros(len(vocabulary), dtype=np.float64)
    for term, count in Counter(tokenize(query)).items():
        column_index = vocabulary.get(term)
        if column_index is not None:
            counts[column_index] = float(count)
    tf = compute_tf(counts.reshape(1, -1))[0]
    return compute_tfidf(tf.reshape(1, -1), idf)[0]

def rank_documents(
    query: str,
    documents: Sequence[str],
    vocabulary: Dict[str, int],
    tfidf_matrix: np.ndarray,
    idf: np.ndarray,
    top_k: int = 5,
) -> List[Tuple[int, float]]:
    query_vector = transform_query(query, vocabulary, idf)
    scores = np.array([cosine_similarity(query_vector, row) for row in tfidf_matrix])
    indices = np.argsort(-scores, kind="stable")[:min(top_k, len(documents))]
    return [(int(i), float(scores[i])) for i in indices]

def test_build_vocabulary() -> None:
    docs = ["cat eats fish", "dog eats fish", "cat likes fish"]
    assert build_vocabulary(docs) == {
        "cat": 0, "dog": 1, "eats": 2, "fish": 3, "likes": 4
    }

def test_compute_counts() -> None:
    docs = ["cat eats fish", "dog eats fish", "cat likes fish"]
    vocab = build_vocabulary(docs)
    actual = compute_counts(docs, vocab)
    expected = np.array([[1,0,1,1,0],[0,1,1,1,0],[1,0,0,1,1]], dtype=float)
    np.testing.assert_array_equal(actual, expected)

def test_compute_tf() -> None:
    counts = np.array([[1,0,1,1,0],[0,1,1,1,0]], dtype=float)
    actual = compute_tf(counts)
    expected = np.array([1/3,0,1/3,1/3,0])
    np.testing.assert_allclose(actual[0], expected, atol=1e-12)
    np.testing.assert_allclose(actual.sum(axis=1), np.ones(2), atol=1e-12)

def test_compute_idf() -> None:
    counts = np.array([[1,0,1,1,0],[0,1,1,1,0],[1,0,0,1,1]], dtype=float)
    actual = compute_idf(counts)
    expected = np.array([math.log(3/2), math.log(3), math.log(3/2), 0, math.log(3)])
    np.testing.assert_allclose(actual, expected, atol=1e-12)

def test_compute_tfidf() -> None:
    tf = np.array([[1/3,0,1/3,1/3,0]], dtype=float)
    idf = np.array([math.log(3/2), math.log(3), math.log(3/2), 0, math.log(3)])
    actual = compute_tfidf(tf, idf)
    expected = np.array([[math.log(3/2)/3,0,math.log(3/2)/3,0,0]])
    np.testing.assert_allclose(actual, expected, atol=1e-12)

def test_cosine_similarity() -> None:
    x = np.array([1.,1.,1.])
    y = np.array([1.,1.,0.])
    assert abs(cosine_similarity(x, y) - 2/math.sqrt(6)) < 1e-12

def test_query_and_ranking() -> None:
    docs = ["cat eats fish", "dog eats fish", "cat likes fish"]
    vocab = build_vocabulary(docs)
    counts = compute_counts(docs, vocab)
    tf = compute_tf(counts)
    idf = compute_idf(counts)
    tfidf = compute_tfidf(tf, idf)
    q = transform_query("cat eats fish", vocab, idf)
    assert q.shape == (len(vocab),)
    ranked = rank_documents("cat eats fish", docs, vocab, tfidf, idf, 3)
    assert len(ranked) == 3
    assert ranked[0][0] == 0

def run_all_tests() -> None:
    test_build_vocabulary()
    test_compute_counts()
    test_compute_tf()
    test_compute_idf()
    test_compute_tfidf()
    test_cosine_similarity()
    test_query_and_ranking()
    print("7/7 unit tests passed.")
  
if __name__ == "__main__":
    run_all_tests()

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
ref_vectorizer = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b")
ref_counts = ref_vectorizer.fit_transform(docs).toarray().astype(float)
ref_features = ref_vectorizer.get_feature_names_out()
ref_vocab = {term:i for i, term in enumerate(ref_features)}
ref_tf = ref_counts / ref_counts.sum(axis=1, keepdims=True)
ref_transformer = TfidfTransformer(norm=None, use_idf=True, smooth_idf=False)
ref_transformer.fit(ref_counts)
ref_idf = ref_transformer.idf_ - 1.0
ref_tfidf = ref_tf * ref_idf
print("Vocabulary equal:", vocab == ref_vocab)
print("Counts equal:", np.array_equal(counts, ref_counts))
print("TF equal:", np.allclose(tf, ref_tf))
print("IDF equal:", np.allclose(idf, ref_idf))
print("TF-IDF equal:", np.allclose(tfidf, ref_tfidf))
