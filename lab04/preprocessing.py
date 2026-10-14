import html
import re
import unicodedata

import numpy as np
import pandas as pd
from gensim.utils import simple_preprocess
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


def normalize_text(text):
    if pd.isna(text):
        return ""
    text = html.unescape(str(text))
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def inspect_split(frame, split_name):
    text = frame["text"].fillna("").astype(str)
    normalized = text.map(normalize_text)
    lengths = text.map(lambda value: len(simple_preprocess(value, deacc=True)))
    return {
        "split": split_name,
        "samples": int(len(frame)),
        "classes": int(frame["label"].nunique(dropna=True)),
        "missing_text": int(frame["text"].isna().sum()),
        "missing_labels": int(frame["label"].isna().sum()),
        "duplicate_documents": int(normalized.duplicated().sum()),
        "mean_words": round(float(lengths.mean()), 2) if len(lengths) else 0.0,
        "median_words": round(float(lengths.median()), 2) if len(lengths) else 0.0,
        "p95_words": round(float(lengths.quantile(0.95)), 2) if len(lengths) else 0.0,
    }


def stratified_sample(frame, sample_size, random_state=42):
    if sample_size is None or sample_size >= len(frame):
        return frame.sample(frac=1, random_state=random_state).reset_index(drop=True)
    class_counts = frame["label"].value_counts()
    if class_counts.min() < 2:
        return frame.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    if sample_size < frame["label"].nunique():
        raise ValueError("sample_size must be at least the number of classes")
    grouped = []
    remaining = sample_size
    classes = list(class_counts.index)
    for position, label in enumerate(classes):
        class_frame = frame[frame["label"] == label]
        if position == len(classes) - 1:
            take = min(remaining, len(class_frame))
        else:
            take = min(len(class_frame), round(sample_size * len(class_frame) / len(frame)))
            remaining -= take
        grouped.append(class_frame.sample(n=take, random_state=random_state + position))
    sampled = pd.concat(grouped, axis=0)
    if len(sampled) < sample_size:
        unused = frame.drop(index=sampled.index, errors="ignore")
        addition = unused.sample(n=min(sample_size - len(sampled), len(unused)), random_state=random_state)
        sampled = pd.concat([sampled, addition], axis=0)
    return sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)


def remove_test_overlap(train_frame, test_frame):
    train_texts = set(train_frame["text"].map(normalize_text))
    normalized_test = test_frame["text"].map(normalize_text)
    keep = ~normalized_test.isin(train_texts)
    removed = int((~keep).sum())
    return test_frame.loc[keep].reset_index(drop=True), removed


def average_word2vec(texts, word2vec_model, vector_size=100):
    matrix = np.zeros((len(texts), vector_size), dtype=np.float32)
    keyed_vectors = word2vec_model.wv
    for row_index, text in enumerate(texts):
        tokens = simple_preprocess(normalize_text(text), deacc=True)
        vectors = [keyed_vectors[token] for token in tokens if token in keyed_vectors]
        if vectors:
            matrix[row_index] = np.mean(vectors, axis=0)
    return matrix


def binary_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }
