import numpy as np


def softmax(x, axis=-1):
    x = np.asarray(x, dtype=float)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V):
    Q = np.asarray(Q, dtype=float)
    K = np.asarray(K, dtype=float)
    V = np.asarray(V, dtype=float)
    if Q.ndim != 2 or K.ndim != 2 or V.ndim != 2:
        raise ValueError("Q, K va V phải là ma trận hai chiều")
    if Q.shape[1] != K.shape[1]:
        raise ValueError("Q và K phải có cùng số chiều đặc trưng")
    if K.shape[0] != V.shape[0]:
        raise ValueError("K và V phải có cùng số token")
    scores = Q @ K.T / np.sqrt(K.shape[1])
    attention_weights = softmax(scores, axis=-1)
    context = attention_weights @ V
    return scores, attention_weights, context


def self_attention(X, WQ=None, WK=None, WV=None):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X phải là ma trận hai chiều")
    dim = X.shape[1]
    WQ = np.eye(dim) if WQ is None else np.asarray(WQ, dtype=float)
    WK = np.eye(dim) if WK is None else np.asarray(WK, dtype=float)
    WV = np.eye(dim) if WV is None else np.asarray(WV, dtype=float)
    if WQ.shape[0] != dim or WK.shape[0] != dim or WV.shape[0] != dim:
        raise ValueError("Các ma trận chiếu phải nhận đầu vào có cùng số chiều với X")
    Q = X @ WQ
    K = X @ WK
    V = X @ WV
    scores, weights, context = scaled_dot_product_attention(Q, K, V)
    return Q, K, V, scores, weights, context
