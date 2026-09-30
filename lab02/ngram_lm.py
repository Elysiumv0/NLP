from __future__ import annotations
import math
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple
BOS, EOS, UNK = "<s>", "</s>", "<unk>"
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())

def build_vocabulary(corpus: Iterable[Sequence[str]], min_count: int = 1) -> Tuple[set, Counter]:
    word_counts: Counter = Counter()
    for sent in corpus:
        word_counts.update(sent)
    vocab = {w for w, c in word_counts.items() if c >= min_count}
    vocab.update({BOS, EOS, UNK})
    return vocab, word_counts

def pad_sentence(sent: Sequence[str], n: int) -> List[str]:
    return [BOS] * (n - 1) + list(sent) + [EOS]

def count_ngrams(corpus: Iterable[Sequence[str]], n: int) -> Counter:
    counts: Counter = Counter()
    for sent in corpus:
        padded = pad_sentence(sent, n)
        for i in range(len(padded) - n + 1):
            counts[tuple(padded[i:i + n])] += 1
    return counts

class NGramLanguageModel:
    def __init__(self, n: int, smoothing: str = "mle", k: float = 1.0, min_count: int = 1):
        self.n = n
        self.smoothing = smoothing
        self.k = k
        self.min_count = min_count
        self.vocab: set = set()
        self.ngram_counts: Counter = Counter()
        self.context_counts: Counter = Counter()
        self._successors: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self.total_tokens = 0  
        self.V = 0  

    def fit(self, corpus: Iterable[Sequence[str]]) -> "NGramLanguageModel":
        corpus = [list(s) for s in corpus]
        self.vocab, _ = build_vocabulary(corpus, self.min_count)
        corpus = [self._map_oov(s) for s in corpus]
        self.ngram_counts = count_ngrams(corpus, self.n)
        self.context_counts = Counter()
        self._successors = defaultdict(Counter)
        for gram, c in self.ngram_counts.items():
            ctx, w = gram[:-1], gram[-1]
            self.context_counts[ctx] += c
            self._successors[ctx][w] += c
        self.total_tokens = sum(self.ngram_counts.values())
        self.V = len(self.vocab) - 1
        return self

    def _map_oov(self, sent: Sequence[str]) -> List[str]:
        return [w if w in self.vocab else UNK for w in sent]

    def _norm_context(self, context: Sequence[str]) -> Tuple[str, ...]:
        m = self.n - 1
        if m == 0:
            return ()
        ctx = self._map_oov(context)[-m:]
        return tuple([BOS] * (m - len(ctx)) + ctx)

    def probability(self, context: Sequence[str], word: str) -> float:
        ctx = self._norm_context(context)
        w = word if word in self.vocab else UNK
        c_hw = self.ngram_counts.get(ctx + (w,), 0)
        c_h = self.total_tokens if self.n == 1 else self.context_counts.get(ctx, 0)
        if self.smoothing == "mle":
            return c_hw / c_h if c_h > 0 else 0.0
        return (c_hw + self.k) / (c_h + self.k * self.V)

    def next_word_distribution(self, context: Sequence[str], top_k: int | None = None) -> List[Tuple[str, float]]:
        ctx = self._norm_context(context)
        if self.smoothing == "mle":
            succ = self._successors.get(ctx if self.n > 1 else (), Counter())
            c_h = self.total_tokens if self.n == 1 else self.context_counts.get(ctx, 0)
            dist = [(w, c / c_h) for w, c in succ.items()] if c_h else []
        else:
            dist = [(w, self.probability(context, w)) for w in self.vocab if w != BOS]
        dist.sort(key=lambda x: (-x[1], x[0]))
        return dist[:top_k] if top_k else dist

    def sentence_log_probability(self, sentence: Sequence[str]) -> float:
        padded = pad_sentence(self._map_oov(sentence), self.n)
        total = 0.0
        for i in range(self.n - 1, len(padded)):
            p = self.probability(padded[max(0, i - self.n + 1):i], padded[i])
            if p <= 0.0:
                return float("-inf")
            total += math.log(p)
        return total

    def sentence_probability(self, sentence: Sequence[str]) -> float:
        lp = self.sentence_log_probability(sentence)
        return 0.0 if lp == float("-inf") else math.exp(lp)

    def perplexity(self, corpus: Iterable[Sequence[str]]) -> float:
        total_lp, n_tokens = 0.0, 0
        for sent in corpus:
            lp = self.sentence_log_probability(sent)
            if lp == float("-inf"):
                return float("inf")
            total_lp += lp
            n_tokens += len(sent) + 1 
        return math.exp(-total_lp / n_tokens) if n_tokens else float("nan")

    def unseen_ngram_rate(self, corpus: Iterable[Sequence[str]]) -> float:
        unseen = total = 0
        for sent in corpus:
            padded = pad_sentence(self._map_oov(sent), self.n)
            for i in range(len(padded) - self.n + 1):
                total += 1
                unseen += self.ngram_counts.get(tuple(padded[i:i + self.n]), 0) == 0
        return unseen / total if total else float("nan")

    def rank_candidates(self, context: Sequence[str], candidates: Dict[str, Sequence[str]],
                        length_normalize: bool = False):
        results = []
        for name, cand in candidates.items():
            ctx = list(context)
            lp = 0.0
            for w in cand:
                p = self.probability(ctx, w)
                if p <= 0:
                    lp = float("-inf")
                    break
                lp += math.log(p)
                ctx.append(w)
            if length_normalize and lp != float("-inf"):
                lp /= len(cand)
            results.append((name, lp))
        return sorted(results, key=lambda x: -x[1])

if __name__ == "__main__":
    toy = [s.split() for s in ["the cat eats fish", "the cat likes fish", "the dog eats meat"]]
    lm = NGramLanguageModel(2).fit(toy)
    print("P(cat|the) =", lm.probability(["the"], "cat"))
    print("P(S) =", lm.sentence_probability(toy[0]))
