import argparse
import os
import random
import re
import time

import matplotlib
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer, BertConfig, BertModel, get_linear_schedule_with_warmup

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPERIMENTS = ["pretrained_vs_scratch", "frozen_vs_finetune", "data", "length"]


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="distilbert-base-uncased")
    p.add_argument("--train_size", type=int, default=4000)
    p.add_argument("--val_size", type=int, default=500)
    p.add_argument("--test_size", type=int, default=2000)
    p.add_argument("--ood_size", type=int, default=2000)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--lengths", type=int, nargs="+", default=[64, 128, 256])
    p.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr_ft", type=float, default=3e-5)
    p.add_argument("--lr_frozen", type=float, default=1e-3)
    p.add_argument("--lr_scratch", type=float, default=3e-4)
    p.add_argument("--scratch_layers", type=int, default=2)
    p.add_argument("--scratch_hidden", type=int, default=256)
    p.add_argument("--scratch_heads", type=int, default=4)
    p.add_argument("--scratch_ff", type=int, default=1024)
    p.add_argument("--low_threshold", type=float, default=0.7)
    p.add_argument("--high_threshold", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exps", nargs="+", default=EXPERIMENTS, choices=EXPERIMENTS)
    p.add_argument("--out_dir", default=".")
    return p.parse_args(argv)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def hardware_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


class Data:
    def __init__(self, args, tokenizer):
        imdb = load_dataset("stanfordnlp/imdb")
        pool = imdb["train"].shuffle(seed=args.seed)
        val = pool.select(range(args.val_size))
        train = pool.select(range(args.val_size, args.val_size + args.train_size))
        test = imdb["test"].shuffle(seed=args.seed).select(range(args.test_size))
        ood = load_dataset("fancyzhx/amazon_polarity", split="test").shuffle(seed=args.seed).select(range(args.ood_size))
        self.raw = {
            "train": (train["text"], train["label"]),
            "val": (val["text"], val["label"]),
            "test": (test["text"], test["label"]),
            "ood": (ood["content"], ood["label"]),
        }
        self.tokenizer = tokenizer
        self.cache = {}

    def get(self, split, max_len):
        key = (split, max_len)
        if key not in self.cache:
            texts, labels = self.raw[split]
            enc = self.tokenizer(list(texts), padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
            self.cache[key] = (enc["input_ids"], enc["attention_mask"], torch.tensor(labels))
        return self.cache[key]

    def train_subset(self, frac, max_len):
        ids, mask, y = self.get("train", max_len)
        n = int(len(y) * frac)
        return ids[:n], mask[:n], y[:n]


class Classifier(nn.Module):
    def __init__(self, encoder, hidden, frozen, n_classes=2, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.frozen = frozen
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_classes)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, input_ids, attention_mask):
        if self.frozen:
            with torch.no_grad():
                h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        else:
            h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(self.dropout(h[:, 0]))


def build_model(kind, freeze, args, tokenizer):
    if kind == "pretrained":
        encoder = AutoModel.from_pretrained(args.model_name)
        hidden = encoder.config.hidden_size
        if freeze:
            for param in encoder.parameters():
                param.requires_grad = False
    else:
        config = BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=args.scratch_hidden,
            num_hidden_layers=args.scratch_layers,
            num_attention_heads=args.scratch_heads,
            intermediate_size=args.scratch_ff,
            max_position_embeddings=512,
        )
        encoder = BertModel(config, add_pooling_layer=False)
        hidden = args.scratch_hidden
    return Classifier(encoder, hidden, frozen=freeze)


@torch.no_grad()
def predict(model, ids, mask, batch_size, device):
    model.eval()
    out = []
    for i in range(0, len(ids), batch_size):
        logits = model(ids[i:i + batch_size].to(device), mask[i:i + batch_size].to(device))
        out.append(torch.softmax(logits, dim=-1).cpu())
    return torch.cat(out).numpy()


def compute_metrics(probs, y):
    pred = probs.argmax(1)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {"accuracy": accuracy_score(y, pred), "precision": precision, "recall": recall, "f1": f1}


def train_run(kind, freeze, frac, max_len, data, tokenizer, args, device):
    set_seed(args.seed)
    model = build_model(kind, freeze, args, tokenizer).to(device)
    if freeze:
        lr = args.lr_frozen
    elif kind == "scratch":
        lr = args.lr_scratch
    else:
        lr = args.lr_ft
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    ids, mask, y = data.train_subset(frac, max_len)
    loader = DataLoader(TensorDataset(ids, mask, y), batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    val_ids, val_mask, val_y = data.get("val", max_len)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    best_acc, best_state, train_time = -1.0, None, 0.0
    for _ in range(args.epochs):
        model.train()
        start = time.perf_counter()
        for bi, bm, by in loader:
            logits = model(bi.to(device), bm.to(device))
            loss = F.cross_entropy(logits, by.to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_time += time.perf_counter() - start
        val_acc = compute_metrics(predict(model, val_ids, val_mask, args.batch_size, device), val_y.numpy())["accuracy"]
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                          if k in trainable}
    model.load_state_dict(best_state, strict=False)
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2 ** 20
    else:
        peak = psutil.Process().memory_info().rss / 2 ** 20
    result = {
        "train_time_s": train_time,
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "peak_mem_mb": peak,
    }
    for split in ("test", "ood"):
        s_ids, s_mask, s_y = data.get(split, max_len)
        probs = predict(model, s_ids, s_mask, args.batch_size, device)
        result[split] = {"probs": probs, "labels": s_y.numpy(), "metrics": compute_metrics(probs, s_y.numpy())}
    return result


def make_plan(args):
    mt = args.max_len
    configs = [("pretrained", False, 1.0, mt)]

    def add(c):
        if c not in configs:
            configs.append(c)

    exps = set(args.exps)
    if "pretrained_vs_scratch" in exps:
        add(("scratch", False, 1.0, mt))
    if "frozen_vs_finetune" in exps:
        add(("pretrained", True, 1.0, mt))
    if "data" in exps:
        for frac in args.fractions:
            add(("pretrained", False, frac, mt))
            add(("scratch", False, frac, mt))
    if "length" in exps:
        for length in args.lengths:
            add(("pretrained", False, 1.0, length))
    return configs


def plot_data_efficiency(df, path):
    sub = df[(df.eval_set == "imdb_test") & df.encoder_trainable & (df.max_len == df.max_len.mode()[0])]
    fig, ax = plt.subplots(figsize=(6, 4))
    for model_type, group in sub.groupby("model_type"):
        group = group.sort_values("train_frac")
        x = group.train_frac * 100
        ax.plot(x, group.accuracy, marker="o", label=f"{model_type} accuracy")
        ax.plot(x, group.f1, marker="s", linestyle="--", label=f"{model_type} F1")
    ax.set_xscale("log")
    ticks = sorted(sub.train_frac.unique() * 100)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}%" for t in ticks])
    ax.set_xlabel("Training data")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(probs, labels, path):
    cm = confusion_matrix(labels, probs.argmax(1))
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["negative", "positive"])
    ax.set_yticklabels(["negative", "positive"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confidence(probs, labels, args, path):
    pred = probs.argmax(1)
    conf = probs.max(1)
    correct = pred == labels
    bins = {
        "low": conf < args.low_threshold,
        "medium": (conf >= args.low_threshold) & (conf < args.high_threshold),
        "high": conf >= args.high_threshold,
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    names = list(bins)
    accs = [correct[m].mean() if m.sum() else 0.0 for m in bins.values()]
    counts = [int(m.sum()) for m in bins.values()]
    axes[0].bar(names, accs)
    for i, (a, n) in enumerate(zip(accs, counts)):
        axes[0].text(i, a, f"n={n}", ha="center", va="bottom")
    axes[0].set_ylim(0, 1.1)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xlabel("Confidence level")
    axes[1].hist([conf[correct], conf[~correct]], bins=20, range=(0.5, 1.0), label=["correct", "wrong"])
    axes[1].set_xlabel("Confidence")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def clean_text(text, limit=200):
    text = re.sub(r"<br\s*/?>", " ", text)
    text = text.replace("|", "/").replace("\n", " ").strip()
    return text[:limit]


def select_errors(probs, labels):
    pred = probs.argmax(1)
    conf = probs.max(1)
    order = np.argsort(-conf)
    fp = [i for i in order if pred[i] == 1 and labels[i] == 0][:3]
    fn = [i for i in order if pred[i] == 0 and labels[i] == 1][:3]
    chosen = set(fp + fn)
    confident = [i for i in order if pred[i] != labels[i] and i not in chosen][:2]
    return fp + fn + confident


def write_error_samples(probs, labels, texts, out_dir):
    pred = probs.argmax(1)
    conf = probs.max(1)
    indices = select_errors(probs, labels)
    names = ["negative", "positive"]
    rows = [{"text": clean_text(texts[i]), "true": names[labels[i]], "pred": names[pred[i]],
             "confidence": round(float(conf[i]), 4)} for i in indices]
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "error_samples.csv"), index=False)
    lines = ["| Text | True | Pred | Confidence | Error type | Hypothesis |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['text']} | {r['true']} | {r['pred']} | {r['confidence']:.4f} |  |  |")
    path = os.path.join(out_dir, "error_analysis.md")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            content = f.read()
        table = "\n".join(lines)
        content = re.sub(r"(<!-- table:start -->).*?(<!-- table:end -->)",
                         lambda m: f"{m.group(1)}\n{table}\n{m.group(2)}", content, flags=re.S)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)


def run_all(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    data = Data(args, tokenizer)
    hardware = hardware_name()
    runs = {}
    rows = []
    for config in make_plan(args):
        kind, freeze, frac, max_len = config
        result = train_run(kind, freeze, frac, max_len, data, tokenizer, args, device)
        runs[config] = result
        print(kind, "frozen" if freeze else "trainable", frac, max_len, round(result["test"]["metrics"]["accuracy"], 4))
        for split, name in (("test", "imdb_test"), ("ood", "amazon_ood")):
            rows.append({
                "model_name": args.model_name if kind == "pretrained" else "scratch_transformer",
                "model_type": kind,
                "encoder_trainable": not freeze,
                "train_frac": frac,
                "max_len": max_len,
                "eval_set": name,
                **result[split]["metrics"],
                "train_time_s": result["train_time_s"],
                "total_params": result["total_params"],
                "trainable_params": result["trainable_params"],
                "peak_mem_mb": result["peak_mem_mb"],
                "seed": args.seed,
                "hardware": hardware,
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "results.csv"), index=False)
    main = runs[("pretrained", False, 1.0, args.max_len)]["test"]
    plot_confusion(main["probs"], main["labels"], os.path.join(args.out_dir, "confusion_matrix.png"))
    plot_confidence(main["probs"], main["labels"], args, os.path.join(args.out_dir, "confidence_analysis.png"))
    write_error_samples(main["probs"], main["labels"], data.raw["test"][0], args.out_dir)
    if "data" in args.exps:
        plot_data_efficiency(df, os.path.join(args.out_dir, "data_efficiency.png"))
    return df


if __name__ == "__main__":
    run_all(parse_args())
