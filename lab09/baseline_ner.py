import argparse
from pathlib import Path
import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from ner_utils import ROOT, build_records, flatten_features, load_conll2003, metric_counts, save_csv, token_features


def train_baseline(train_dataset, output_path=None):
    features, labels = flatten_features(train_dataset)
    vectorizer = DictVectorizer(sparse=True)
    features_matrix = vectorizer.fit_transform(features)
    classifier = LogisticRegression(max_iter=300, solver="lbfgs", random_state=42)
    classifier.fit(features_matrix, labels)
    result = {"vectorizer": vectorizer, "classifier": classifier}
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(result, output_path)
    return result


def predict_baseline(model, dataset):
    all_features = []
    lengths = []
    for example in dataset:
        words = example["tokens"]
        lengths.append(len(words))
        all_features.extend(token_features(words, i) for i in range(len(words)))
    flat_predictions = model["classifier"].predict(model["vectorizer"].transform(all_features))
    result = []
    offset = 0
    for count in lengths:
        result.append(flat_predictions[offset:offset + count].tolist())
        offset += count
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "models" / "baseline.joblib"))
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    args = parser.parse_args()
    dataset = load_conll2003()
    train = dataset["train"] if args.max_train is None else dataset["train"].select(range(min(args.max_train, len(dataset["train"]))))
    test = dataset["test"] if args.max_test is None else dataset["test"].select(range(min(args.max_test, len(dataset["test"]))))
    model = train_baseline(train, args.model)
    records = build_records(test, predict_baseline(model, test))
    scores = metric_counts(records)
    save_csv(ROOT / "artifacts" / "baseline_only_metrics.csv",
             ["Model", "Precision", "Recall", "F1", "TP", "FP", "FN", "Sentences", "Split"],
             [{"Model": "Baseline", **scores, "Sentences": len(records), "Split": "test"}])


if __name__ == "__main__":
    main()
