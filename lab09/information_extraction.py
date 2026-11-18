import argparse
import json
import re
from pathlib import Path
import torch
from ner_utils import ID_TO_LABEL, ROOT, grouped_entities, load_conll2003, metric_counts, save_csv


def infer_word_labels(model, tokenizer, words, max_length=256):
    if not words:
        return [], []
    encoded = tokenizer(words, is_split_into_words=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
    word_ids = encoded.word_ids(batch_index=0)
    inputs = {key: value.to(model.device) for key, value in encoded.items()}
    model.eval()
    with torch.inference_mode():
        predicted_ids = model(**inputs).logits.argmax(dim=-1)[0].tolist()
    tags, indices = [], []
    previous = None
    for index, word_id in enumerate(word_ids):
        if word_id is not None and word_id != previous:
            tags.append(model.config.id2label[int(predicted_ids[index])])
            indices.append(word_id)
        previous = word_id
    return tags, indices


def extract_information(text, model, tokenizer, max_length=256):
    words = re.findall(r"\w+(?:[-']\w+)*|[^\w\s]", text, flags=re.UNICODE)
    tags, indices = infer_word_labels(model, tokenizer, words, max_length)
    return grouped_entities([words[i] for i in indices], tags)


def evaluate_information_extraction(model, tokenizer, dataset=None, limit=30, max_length=256):
    dataset = dataset if dataset is not None else load_conll2003()["test"]
    gold_rows, prediction_rows, records = [], [], []
    for index, row in enumerate(dataset.select(range(min(limit, len(dataset))))):
        words = row["tokens"]
        prediction, indices = infer_word_labels(model, tokenizer, words, max_length=max_length)
        trimmed_words = [words[i] for i in indices]
        original_gold = [ID_TO_LABEL[int(row["ner_tags"][i])] for i in indices]
        gold = [tag if tag == "O" or not tag.endswith("MISC") else "O" for tag in original_gold]
        predicted = [tag if tag == "O" or not tag.endswith("MISC") else "O" for tag in prediction]
        gold_rows.append({"id": index, "text": " ".join(trimmed_words), "entities": grouped_entities(trimmed_words, gold)})
        prediction_rows.append({"id": index, "text": " ".join(trimmed_words), "entities": grouped_entities(trimmed_words, predicted)})
        records.append({"id": index, "words": trimmed_words, "gold": gold, "prediction": predicted})
    out = ROOT / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    for file_name, rows in (("ie_gold.jsonl", gold_rows), ("ie_predictions.jsonl", prediction_rows)):
        with (out / file_name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    overall = metric_counts(records)
    save_csv(out / "ie_metrics.csv", ["Scope", "Precision", "Recall", "F1", "TP", "FP", "FN", "Sentences"],
             [{"Scope": "PERSON/ORGANIZATION/LOCATION", **overall, "Sentences": len(records)}])
    return overall


def main():
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(ROOT / "models" / "ner"))
    parser.add_argument("--text", default="Barack Obama visited Paris.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--evaluate", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if args.evaluate:
        result = evaluate_information_extraction(model, tokenizer, limit=args.evaluate,
                                                  max_length=args.max_length)
    else:
        result = extract_information(args.text, model, tokenizer, max_length=args.max_length)
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(serialized + "\n", encoding="utf-8")
    else:
        print(serialized)


if __name__ == "__main__":
    main()
