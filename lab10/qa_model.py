import argparse
import collections
import csv
import json
import random
import re
import string
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

MODEL = 'distilbert-base-uncased'
SEED = 42
ROOT = Path(__file__).resolve().parent
STOP = set('a an the of in on at to is are was were be been for by with and or what which who where when why how did does do from that this it as'.split())


def normalize(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def metrics(prediction, golds):
    p = normalize(prediction).split()
    scores = []
    for gold in golds:
        g = normalize(gold).split()
        overlap = sum((collections.Counter(p) & collections.Counter(g)).values())
        f1 = 2 * overlap / (len(p) + len(g)) if p and g else float(p == g)
        scores.append((float(p == g), f1))
    return max(s[0] for s in scores), max(s[1] for s in scores)


def terms(text):
    return set(re.findall(r'\w+', text.lower())) - STOP


def sentences(text):
    return [m for m in re.finditer(r'[^.!?]+(?:[.!?]+|$)', text) if m.group().strip()]


def baseline(question, context):
    q = terms(question)
    spans = sentences(context)
    if not spans:
        return {'text': '', 'start': 0, 'end': 0}
    s = max(spans, key=lambda m: len(q & terms(m.group())) / max(1, len(q)))
    tokens = list(re.finditer(r"[\w]+(?:[-'][\w]+)*", s.group()))
    candidates = []
    for i in range(len(tokens)):
        for j in range(i, min(i + 5, len(tokens))):
            words = [t.group() for t in tokens[i:j+1]]
            if any(w.lower() in q for w in words) or words[0].lower() in STOP or words[-1].lower() in STOP:
                continue
            score = sum(w[0].isupper() or w.isdigit() for w in words) + 0.3 * len(words)
            if question.lower().startswith(('when', 'how many', 'how much')):
                score += 3 * any(w.isdigit() for w in words)
            a, b = s.start() + tokens[i].start(), s.start() + tokens[j].end()
            candidates.append((score, a, b))
    if candidates:
        _, a, b = max(candidates, key=lambda x: x[0])
    else:
        a, b = s.start(), s.end()
        while a < b and context[a].isspace():
            a += 1
    return {'text': context[a:b], 'start': a, 'end': b}


def align(offsets, sequence_ids, answer_start, answer_end, cls_index=0):
    context_ids = [i for i, s in enumerate(sequence_ids) if s == 1 and offsets[i][1] > offsets[i][0]]
    if not context_ids or answer_end <= answer_start:
        return cls_index, cls_index
    if offsets[context_ids[0]][0] > answer_start or offsets[context_ids[-1]][1] < answer_end:
        return cls_index, cls_index
    start = next((i for i in context_ids if offsets[i][0] <= answer_start < offsets[i][1]), None)
    end = next((i for i in reversed(context_ids) if offsets[i][0] < answer_end <= offsets[i][1]), None)
    if start is None or end is None or start > end:
        return cls_index, cls_index
    return start, end


def features(tokenizer, examples, max_length, stride, labeled=True):
    question_limit = min(64, max_length - stride - tokenizer.num_special_tokens_to_add(pair=True) - 1)
    questions = [x['question'].strip() for x in examples]
    question_ids = tokenizer(questions, add_special_tokens=False)['input_ids']
    questions = [q if len(ids) <= question_limit else tokenizer.decode(ids[:question_limit], skip_special_tokens=True) for q, ids in zip(questions, question_ids)]
    encoded = tokenizer(questions, [x['context'] for x in examples], truncation='only_second', max_length=max_length, stride=stride, padding='max_length', return_overflowing_tokens=True, return_offsets_mapping=True)
    output = []
    for i, source in enumerate(encoded['overflow_to_sample_mapping']):
        row = {k: encoded[k][i] for k in tokenizer.model_input_names if k in encoded}
        offsets = encoded['offset_mapping'][i]
        seq = encoded.sequence_ids(i)
        cls = row['input_ids'].index(tokenizer.cls_token_id)
        row['example_index'] = source
        row['offsets'] = [tuple(o) if s == 1 and o[1] > o[0] else None for o, s in zip(offsets, seq)]
        if labeled:
            ans = examples[source]['answers']
            pos = (cls, cls)
            for a, text in zip(ans['answer_start'], ans['text']):
                pos = align(offsets, seq, a, a + len(text), cls)
                if pos != (cls, cls):
                    break
            row['start_positions'], row['end_positions'] = pos
        output.append(row)
    return output


class QAModel(nn.Module):
    def __init__(self, model_name=MODEL, config=None):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_name) if config is None else AutoModel.from_config(config)
        self.qa_head = nn.Linear(self.backbone.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask, token_type_ids=None, start_positions=None, end_positions=None):
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if token_type_ids is not None:
            inputs['token_type_ids'] = token_type_ids
        hidden = self.backbone(**inputs).last_hidden_state
        start, end = self.qa_head(hidden).unbind(-1)
        start = start.masked_fill(attention_mask == 0, -1e4)
        end = end.masked_fill(attention_mask == 0, -1e4)
        loss = None
        if start_positions is not None:
            loss = (nn.functional.cross_entropy(start, start_positions) + nn.functional.cross_entropy(end, end_positions)) / 2
        return start, end, loss


def collate(rows):
    return {k: torch.tensor([r[k] for r in rows], dtype=torch.long) for k in rows[0] if k not in ('offsets', 'example_index')}


def best_span(start, end, offsets, context, max_answer=30):
    best = None
    for i, a in enumerate(offsets):
        if a is None:
            continue
        for j in range(i, min(i + max_answer, len(offsets))):
            b = offsets[j]
            if b is None:
                break
            score = float(start[i] + end[j])
            if best is None or score > best['score']:
                best = {'text': context[a[0]:b[1]], 'start': a[0], 'end': b[1], 'score': score}
    return best


def evaluate(model, rows, examples, batch_size, device):
    model.eval()
    best = {}
    total_loss = 0.0
    with torch.inference_mode():
        for a in range(0, len(rows), batch_size):
            chunk = rows[a:a+batch_size]
            batch = {k: v.to(device) for k, v in collate(chunk).items()}
            start, end, loss = model(**batch)
            if loss is not None:
                total_loss += loss.item() * len(chunk)
            start, end = start.cpu().tolist(), end.cpu().tolist()
            for r, s, e in zip(chunk, start, end):
                idx = r['example_index']
                pred = best_span(s, e, r['offsets'], examples[idx]['context'])
                if pred is not None and (idx not in best or pred['score'] > best[idx]['score']):
                    best[idx] = pred
    predictions = [best.get(i, {'text': '', 'start': 0, 'end': 0}) for i in range(len(examples))]
    return total_loss / max(1, len(rows)), predictions


def summarize(examples, predictions):
    scores = [metrics(p['text'], e['answers']['text']) for e, p in zip(examples, predictions)]
    if not scores:
        return None, None
    return tuple(100 * sum(s[i] for s in scores) / len(scores) for i in range(2))


def seed_all(seed):
    from transformers import set_seed
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one(tokenizer, train, val, test, max_length, stride, args, device):
    seed_all(args.seed)
    train_rows = features(tokenizer, train, max_length, stride)
    val_rows = features(tokenizer, val, max_length, stride)
    test_rows = features(tokenizer, test, max_length, stride)
    model = QAModel(args.model).to(device)
    loader = DataLoader(train_rows, batch_size=args.batch_size, shuffle=True, collate_fn=collate, generator=torch.Generator().manual_seed(args.seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    history = []
    begin = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            _, _, loss = model(**batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            n = batch['input_ids'].shape[0]
            loss_sum += loss.item() * n
            count += n
        val_loss, val_preds = evaluate(model, val_rows, val, args.batch_size, device)
        em, f1 = summarize(val, val_preds)
        history.append({'max_length': max_length, 'epoch': epoch, 'train_loss': loss_sum / count, 'validation_loss': val_loss, 'validation_em': em, 'validation_f1': f1})
    if device.type == 'cuda':
        torch.cuda.synchronize()
    duration = time.perf_counter() - begin
    test_loss, preds = evaluate(model, test_rows, test, args.batch_size, device)
    em, f1 = summarize(test, preds)
    row = {'experiment': 'context_length', 'model': args.model, 'group': str(max_length), 'n': len(test), 'EM': em, 'F1': f1, 'training_time_s': duration, 'test_loss': test_loss, 'status': 'measured'}
    return model, preds, row, history


def split_by_context(dataset, n_train, n_val, n_test, seed):
    source = dataset['train'].shuffle(seed=seed)
    official_val = dataset['validation'].shuffle(seed=seed)
    val = list(official_val.select(range(min(n_val, len(official_val)))))
    val_contexts = {e['context'] for e in val}
    test = []
    for e in official_val:
        if e['context'] not in val_contexts:
            test.append(e)
            if len(test) == n_test:
                break
    excluded = val_contexts | {e['context'] for e in test}
    train = []
    for e in source:
        if e['context'] not in excluded:
            train.append(e)
            if len(train) == n_train:
                break
    if (len(train), len(val), len(test)) != (n_train, n_val, n_test):
        raise ValueError('Not enough examples for requested splits')
    return train, val, test


def length_group(e, tokenizer):
    n = len(tokenizer(e['answers']['text'][0], add_special_tokens=False)['input_ids'])
    return '1' if n == 1 else '2–3' if n <= 3 else '4–6' if n <= 6 else '7+'


def overlap_value(e):
    a = e['answers']['answer_start'][0]
    q = terms(e['question'])
    sentence = next((m.group() for m in sentences(e['context']) if m.start() <= a < m.end()), e['context'])
    return len(q & terms(sentence)) / max(1, len(q))


def perturb(e, mode):
    context = e['context']
    if mode in ('irrelevant', 'distractor'):
        prefix = 'The laboratory stores spare chairs and unused notebooks. The weather was mild during the afternoon. ' if mode == 'irrelevant' else 'An unrelated quiz contained the question: ' + e['question'].strip() + ' The quiz answer is not provided. '
        return {**e, 'context': prefix + context, 'answers': {'text': e['answers']['text'], 'answer_start': [a + len(prefix) for a in e['answers']['answer_start']]}}
    cuts = [0] + [m.end() for m in re.finditer(r'(?<=[.!?])\s+', context)] + [len(context)]
    cuts = sorted(set(cuts))
    spans = list(zip(cuts, cuts[1:]))
    if len(spans) < 2 or any(not any(a >= l and a + len(t) <= r for l, r in spans) for a, t in zip(e['answers']['answer_start'], e['answers']['text'])):
        return None
    parts = []
    mapped = {}
    cursor = 0
    for l, r in reversed(spans):
        piece = context[l:r]
        parts.append(piece)
        for a in e['answers']['answer_start']:
            if l <= a < r:
                mapped[a] = cursor + a - l
        cursor += len(piece) + 1
    return {**e, 'context': ' '.join(parts), 'answers': {'text': e['answers']['text'], 'answer_start': [mapped[a] for a in e['answers']['answer_start']]}}


def group_row(experiment, group, examples, preds, model=MODEL):
    em, f1 = summarize(examples, preds)
    return {'experiment': experiment, 'model': model, 'group': group, 'n': len(examples), 'EM': em, 'F1': f1, 'status': 'measured'}


def write_csv(path, rows, fields):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


RESULT_FIELDS = ['experiment', 'model', 'group', 'n', 'EM', 'F1', 'training_time_s', 'test_loss', 'status']
HISTORY_FIELDS = ['max_length', 'epoch', 'train_loss', 'validation_loss', 'validation_em', 'validation_f1']


def error_type(e, p):
    candidates = [(a, a + len(t), t) for a, t in zip(e['answers']['answer_start'], e['answers']['text'])]
    if any(p['start'] == a and p['end'] == b for a, b, _ in candidates):
        return 'correct', 'Trùng vị trí với một đáp án chuẩn.', candidates[0][2]
    a, b, gold = max(candidates, key=lambda x: (int(p['start'] == x[0]) + int(p['end'] == x[1]), metrics(p['text'], [x[2]])[1], max(0, min(p['end'], x[1]) - max(p['start'], x[0]))))
    if p['start'] == a:
        return 'wrong end', 'Thiếu phần cuối của gold.' if p['end'] < b else 'Lấy thêm văn bản sau gold.', gold
    if p['end'] == b:
        return 'wrong start', 'Thiếu phần đầu của gold.' if p['start'] > a else 'Lấy thêm văn bản trước gold.', gold
    if p['start'] < b and p['end'] > a:
        return 'wrong span', 'Cả hai biên sai nhưng span còn giao với đáp án.', gold
    return 'wrong span', 'Span dự đoán nằm ngoài vị trí đáp án được chọn.', gold


def error_report(examples, preds, path):
    groups = collections.defaultdict(list)
    partial = []
    count = collections.Counter()
    for e, p in zip(examples, preds):
        em, f1 = metrics(p['text'], e['answers']['text'])
        kind, reason, gold = error_type(e, p)
        count[kind] += 1
        if kind != 'correct' and p['text'] and p['text'] in gold and p['text'] != gold:
            reason += ' Partial answer.'
        if kind == 'wrong span' and p['text'] and p['text'] == gold:
            reason += ' Cùng answer text nhưng khác vị trí xuất hiện.'
        item = (e['question'], gold, p['text'], kind, reason, f'{f1:.4f}')
        groups[kind].append(item)
        if em == 0 and f1 > 0:
            partial.append(item)
    def table(items):
        lines = ['| Question | Gold | Prediction | Error type | Explanation | F1 |', '|---|---|---|---|---|---:|']
        for item in items[:5]:
            lines.append('| ' + ' | '.join(str(x).replace('|', '\\|').replace('\n', ' ') for x in item) + ' |')
        return '\n'.join(lines)
    lines = ['# Error analysis', '', f'Tổng số câu: {len(examples)}', '', f'Wrong start riêng: {count["wrong start"]}', f'Wrong end riêng: {count["wrong end"]}', f'Wrong span: {count["wrong span"]}', f'Tổng số sai start: {count["wrong start"] + count["wrong span"]}', f'Tổng số sai end: {count["wrong end"] + count["wrong span"]}', '', 'Các nhóm được xác định theo vị trí ký tự, đối chiếu với đáp án chuẩn có biên phù hợp nhất. EM và F1 lấy giá trị cao nhất trên các đáp án chuẩn.', '']
    for kind in ('wrong start', 'wrong end', 'wrong span'):
        lines += [f'## {kind}', '', f'Số ví dụ tìm được: {len(groups[kind])}', '', table(groups[kind]), '']
    lines += ['## EM = 0 và F1 > 0', '', f'Số ví dụ tìm được: {len(partial)}', '', table(partial), '', '## Nhận xét', '', 'Sai start có thể do model chọn thiếu từ bổ nghĩa hoặc lấy thêm từ đứng trước thực thể. Sai end có thể do model chọn thiếu thành phần phía sau hoặc mở rộng sang cụm từ lân cận. Sai cả hai biên có thể liên quan đến distractor, nhiều span hợp lý hoặc sự mơ hồ của câu hỏi. Truncation, context dài, thiếu thông tin và tokenization cũng là các giả thuyết cần kiểm tra trực tiếp bằng context và offset mapping. Không thể xác định nguyên nhân ngữ nghĩa chỉ từ hai vị trí biên.', '', 'F1 vẫn dương nếu prediction và gold có token chung sau normalization, dù EM bằng 0.']
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def plots(rows, history, root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(root / name, dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for length in (128, 256, 384):
        h = [x for x in history if x['max_length'] == length]
        if h:
            ep = [x['epoch'] for x in h]
            axes[0].plot(ep, [x['train_loss'] for x in h], marker='o', label=f'Train {length}')
            axes[0].plot(ep, [x['validation_loss'] for x in h], marker='s', linestyle='--', label=f'Val {length}')
            axes[1].plot(ep, [x['validation_f1'] for x in h], marker='o', label=f'F1 {length}')
            axes[1].plot(ep, [x['validation_em'] for x in h], marker='s', linestyle='--', label=f'EM {length}')
    for ax, title, ylabel in zip(axes, ['Training and validation loss', 'Validation EM and F1'], ['Loss', 'Score (%)']):
        ax.set(title=title, xlabel='Epoch', ylabel=ylabel)
        ax.legend()
        ax.grid(alpha=0.2)
    save(fig, 'learning_curves.png')
    for exp, name, title in [('model_comparison', 'em_f1_analysis.png', 'Baseline vs fine-tuned Transformer'), ('answer_length', 'answer_length_analysis.png', 'Scores by answer length')]:
        subset = [r for r in rows if r['experiment'] == exp and r['EM'] is not None]
        fig, ax = plt.subplots(figsize=(7, 4))
        x = list(range(len(subset)))
        ax.bar([i - .18 for i in x], [r['EM'] for r in subset], .36, label='EM')
        ax.bar([i + .18 for i in x], [r['F1'] for r in subset], .36, label='F1')
        ax.set_xticks(x, [r['group'] for r in subset])
        ax.set(title=title, ylabel='Score (%)', ylim=(0, 100))
        ax.legend()
        save(fig, name)



def experiment_report(rows, path):
    def get(exp, group):
        return next((r for r in rows if r['experiment'] == exp and r['group'] == group), None)
    def compare(exp, a, b):
        x, y = get(exp, a), get(exp, b)
        if not x or not y or x['F1'] is None or y['F1'] is None:
            return 'Không đủ dữ liệu để so sánh.'
        return f'{b} so với {a}: EM thay đổi {y["EM"] - x["EM"]:+.2f} điểm phần trăm, F1 thay đổi {y["F1"] - x["F1"]:+.2f} điểm phần trăm.'
    lines = ['# Experiment analysis', '', '## Context length', '', compare('context_length', '128', '256'), compare('context_length', '256', '384'), '', 'Max length lớn hơn không bảo đảm chất lượng tăng vì thông tin bổ sung có thể không liên quan. Thời gian và điểm số được ghi trong results.csv.', '', '## Baseline', '', compare('model_comparison', 'Baseline', 'Fine-tuned'), '', '## Prediction 1', '', compare('lexical_overlap', 'low', 'high'), compare('distractor', 'original', 'modified'), '', 'Lexical overlap là phân nhóm quan sát; thí nghiệm distractor kiểm tra tác động của việc thêm câu chứa question.', '', '## Prediction 2', '', compare('answer_length', '1', '2–3'), compare('answer_length', '2–3', '4–6'), compare('answer_length', '4–6', '7+'), '', '## Prediction 3', '', compare('irrelevant', 'original', 'modified'), compare('distractor', 'original', 'modified'), '', '## Prediction 4', '', compare('reorder', 'original', 'modified'), '', 'Các chênh lệch trên mô tả tập test đã chọn, không tự chứng minh quan hệ nhân quả cho các nhóm quan sát.']
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(args):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    dataset = load_dataset('rajpurkar/squad')
    train, val, test = split_by_context(dataset, args.train_size, args.val_size, args.test_size, args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    rows, history = [], []
    base = [baseline(e['question'], e['context']) for e in test]
    rows.append(group_row('model_comparison', 'Baseline', test, base, 'keyword_overlap'))
    for length in (128, 256, 384):
        model, preds, row, h = train_one(tokenizer, train, val, test, length, args.stride, args, device)
        rows.append(row)
        history.extend(h)
        write_csv(root / 'results.csv', rows, RESULT_FIELDS)
        write_csv(root / 'history.csv', history, HISTORY_FIELDS)
        if length != 384:
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    checkpoint = root / 'checkpoint'
    checkpoint.mkdir(exist_ok=True)
    tokenizer.save_pretrained(checkpoint)
    model.backbone.config.save_pretrained(checkpoint)
    torch.save({'model_name': args.model, 'state_dict': model.state_dict(), 'max_length': 384, 'stride': args.stride}, checkpoint / 'model.pt')
    rows.append(group_row('model_comparison', 'Fine-tuned', test, preds, args.model))
    for group in ('high', 'low'):
        idx = [i for i, e in enumerate(test) if (overlap_value(e) >= 0.5) == (group == 'high')]
        rows.append(group_row('lexical_overlap', group, [test[i] for i in idx], [preds[i] for i in idx], args.model))
    for group in ('1', '2–3', '4–6', '7+'):
        idx = [i for i, e in enumerate(test) if length_group(e, tokenizer) == group]
        rows.append(group_row('answer_length', group, [test[i] for i in idx], [preds[i] for i in idx], args.model))
    for mode in ('irrelevant', 'distractor', 'reorder'):
        pairs = [(i, perturb(e, mode)) for i, e in enumerate(test)]
        pairs = [(i, e) for i, e in pairs if e is not None]
        original = [test[i] for i, _ in pairs]
        changed = [e for _, e in pairs]
        rows.append(group_row(mode, 'original', original, [preds[i] for i, _ in pairs], args.model))
        if changed:
            _, new_preds = evaluate(model, features(tokenizer, changed, 384, args.stride, False), changed, args.batch_size, device)
            rows.append(group_row(mode, 'modified', changed, new_preds, args.model))
    write_csv(root / 'results.csv', rows, RESULT_FIELDS)
    write_csv(root / 'history.csv', history, HISTORY_FIELDS)
    error_report(test, preds, root / 'error_analysis.md')
    experiment_report(rows, root / 'experiment_analysis.md')
    (root / 'predictions.json').write_text(json.dumps([{'example': e, 'baseline': b, 'prediction': p} for e, b, p in zip(test, base, preds)], ensure_ascii=False, indent=2), encoding='utf-8')
    plots(rows, history, root)
    info = {'Model': args.model, 'Tokenizer': args.model + ' fast', 'Dataset': 'SQuAD 1.1 rajpurkar/squad', 'Train split': len(train), 'Validation split': len(val), 'Test split': len(test), 'Max sequence length': [128, 256, 384], 'Stride': args.stride, 'Batch size': args.batch_size, 'Learning rate': args.lr, 'Epochs': args.epochs, 'Optimizer': 'AdamW', 'Random seed': args.seed, 'Hardware': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU', 'Training time': {r['group']: r['training_time_s'] for r in rows if r['experiment'] == 'context_length'}}
    (root / 'reproducibility.json').write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')
    (root / 'split_ids.json').write_text(json.dumps({k: [e['id'] for e in v] for k, v in [('train', train), ('validation', val), ('test', test)]}, indent=2), encoding='utf-8')
    return rows


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=MODEL)
    p.add_argument('--train-size', type=int, default=8000)
    p.add_argument('--val-size', type=int, default=1000)
    p.add_argument('--test-size', type=int, default=1500)
    p.add_argument('--stride', type=int, default=32)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--output', default=str(ROOT))
    return p


if __name__ == '__main__':
    main(parser().parse_args())
