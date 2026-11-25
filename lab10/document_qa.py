import argparse
import json
from pathlib import Path

import torch
from qa_model import QAModel, features, evaluate


DOCUMENT = 'The Transformer architecture was introduced in 2017. DistilBERT is a smaller version of BERT. Extractive question answering predicts the start and end positions of an answer in a context. The SQuAD dataset contains questions and answer spans based on Wikipedia articles.'


class DocumentQA:
    def __init__(self, checkpoint):
        from transformers import AutoConfig, AutoTokenizer
        checkpoint = Path(checkpoint)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state = torch.load(checkpoint / 'model.pt', map_location='cpu', weights_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
        self.model = QAModel(state['model_name'], AutoConfig.from_pretrained(checkpoint))
        self.model.load_state_dict(state['state_dict'])
        self.model.to(self.device).eval()
        self.max_length = state['max_length']
        self.stride = state['stride']

    def answer(self, question, context):
        if not question.strip() or not context.strip():
            raise ValueError('Question and context must not be empty')
        examples = [{'question': question, 'context': context}]
        rows = features(self.tokenizer, examples, self.max_length, self.stride, False)
        _, predictions = evaluate(self.model, rows, examples, 8, self.device)
        return {k: v for k, v in predictions[0].items() if k != 'score'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('question')
    parser.add_argument('--document', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=Path(__file__).parent / 'checkpoint')
    args = parser.parse_args()
    context = args.document.read_text(encoding='utf-8') if args.document else DOCUMENT
    result = DocumentQA(args.checkpoint).answer(args.question, context)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
