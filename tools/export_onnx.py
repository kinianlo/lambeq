#!/usr/bin/env python3
"""Export the Bobcat BERT encoder to ONNX (body only; heads run in torch).

Usage: python tools/export_onnx.py [model_dir]
Default model_dir: ~/.cache/lambeq/bobcat/bobcat
Writes <model_dir>/bobcat-body.onnx
"""
import sys
from pathlib import Path

import torch

from lambeq.bobcat import BertForChartClassification


class _BertBodyWrapper(torch.nn.Module):
    """Wrap model.bert so the exporter sees a plain tensor output."""

    def __init__(self, bert):
        super().__init__()
        self.bert = bert

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.bert(input_ids,
                         attention_mask=attention_mask,
                         token_type_ids=token_type_ids)[0]


def main() -> None:
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path.home() / '.cache/lambeq/bobcat/bobcat')
    model = BertForChartClassification.from_pretrained(model_dir).eval()
    wrapper = _BertBodyWrapper(model.bert)
    dummy = {
        'input_ids': torch.ones(2, 8, dtype=torch.long),
        'attention_mask': torch.ones(2, 8, dtype=torch.long),
        'token_type_ids': torch.zeros(2, 8, dtype=torch.long),
    }
    out_path = model_dir / 'bobcat-body.onnx'
    torch.onnx.export(
        wrapper,
        (dummy['input_ids'], dummy['attention_mask'],
         dummy['token_type_ids']),
        str(out_path),
        input_names=['input_ids', 'attention_mask', 'token_type_ids'],
        output_names=['last_hidden_state'],
        dynamic_axes={name: {0: 'batch', 1: 'sequence'}
                      for name in ('input_ids', 'attention_mask',
                                   'token_type_ids',
                                   'last_hidden_state')},
        opset_version=17)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
