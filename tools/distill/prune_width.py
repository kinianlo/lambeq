#!/usr/bin/env python3
"""Structured width pruning for a distilled Bobcat student.

Usage:
    python tools/distill/prune_width.py MODEL_DIR OUT_DIR \
        [--heads-removed 4] [--ffn-keep 2048]

Removes the least important attention heads per encoder layer (using the
model's native ``prune_heads``, which records ``config.pruned_heads`` so the
surgery is re-applied automatically on ``from_pretrained``), and prunes the
least important FFN channels per layer down to a uniform ``--ffn-keep`` count.
"""
import argparse

import torch
from torch import nn
from transformers import AutoTokenizer

from lambeq.bobcat import BertForChartClassification


def head_importance(layer: nn.Module, num_heads: int,
                    head_size: int) -> torch.Tensor:
    """L2 norm of the attention output projection columns per head."""
    weight = layer.attention.output.dense.weight  # [hidden, hidden]
    cols = weight.view(weight.shape[0], num_heads, head_size)
    return cols.norm(dim=(0, 2))  # [num_heads]


def heads_to_remove(layer: nn.Module, num_heads: int, head_size: int,
                    n_remove: int) -> list[int]:
    if n_remove <= 0:
        return []
    importance = head_importance(layer, num_heads, head_size)
    return importance.argsort()[:n_remove].sort().values.tolist()


def ffn_keep_indices(layer: nn.Module, keep: int) -> torch.Tensor:
    """Top-``keep`` FFN channels by combined in/out weight norm."""
    in_w = layer.intermediate.dense.weight   # [intermediate, hidden]
    out_w = layer.output.dense.weight        # [hidden, intermediate]
    importance = in_w.norm(dim=1) * out_w.norm(dim=0)  # [intermediate]
    keep = min(keep, importance.shape[0])
    return importance.argsort(descending=True)[:keep].sort().values


def prune_ffn(layer: nn.Module, keep: int) -> None:
    idx = ffn_keep_indices(layer, keep)
    inter = layer.intermediate.dense
    out = layer.output.dense
    hidden = inter.weight.shape[1]
    n_keep = idx.shape[0]

    new_inter = nn.Linear(hidden, n_keep, bias=inter.bias is not None)
    new_inter.weight = nn.Parameter(
        inter.weight.index_select(0, idx).clone())
    if inter.bias is not None:
        new_inter.bias = nn.Parameter(inter.bias.index_select(0, idx).clone())

    new_out = nn.Linear(n_keep, hidden, bias=out.bias is not None)
    new_out.weight = nn.Parameter(out.weight.index_select(1, idx).clone())
    if out.bias is not None:
        new_out.bias = nn.Parameter(out.bias.clone())

    layer.intermediate.dense = new_inter
    layer.output.dense = new_out


def tiny_forward(model: BertForChartClassification) -> tuple[torch.Tensor,
                                                             torch.Tensor]:
    model.eval()
    with torch.no_grad():
        out = model(input_ids=torch.ones(1, 8, dtype=torch.long),
                    attention_mask=torch.ones(1, 8, dtype=torch.long),
                    token_type_ids=torch.zeros(1, 8, dtype=torch.long))
    return out.tag_logits, out.span_logits


def main() -> None:
    argp = argparse.ArgumentParser(description=__doc__)
    argp.add_argument('model_dir')
    argp.add_argument('out_dir')
    argp.add_argument('--heads-removed', type=int, default=4)
    argp.add_argument('--ffn-keep', type=int, default=2048)
    args = argp.parse_args()

    model = BertForChartClassification.from_pretrained(args.model_dir)
    config = model.config
    n_before = sum(p.numel() for p in model.parameters())

    num_heads = config.num_attention_heads
    head_size = config.hidden_size // num_heads

    # Head pruning -------------------------------------------------------
    layers = model.bert.encoder.layer
    prune_map: dict[int, list[int]] = {}
    for i, layer in enumerate(layers):
        removed = heads_to_remove(layer, num_heads, head_size,
                                  args.heads_removed)
        if removed:
            prune_map[i] = removed
    if prune_map:
        model.prune_heads(prune_map)

    # FFN channel pruning ------------------------------------------------
    for layer in layers:
        prune_ffn(layer, args.ffn_keep)
    config.intermediate_size = min(args.ffn_keep, config.intermediate_size)

    n_after = sum(p.numel() for p in model.parameters())

    # Save ---------------------------------------------------------------
    model.save_pretrained(args.out_dir)
    AutoTokenizer.from_pretrained(args.model_dir).save_pretrained(args.out_dir)

    # Reload-allclose check: catches save/load surgery bugs --------------
    tag_a, span_a = tiny_forward(model)
    reloaded = BertForChartClassification.from_pretrained(args.out_dir)
    reloaded_pruned = {k: set(v)
                       for k, v in reloaded.config.pruned_heads.items()}
    expected_pruned = {k: set(v) for k, v in config.pruned_heads.items()}
    assert reloaded_pruned == expected_pruned, (
        'pruned_heads not round-tripped: '
        f'{reloaded.config.pruned_heads} != {config.pruned_heads}')
    assert reloaded.config.intermediate_size == config.intermediate_size
    tag_b, span_b = tiny_forward(reloaded)
    assert torch.allclose(tag_a, tag_b, atol=1e-5), 'tag_logits mismatch'
    assert torch.allclose(span_a, span_b, atol=1e-5), 'span_logits mismatch'

    # Summary ------------------------------------------------------------
    print('per-layer removed head ids:')
    for i in range(len(layers)):
        print(f'  layer {i}: {prune_map.get(i, [])}')
    print(f'config.pruned_heads: {dict(config.pruned_heads)}')
    print(f'intermediate_size: {config.intermediate_size}')
    print(f'params before: {n_before/1e6:.2f}M')
    print(f'params after:  {n_after/1e6:.2f}M '
          f'({100*(n_before-n_after)/n_before:.1f}% removed)')
    print('reload-allclose check: PASSED')


if __name__ == '__main__':
    main()
