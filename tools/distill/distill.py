#!/usr/bin/env python3
"""Online distillation of the Bobcat supertagger.

Usage:
  python tools/distill/distill.py DATA_DIR TEACHER_DIR STUDENT_DIR OUT_DIR \
      [--epochs 3] [--token-budget 4096] [--lr 1e-4] [--tau 2.0] \
      [--checkpoint-every 1000] [--device cuda] [--max-steps N]
Resumes automatically from OUT_DIR/checkpoint.pt if present.
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.nn.functional import kl_div, log_softmax

from lambeq.bobcat import BertForChartClassification, Tagger
from lambeq.bobcat.tagger import chart_size
from transformers import AutoTokenizer


def load_sentences(path: Path) -> list[list[str]]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def make_token_batches(sentences: list[list[str]],
                       budget: int) -> list[list[int]]:
    order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]))
    batches, batch, max_len = [], [], 0
    for i in order:
        new_max = max(max_len, len(sentences[i]))
        if batch and (len(batch) + 1) * new_max > budget:
            batches.append(batch)
            batch, max_len = [], 0
            new_max = len(sentences[i])
        batch.append(i)
        max_len = new_max
    if batch:
        batches.append(batch)
    return batches


def masked_kl(student_logits: torch.Tensor,
              teacher_logits: torch.Tensor,
              mask: torch.Tensor,
              tau: float) -> tuple[torch.Tensor, int]:
    s = log_softmax(student_logits[mask].float() / tau, dim=-1)
    t = log_softmax(teacher_logits[mask].float() / tau, dim=-1)
    n = int(mask.sum())
    return kl_div(s, t, log_target=True,
                  reduction='batchmean') * tau * tau, n


def length_masks(word_counts: list[int],
                 padded_words: int,
                 padded_spans: int,
                 device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    counts = torch.tensor(word_counts, device=device)
    pos_w = torch.arange(padded_words, device=device)
    tag_mask = pos_w.unsqueeze(0) < counts.unsqueeze(1)
    span_counts = torch.tensor([chart_size(c) for c in word_counts],
                               device=device)
    pos_s = torch.arange(padded_spans, device=device)
    span_mask = pos_s.unsqueeze(0) < span_counts.unsqueeze(1)
    return tag_mask, span_mask


def main() -> None:
    argp = argparse.ArgumentParser(description=__doc__)
    argp.add_argument('data_dir')
    argp.add_argument('teacher_dir')
    argp.add_argument('student_dir')
    argp.add_argument('out_dir')
    argp.add_argument('--epochs', type=int, default=3)
    argp.add_argument('--token-budget', type=int, default=4096)
    argp.add_argument('--lr', type=float, default=1e-4)
    argp.add_argument('--tau', type=float, default=2.0)
    argp.add_argument('--warmup', type=int, default=1000)
    argp.add_argument('--checkpoint-every', type=int, default=1000)
    argp.add_argument('--device', default='cuda')
    argp.add_argument('--max-steps', type=int, default=0)
    args = argp.parse_args()

    device = torch.device(args.device)
    amp_dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / 'log.jsonl', 'a')

    teacher = (BertForChartClassification.from_pretrained(args.teacher_dir)
               .eval().to(device))
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = (BertForChartClassification.from_pretrained(args.student_dir)
               .train().to(device))
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_dir)
    helper = Tagger(teacher, tokenizer)  # for prepare_inputs only

    train = load_sentences(Path(args.data_dir) / 'train.jsonl')
    dev = load_sentences(Path(args.data_dir) / 'dev.jsonl')
    batches = make_token_batches(train, args.token_budget)
    total_steps = args.epochs * len(batches)

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr,
                            weight_decay=0.01)

    def lr_lambda(step: int) -> float:
        if step < args.warmup:
            return step / max(1, args.warmup)
        t = (step - args.warmup) / max(1, total_steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    start_epoch = start_batch = step = 0
    ckpt_path = out / 'checkpoint.pt'
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        student.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        start_epoch, start_batch, step = ck['epoch'], ck['batch'], ck['step']
        print(f'resumed at epoch {start_epoch} batch {start_batch} '
              f'step {step}')

    def encode(words_batch: list[list[str]]) -> dict:
        enc = helper.prepare_inputs(words_batch, word_mask=True)
        return {k: torch.as_tensor(v, device=device) for k, v in enc.items()}

    def forward(model: torch.nn.Module, tensors: dict):
        with torch.autocast(device.type, dtype=amp_dtype,
                            enabled=device.type == 'cuda'):
            return model(**tensors)

    def dev_agreement() -> float:
        student.eval()
        with torch.inference_mode():
            tensors = encode(dev)
            t_out = forward(teacher, tensors)
            s_out = forward(student, tensors)
            counts = [len(w) for w in dev]
            tag_mask, _ = length_masks(counts, t_out.tag_logits.shape[1],
                                       1, device)
            agree = (t_out.tag_logits.argmax(-1)[tag_mask]
                     == s_out.tag_logits.argmax(-1)[tag_mask])
            return float(agree.float().mean())

    def save_checkpoint(epoch: int, batch: int) -> None:
        tmp = ckpt_path.with_suffix('.tmp')
        torch.save({'model': student.state_dict(),
                    'opt': opt.state_dict(),
                    'sched': sched.state_dict(),
                    'epoch': epoch, 'batch': batch, 'step': step}, tmp)
        tmp.rename(ckpt_path)

    t_start = time.time()
    done = False
    for epoch in range(start_epoch, args.epochs):
        order = torch.randperm(len(batches),
                               generator=torch.Generator().manual_seed(epoch)
                               ).tolist()
        first = start_batch if epoch == start_epoch else 0
        for bi in range(first, len(order)):
            words = [train[i] for i in batches[order[bi]]]
            tensors = encode(words)
            with torch.inference_mode():
                t_out = forward(teacher, tensors)
            s_out = forward(student, tensors)
            counts = [len(w) for w in words]
            tag_mask, span_mask = length_masks(
                counts, t_out.tag_logits.shape[1],
                t_out.span_logits.shape[1], device)
            kl_t, n_t = masked_kl(s_out.tag_logits,
                                  t_out.tag_logits.clone(), tag_mask,
                                  args.tau)
            kl_s, n_s = masked_kl(s_out.span_logits,
                                  t_out.span_logits.clone(), span_mask,
                                  args.tau)
            loss = (n_t * kl_t + n_s * kl_s) / (n_t + n_s)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1

            if step % 50 == 0:
                rec = {'step': step, 'epoch': epoch,
                       'loss': float(loss),
                       'kl_tag': float(kl_t), 'kl_span': float(kl_s),
                       'lr': sched.get_last_lr()[0],
                       'elapsed': round(time.time() - t_start, 1)}
                if step % 500 == 0:
                    rec['dev_tag_agreement'] = round(dev_agreement(), 4)
                    student.train()
                print(json.dumps(rec), flush=True)
                log_f.write(json.dumps(rec) + '\n')
                log_f.flush()
            if step % args.checkpoint_every == 0:
                save_checkpoint(epoch, bi + 1)
            if args.max_steps and step >= args.max_steps:
                done = True
                break
        start_batch = 0
        if done:
            break

    save_checkpoint(args.epochs, 0)
    student.save_pretrained(out / 'student-final')
    final_dev = dev_agreement()
    print(json.dumps({'final': True, 'step': step,
                      'dev_tag_agreement': round(final_dev, 4)}))
    log_f.close()


if __name__ == '__main__':
    main()
