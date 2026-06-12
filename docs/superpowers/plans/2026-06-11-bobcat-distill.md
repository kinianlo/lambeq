# Distilled Bobcat Supertagger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a 6-layer student of the 30-layer Bobcat supertagger via teacher-only distillation on wikitext, gated at >=95% tree agreement, packaged as a drop-in model dir.

**Architecture:** Depth-pruned `BertForChartClassification` student (same hidden size/tokenizer/heads, layers initialised from equally spaced teacher layers) trained with temperature-scaled KL on tag+span logits against the frozen teacher, online, on unlabeled wikitext-103 sentences. Tree-agreement evaluation reuses `BobcatParser` + `CCGTree.__eq__`.

**Tech Stack:** PyTorch (AMP bf16), transformers, SGE on beaker (A40), the existing lambeq/bobcat stack.

**Spec:** `docs/superpowers/specs/2026-06-11-bobcat-distill-design.md`
**Spec deviation (accepted):** the SGE job does NOT self-resubmit (qsub is unavailable on compute nodes); instead checkpoints are resumable and the controller resubmits manually if a job dies — equivalent function.

---

## Conventions

- Repo `/home/kinianlo/projects/lambeq`, branch `bobcat-throughput`.
  `PY=~/.pyenv/versions/qnlp/bin/python`. 600000ms timeouts on
  model-loading runs. Commit trailer:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Teacher model dir (local): `~/.cache/lambeq/bobcat/bobcat`. On the
  share: `$BASE/cache/lambeq/bobcat/bobcat` where
  `BASE=/SAN/intelsys/discoviz/kinianlo/lambeq-bench` (beaker) =
  `/cs/research/intelsys/discoviz/kinianlo/lambeq-bench` (lab hosts).
- Distill workspace on the share: `$BASE/distill/` (data/, checkpoints/,
  logs/, student dirs).

## File map

```
tools/distill/prepare_data.py    wikitext download/clean/split -> JSONL
tools/distill/make_student.py    depth-pruned student initialisation
tools/distill/distill.py         online distillation training loop
tools/distill/eval_agreement.py  tree-agreement gate teacher vs student
tools/distill/package_student.py final model dir assembly
benchmarks/bobcat_throughput.py  + --model-dir flag
tests/test_distill_smoke.py      CPU smoke (env-guarded)
```

---

### Task 1: Data preparation

**Files:** Create `tools/distill/prepare_data.py`; no test file (the
smoke is running it).

- [ ] **Step 1: write the script** — exactly:

```python
#!/usr/bin/env python3
"""Prepare wikitext-103 sentences for Bobcat distillation.

Usage: python tools/distill/prepare_data.py OUT_DIR [--max-train N]
Writes train.jsonl, heldout.jsonl (2000), dev.jsonl (64) — one JSON
list of tokens per line.
"""
import argparse
import json
import random
import re
from pathlib import Path

SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'])')
TOKEN_OK = re.compile(r'^[\x20-\x7e]+$')  # printable ASCII only


def sentences_from_wikitext():
    from datasets import load_dataset
    ds = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1',
                      split='train')
    for record in ds:
        text = record['text'].strip()
        if not text or text.startswith('='):
            continue
        text = text.replace(' @-@ ', '-').replace(' @.@ ', '.') \
                   .replace(' @,@ ', ',').replace('<unk>', '')
        for sent in SENT_SPLIT.split(text):
            yield sent.strip()


def main() -> None:
    argp = argparse.ArgumentParser(description=__doc__)
    argp.add_argument('out_dir')
    argp.add_argument('--max-train', type=int, default=1_500_000)
    args = argp.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    kept: list[list[str]] = []
    target = args.max_train + 2064
    for sent in sentences_from_wikitext():
        words = sent.split()
        if not 4 <= len(words) <= 60:
            continue
        if not all(TOKEN_OK.match(w) for w in words):
            continue
        key = ' '.join(words)
        if key in seen:
            continue
        seen.add(key)
        kept.append(words)
        if len(kept) >= target:
            break

    random.Random(0).shuffle(kept)
    splits = {'heldout.jsonl': kept[:2000],
              'dev.jsonl': kept[2000:2064],
              'train.jsonl': kept[2064:]}
    for name, rows in splits.items():
        with open(out / name, 'w') as f:
            for words in rows:
                f.write(json.dumps(words) + '\n')
        print(f'{name}: {len(rows)} sentences')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: install datasets + run locally**

```bash
$PY -m pip install -q datasets
$PY tools/distill/prepare_data.py /tmp/distill-data
wc -l /tmp/distill-data/*.jsonl
```

Expected: heldout 2000, dev 64, train ~1.5M (the wikitext download is
~1GB on first run; allow time). If `datasets` cannot reach the hub,
report BLOCKED (the controller will arrange an alternate corpus).

- [ ] **Step 3: ship to the share** (controller runs this; the share is
  reachable from goosander/beaker, not from the laptop directly):

```bash
rsync -az -e "ssh -o BatchMode=yes -o ProxyJump=knuckles" /tmp/distill-data/ \
  kinianlo@goosander-l.cs.ucl.ac.uk:/cs/research/intelsys/discoviz/kinianlo/lambeq-bench/distill/data/
```

- [ ] **Step 4: commit**

```bash
git add tools/distill/prepare_data.py
git commit -m "Add wikitext data preparation for supertagger distillation

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Student init + training loop + smoke test

**Files:** Create `tools/distill/make_student.py`,
`tools/distill/distill.py`, `tests/test_distill_smoke.py`.

- [ ] **Step 1: `make_student.py`** — exactly:

```python
#!/usr/bin/env python3
"""Initialise a depth-pruned Bobcat student from the teacher.

Usage: python tools/distill/make_student.py TEACHER_DIR STUDENT_DIR --layers 6
Copies embeddings, heads and equally spaced encoder layers.
"""
import argparse
import re

import torch
from transformers import AutoTokenizer

from lambeq.bobcat import BertForChartClassification

LAYER_RE = re.compile(r'(bert\.encoder\.layer\.)(\d+)(\..*)')


def layer_map(teacher_layers: int, student_layers: int) -> list[int]:
    if student_layers == 1:
        return [teacher_layers - 1]
    return [round(i * (teacher_layers - 1) / (student_layers - 1))
            for i in range(student_layers)]


def main() -> None:
    argp = argparse.ArgumentParser(description=__doc__)
    argp.add_argument('teacher_dir')
    argp.add_argument('student_dir')
    argp.add_argument('--layers', type=int, default=6)
    args = argp.parse_args()

    teacher = BertForChartClassification.from_pretrained(args.teacher_dir)
    config = teacher.config
    t_layers = config.num_hidden_layers
    mapping = layer_map(t_layers, args.layers)
    print(f'layer map (student -> teacher): {mapping}')

    config.num_hidden_layers = args.layers
    student = BertForChartClassification(config)

    teacher_sd = teacher.state_dict()
    new_sd = {}
    for key in student.state_dict():
        m = LAYER_RE.match(key)
        src = (f'{m.group(1)}{mapping[int(m.group(2))]}{m.group(3)}'
               if m else key)
        new_sd[key] = teacher_sd[src].clone()
    student.load_state_dict(new_sd, strict=True)
    student.save_pretrained(args.student_dir)
    AutoTokenizer.from_pretrained(args.teacher_dir).save_pretrained(
        args.student_dir)
    n = sum(p.numel() for p in student.parameters())
    print(f'student saved: {args.layers} layers, {n/1e6:.0f}M params')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: `distill.py`** — exactly:

```python
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
```

- [ ] **Step 3: smoke test** — create `tests/test_distill_smoke.py`:

```python
"""CPU smoke test for the distillation pipeline (slow; opt-in).

Run with: LAMBEQ_DISTILL_SMOKE=1 pytest tests/test_distill_smoke.py -q
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

if not os.environ.get('LAMBEQ_DISTILL_SMOKE'):
    pytest.skip('set LAMBEQ_DISTILL_SMOKE=1 to run', allow_module_level=True)

TEACHER = Path.home() / '.cache/lambeq/bobcat/bobcat'


def test_distill_pipeline_smoke(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    sents = [['Alice', 'likes', 'Bob', 'today'],
             ['the', 'cat', 'sees', 'the', 'dog'],
             ['I', 'do', 'not', 'like', 'green', 'eggs']] * 20
    for name, rows in (('train.jsonl', sents),
                       ('dev.jsonl', sents[:8]),
                       ('heldout.jsonl', sents[:10])):
        with open(data / name, 'w') as f:
            for s in rows:
                f.write(json.dumps(s) + '\n')

    student = tmp_path / 'student-init'
    subprocess.run([sys.executable, 'tools/distill/make_student.py',
                    str(TEACHER), str(student), '--layers', '2'],
                   check=True)
    out = tmp_path / 'run'
    subprocess.run([sys.executable, 'tools/distill/distill.py',
                    str(data), str(TEACHER), str(student), str(out),
                    '--device', 'cpu', '--max-steps', '5',
                    '--token-budget', '64', '--checkpoint-every', '2'],
                   check=True)
    assert (out / 'student-final' / 'model.safetensors').exists() \
        or (out / 'student-final' / 'pytorch_model.bin').exists()
    assert (out / 'checkpoint.pt').exists()
    # resume path: run again for 2 more steps
    subprocess.run([sys.executable, 'tools/distill/distill.py',
                    str(data), str(TEACHER), str(student), str(out),
                    '--device', 'cpu', '--max-steps', '7',
                    '--token-budget', '64', '--checkpoint-every', '2'],
                   check=True)
```

- [ ] **Step 4: run the smoke**

```bash
LAMBEQ_DISTILL_SMOKE=1 $PY -m pytest tests/test_distill_smoke.py -q
```

Expected: 1 passed (CPU; a couple of minutes — teacher loads twice).
Debug until green; the resume-print must appear in the second run's
output.

- [ ] **Step 5: commit**

```bash
git add tools/distill/make_student.py tools/distill/distill.py \
    tests/test_distill_smoke.py
git commit -m "Add student initialisation and distillation training loop

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Evaluation gate + packaging + benchmark flag

**Files:** Create `tools/distill/eval_agreement.py`,
`tools/distill/package_student.py`; Modify
`benchmarks/bobcat_throughput.py`.

- [ ] **Step 1: `eval_agreement.py`** — exactly:

```python
#!/usr/bin/env python3
"""Tree-agreement evaluation: student Bobcat vs teacher Bobcat.

Usage: python tools/distill/eval_agreement.py TEACHER_DIR STUDENT_DIR \
    SENTENCES.jsonl [--device cuda] [--gate 0.95] [--limit N]
Exit code 1 if --gate is set and agreement falls below it.
"""
import argparse
import json
import sys

from lambeq import BobcatParser, VerbosityLevel


def main() -> int:
    argp = argparse.ArgumentParser(description=__doc__)
    argp.add_argument('teacher_dir')
    argp.add_argument('student_dir')
    argp.add_argument('sentence_file')
    argp.add_argument('--device', default='cuda')
    argp.add_argument('--gate', type=float, default=None)
    argp.add_argument('--limit', type=int, default=None)
    args = argp.parse_args()

    with open(args.sentence_file) as f:
        sentences = [json.loads(line) for line in f][:args.limit]

    common = dict(verbose=VerbosityLevel.SUPPRESS.value,
                  device=args.device, parser_backend='rust')
    teacher = BobcatParser(model_name_or_path=args.teacher_dir, **common)
    student = BobcatParser(model_name_or_path=args.student_dir, **common)

    t_trees = teacher.sentences2trees(sentences, tokenised=True,
                                      suppress_exceptions=True)
    s_trees = student.sentences2trees(sentences, tokenised=True,
                                      suppress_exceptions=True)

    n = len(sentences)
    same = sum(1 for a, b in zip(t_trees, s_trees)
               if a is not None and a == b)
    both_fail = sum(1 for a, b in zip(t_trees, s_trees)
                    if a is None and b is None)
    t_fail = sum(t is None for t in t_trees)
    s_fail = sum(t is None for t in s_trees)
    agreement = (same + both_fail) / n
    print(json.dumps({'sentences': n,
                      'agreement': round(agreement, 4),
                      'identical_trees': same,
                      'both_fail': both_fail,
                      'teacher_fail': t_fail,
                      'student_fail': s_fail}))
    if args.gate is not None and agreement < args.gate:
        print(f'GATE FAILED: {agreement:.4f} < {args.gate}',
              file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
```

- [ ] **Step 2: `package_student.py`** — exactly:

```python
#!/usr/bin/env python3
"""Assemble a complete drop-in Bobcat model dir from distilled weights.

Usage: python tools/distill/package_student.py STUDENT_WEIGHTS_DIR \
    TEACHER_DIR OUT_DIR
Copies grammar/pipeline/tokenizer/version files from the teacher next
to the student weights.
"""
import argparse
import shutil
from pathlib import Path

COPY = ['grammar.json', 'pipeline_config.json', 'version.txt',
        'tokenizer.json', 'tokenizer_config.json', 'vocab.txt',
        'special_tokens_map.json']


def main() -> None:
    argp = argparse.ArgumentParser(description=__doc__)
    argp.add_argument('student_weights_dir')
    argp.add_argument('teacher_dir')
    argp.add_argument('out_dir')
    args = argp.parse_args()
    src = Path(args.student_weights_dir)
    teacher = Path(args.teacher_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        shutil.copy2(item, out / item.name)
    for name in COPY:
        path = teacher / name
        if path.exists() and not (out / name).exists():
            shutil.copy2(path, out / name)
    print(f'packaged: {out}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 3: benchmark `--model-dir`.** In
`benchmarks/bobcat_throughput.py` add
`argp.add_argument('--model-dir', default=None)` and construct the
parser with
`BobcatParser(model_name_or_path=args.model_dir or 'bobcat', ...)`
(adjust the existing constructor call; all other kwargs unchanged).

- [ ] **Step 4: local micro-check.** Using the Task-2 smoke artifacts
(rerun the smoke if `tmp_path` is gone — or regenerate a tiny student
via `make_student.py --layers 2` into `/tmp/distill-student2`):

```bash
$PY tools/distill/make_student.py ~/.cache/lambeq/bobcat/bobcat /tmp/distill-student2 --layers 2
$PY tools/distill/package_student.py /tmp/distill-student2 ~/.cache/lambeq/bobcat/bobcat /tmp/distill-packaged
$PY tools/distill/eval_agreement.py ~/.cache/lambeq/bobcat/bobcat /tmp/distill-packaged /tmp/distill-data/heldout.jsonl --device cpu --limit 10
```

Expected: a JSON line with agreement (likely LOW — the layers are
untrained-as-a-6-stack; that's fine, this checks plumbing not quality);
exit 0 (no --gate). The packaged dir must load through BobcatParser
without error (eval_agreement does that implicitly).

- [ ] **Step 5: commit**

```bash
git add tools/distill/eval_agreement.py tools/distill/package_student.py \
    benchmarks/bobcat_throughput.py
git commit -m "Add distillation evaluation gate, packaging and benchmark flag

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Launch training on beaker (controller-run)

- [ ] Sync branch + data to the share; `git pull` in
  `$BASE/lambeq-beaker`.
- [ ] Build the 6-layer student on the beaker login node (CPU, quick):

```bash
$BASE/venv-beaker/bin/python tools/distill/make_student.py \
  $BASE/cache/lambeq/bobcat/bobcat $BASE/distill/student6-init --layers 6
```

- [ ] SGE job (`$BASE/distill/distill_job.sh`):

```bash
#$ -S /bin/bash
#$ -l h_rt=12:00:0
#$ -l tmem=20G
#$ -l gpu=true
#$ -j y
#$ -N lambeq_distill
#$ -o /SAN/intelsys/discoviz/kinianlo/lambeq-bench/distill/
hostname; nvidia-smi -L | head -1; date
BASE=/SAN/intelsys/discoviz/kinianlo/lambeq-bench
export XDG_CACHE_HOME=$BASE/cache
cd $BASE/lambeq-beaker
$BASE/venv-beaker/bin/python tools/distill/distill.py \
  $BASE/distill/data $BASE/cache/lambeq/bobcat/bobcat \
  $BASE/distill/student6-init $BASE/distill/run6 \
  --epochs 3 --device cuda
date
```

Submit with `qsub -l h=!mitchell.local`; poll; on h_rt death or node
failure, resubmit (the checkpoint resumes). Watch `run6/log.jsonl`:
loss should fall steadily and `dev_tag_agreement` should exceed 0.95
within the first epoch if distillation is working; if it plateaus
below 0.90 by end of epoch 1, stop and reassess (lr or depth).

---

### Task 5: Gate, package, benchmark, record (controller-run)

- [ ] `eval_agreement.py` on beaker or goosander GPU:
  `TEACHER vs $BASE/distill/run6/student-final` (packaged first) over
  `$BASE/distill/data/heldout.jsonl --gate 0.95`, plus the 937 corpus
  files (secondary, no gate), plus tag top-1 from the training log.
- [ ] If gate fails at 6 layers: rerun Task 4 with
  `make_student.py --layers 12` -> `$BASE/distill/run12` and re-gate.
- [ ] Package: `package_student.py $BASE/distill/run6/student-final
  $BASE/cache/lambeq/bobcat/bobcat $BASE/distill/bobcat-student6`.
- [ ] Throughput on goosander:
  `benchmarks/bobcat_throughput.py` long+short corpora `--device cuda
  --parser-backend rust --model-dir $BASE/distill/bobcat-student6`
  vs the teacher (no --model-dir). Verify >=3x encoder forward
  (compare tagging-stage rates).
- [ ] RESULTS.md `## Tier 4 (distilled student)` section: agreement
  table, throughput table, training cost (steps/wall time), verdict vs
  the spec gates. Commit + push. Copy the packaged student dir back to
  the laptop (`~/.cache/lambeq/bobcat-student6`) for local use.
