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
