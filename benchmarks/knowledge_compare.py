"""Compare two MMLU-Pro runs on the questions both have answered so far.

usage: python3 benchmarks/knowledge_compare.py <a.jsonl> <b.jsonl>
Works while a run is still in progress (reads the per-question .jsonl logs).
"""
import json
import sys
from collections import defaultdict


def load(p):
    out = {}
    for line in open(p):
        r = json.loads(line)
        out[r["question_id"]] = r
    return out


a, b = load(sys.argv[1]), load(sys.argv[2])
common = sorted(set(a) & set(b))
if not common:
    sys.exit("no questions in common yet")
cats = defaultdict(lambda: [0, 0, 0])
for q in common:
    c = cats[a[q]["category"]]
    c[0] += 1
    c[1] += a[q]["correct"]
    c[2] += b[q]["correct"]
na, nb = sum(a[q]["correct"] for q in common), sum(b[q]["correct"] for q in common)
print(f"{len(common)} questions in common (of {len(b)} answered in {sys.argv[2]})")
print(f"  {sys.argv[1].rsplit('/', 1)[-1]:40s} {na / len(common):6.1%}")
print(f"  {sys.argv[2].rsplit('/', 1)[-1]:40s} {nb / len(common):6.1%}")
print(f"  truncated in b: {sum(b[q]['finish'] == 'length' for q in common)}")
for cat, (n, ca, cb) in sorted(cats.items()):
    print(f"    {cat:18s} n={n:3d}  {ca / n:5.0%}  {cb / n:5.0%}")
