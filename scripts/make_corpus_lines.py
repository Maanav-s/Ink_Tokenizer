"""Write the text lines that the teacher corpus is generated from.

The teacher's charset is plain ASCII text (no '=', '<', '^', '_' or Greek),
so whiteboard math can't be written as-is. The mix is:

- prose: runs of dictionary words, sometimes capitalized or punctuated;
- notation: what the charset allows from engineering notes: single capital
  variables, truth-table rows, Boolean expressions with + and words, values
  with units, short identifiers;
- page text: strings from artifacts/*/content.json that fit the charset.

For distillation the line only needs to exercise the teacher's character
transitions, not to make sense. The word list is a host file (default
/usr/share/dict/words), so run this where one exists and copy the output
along with the corpus. It needs only the standard library, so it runs on the
host (the container can't see the host's word list).

Usage: python3 scripts/make_corpus_lines.py --out data/teacher_corpus/lines.txt
           [--count 200000] [--words /usr/share/dict/words] [--seed 0]
"""
import argparse
import glob
import json
import os
import random
import re


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_CHARS = 50  # the teacher loses its place more often on long lines


def prose(r, words):
    n = r.randint(1, 7)
    ws = [r.choice(words) for _ in range(n)]
    if r.random() < 0.5:
        ws[0] = ws[0].capitalize()
    line = " ".join(ws)
    if r.random() < 0.3:
        line += r.choice(".,:;?!")
    return line


def notation(r):
    var = lambda: r.choice("ABCDEFGHJKLMNPQRSTUVWXYZ")
    kind = r.randrange(6)
    if kind == 0:  # truth-table row
        return " ".join(r.choice("01") for _ in range(r.randint(2, 5)))
    if kind == 1:  # Boolean expression
        ops = [" + ", " and ", " or ", " xor ", " nand "]
        terms = [("not " if r.random() < 0.2 else "") + var() for _ in range(r.randint(2, 4))]
        expr = terms[0]
        for t in terms[1:]:
            expr += r.choice(ops) + t
        return f"({expr})" if r.random() < 0.3 else expr
    if kind == 2:  # value with unit
        value = r.choice([str(r.randint(1, 999)), f"{r.uniform(0.1, 99):.{r.randint(1, 2)}f}"])
        return f"{var()}{r.randint(1, 9)} {value}{r.choice(['k', 'M', 'm', 'u', 'n', ''])}{r.choice(['V', 'A', 'Hz', 'F', 'H', 's', ''])}"
    if kind == 3:  # identifier, function call
        name = r.choice(["f", "g", "clk", "rst", "en", "out", "sel", "Vin", "Vout", "data", "addr"])
        return f"{name}({', '.join(var() for _ in range(r.randint(1, 3)))})"
    if kind == 4:  # numbers
        return " ".join(str(r.randint(0, 9999)) for _ in range(r.randint(1, 4)))
    return " ".join(var() for _ in range(r.randint(1, 6)))  # column headers


def page_strings(charset):
    out = set()

    def walk(o):
        if isinstance(o, str):
            for part in o.split("\n"):
                part = part.strip()
                if part and len(part) <= MAX_CHARS and all(ch in charset for ch in part):
                    out.add(part)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    for path in glob.glob(os.path.join(ROOT, "artifacts", "*", "content.json")):
        walk(json.load(open(path)))
    return sorted(out)


def main(out, count, words_path, seed):
    meta = os.path.join(ROOT, "models", "teacher", "Epoch_52", "meta.json")
    charset = set(json.load(open(meta))["charset"])
    words = sorted({w.strip() for w in open(words_path) if re.fullmatch(r"[A-Za-z]{1,12}", w.strip())})
    pages = page_strings(charset)
    r = random.Random(seed)
    lines = set()
    while len(lines) < count:
        u = r.random()
        line = prose(r, words) if u < 0.65 else notation(r) if u < 0.95 or not pages else r.choice(pages)
        if len(line) <= MAX_CHARS and all(ch in charset for ch in line):
            lines.add(line)
    lines = sorted(lines)
    r.shuffle(lines)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{len(lines)} lines ({len(pages)} page strings available) -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data/teacher_corpus/lines.txt")
    ap.add_argument("--count", type=int, default=200000)
    ap.add_argument("--words", default="/usr/share/dict/words")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(args.out, args.count, args.words, args.seed)
