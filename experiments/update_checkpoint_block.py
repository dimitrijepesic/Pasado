"""Replaces the text between <!-- BEGIN:LABEL --> and <!-- END:LABEL --> in CHECKPOINT.md with
the contents of FILE (e.g. tables.md of the LiRPA analysis, table.md of the compilers).

  python experiments/update_checkpoint_block.py LABEL FILE [--note "text above the table"]
"""
import argparse
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("file")
ap.add_argument("--note", default="")
args = ap.parse_args()
path = os.path.join(ROOT, "CHECKPOINT.md")
doc = open(path).read()
begin, end = f"<!-- BEGIN:{args.label} -->", f"<!-- END:{args.label} -->"
if begin not in doc or end not in doc:
    raise SystemExit(f"markers for {args.label} not found")
head, rest = doc.split(begin, 1)
_, tail = rest.split(end, 1)
body = (args.note + "\n\n" if args.note else "") + open(args.file).read().strip()
open(path, "w").write(head + begin + "\n" + body + "\n" + end + tail)
print("updated block", args.label)
