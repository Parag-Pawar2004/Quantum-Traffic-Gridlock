#!/usr/bin/env bash
# Regenerates the submission outputs for every scenario in scenarios/ (seed 0 = canonical).
set -e
for f in scenarios/*.json; do
  name=$(python -c "import json,sys;print(json.load(open(sys.argv[1])).get('name','x'))" "$f")
  python run.py --scenario "$f" --seed 0 --out "outputs/$name"
done
