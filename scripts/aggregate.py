"""
Aggregate compliance results across all runs.

Reads all raw JSON files and produces summary tables:
  - Overall comply/refuse rates
  - By model
  - By prompt
  - By model x prompt (full matrix)
  - Experimental vs control comparison

Usage:
    python scripts/aggregate.py                # Print tables
    python scripts/aggregate.py --json         # Save to data/aggregate.json
"""

import argparse
import glob
import json
import os
from collections import defaultdict

from config import DATA_DIR


def load_all():
    raw_dir = os.path.join(DATA_DIR, "raw")
    records = []
    for f in sorted(glob.glob(os.path.join(raw_dir, "*.json"))):
        with open(f) as fh:
            records.append(json.load(fh))
    return records


def aggregate(records):
    by_model = defaultdict(lambda: {"comply": 0, "refuse": 0})
    by_prompt = defaultdict(lambda: {"comply": 0, "refuse": 0})
    by_model_prompt = defaultdict(lambda: {"comply": 0, "refuse": 0})
    by_category = {"experimental": {"comply": 0, "refuse": 0}, "control": {"comply": 0, "refuse": 0}}
    total = {"comply": 0, "refuse": 0}

    for r in records:
        model = r.get("model", "unknown")
        prompt_id = r.get("prompt_id", "unknown")
        compliance = r.get("compliance", "")

        if compliance not in ("comply", "refuse"):
            continue

        total[compliance] += 1
        by_model[model][compliance] += 1
        by_prompt[prompt_id][compliance] += 1
        by_model_prompt[(model, prompt_id)][compliance] += 1

        if prompt_id.startswith("C"):
            by_category["control"][compliance] += 1
        else:
            by_category["experimental"][compliance] += 1

    return total, by_model, by_prompt, by_model_prompt, by_category


def prompt_sort_key(pid):
    prefix = pid[0]
    rest = pid[1:]
    num = ""
    suffix = ""
    for c in rest:
        if c.isdigit():
            num += c
        else:
            suffix += c
    return (prefix, int(num) if num else 0, suffix)


def print_tables(total, by_model, by_prompt, by_model_prompt, by_category):
    n = total["comply"] + total["refuse"]
    print(f"OVERALL: {total['comply']}/{n} comply ({total['comply']/n*100:.1f}%)")
    print(f"         {total['refuse']}/{n} refuse ({total['refuse']/n*100:.1f}%)")

    # Experimental vs control
    print(f"\nEXPERIMENTAL vs CONTROL:")
    for cat in ("experimental", "control"):
        c = by_category[cat]
        t = c["comply"] + c["refuse"]
        rate = c["comply"] / t * 100 if t else 0
        print(f"  {cat:14s} comply={c['comply']}/{t} ({rate:.1f}%)  refuse={c['refuse']}/{t}")

    # By model
    print(f"\nBY MODEL:")
    models = sorted(by_model.keys())
    for model in models:
        m = by_model[model]
        t = m["comply"] + m["refuse"]
        rate = m["comply"] / t * 100 if t else 0
        print(f"  {model:25s} comply={m['comply']}/{t} ({rate:.1f}%)  refuse={m['refuse']}/{t}")

    # By model — experimental only
    print(f"\nBY MODEL (experimental only):")
    exp_by_model = defaultdict(lambda: {"comply": 0, "refuse": 0})
    for (model, pid), counts in by_model_prompt.items():
        if not pid.startswith("C"):
            exp_by_model[model]["comply"] += counts["comply"]
            exp_by_model[model]["refuse"] += counts["refuse"]
    for model in models:
        m = exp_by_model[model]
        t = m["comply"] + m["refuse"]
        rate = m["comply"] / t * 100 if t else 0
        print(f"  {model:25s} comply={m['comply']}/{t} ({rate:.1f}%)  refuse={m['refuse']}/{t}")

    # By prompt
    print(f"\nBY PROMPT:")
    prompts = sorted(by_prompt.keys(), key=prompt_sort_key)
    for pid in prompts:
        p = by_prompt[pid]
        t = p["comply"] + p["refuse"]
        rate = p["comply"] / t * 100 if t else 0
        label = "ctrl" if pid.startswith("C") else "exp "
        print(f"  {pid:6s} [{label}] comply={p['comply']}/{t} ({rate:.1f}%)  refuse={p['refuse']}/{t}")

    # Full matrix
    print(f"\nMATRIX (comply/total per model x prompt):")
    header = f"  {'':6s}" + "".join(f"{m:>14s}" for m in models)
    print(header)
    for pid in prompts:
        row = f"  {pid:6s}"
        for model in models:
            key = (model, pid)
            if key in by_model_prompt:
                c = by_model_prompt[key]
                t = c["comply"] + c["refuse"]
                row += f"{'':>4s}{c['comply']}/{t:>2d}{'':>5s}"
            else:
                row += f"{'':>6s}-{'':>7s}"
        print(row)


def save_json(total, by_model, by_prompt, by_model_prompt, by_category):
    output = {
        "total": total,
        "by_category": by_category,
        "by_model": dict(by_model),
        "by_prompt": dict(by_prompt),
        "matrix": {
            f"{model}|{pid}": counts
            for (model, pid), counts in sorted(by_model_prompt.items())
        },
    }
    path = os.path.join(DATA_DIR, "aggregate.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Save to data/aggregate.json")
    parser.add_argument("--prompt", nargs="+", help="Only aggregate these prompt IDs (e.g. P11 P12)")
    args = parser.parse_args()

    records = load_all()

    if args.prompt:
        records = [r for r in records if r.get("prompt_id") in args.prompt]

    print(f"Loaded {len(records)} records")

    total, by_model, by_prompt, by_model_prompt, by_category = aggregate(records)
    print_tables(total, by_model, by_prompt, by_model_prompt, by_category)

    if args.json:
        save_json(total, by_model, by_prompt, by_model_prompt, by_category)


if __name__ == "__main__":
    main()
