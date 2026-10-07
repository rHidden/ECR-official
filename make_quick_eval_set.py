"""Built so that we can test on a subset of the real data that takes a fraction of the time to run
    created dataset is saved in gitignored files redial_gen_quick under src_emo/data

    python3 make_quick_eval_set.py --percent 10          # random 10% of the validation and test sets
    python3 make_quick_eval_set.py                       # ~1000 test and ~160 validation recommendations
    python3 make_quick_eval_set.py --test-targets 500    # smaller and faster
"""

import argparse
import json
import os
import random

SRC = os.path.join("src_emo", "data", "redial_gen")
DST = os.path.join("src_emo", "data", "redial_gen_quick")
SPLITS = ["train", "valid", "test"]


def sample_split(split, targets, percent, rng):
    """Random dialogue turns from one split until they hold `targets` recommendations (or `percent` of the split's
    recommendations, if given), kept in file order."""
    with open(os.path.join(SRC, f"{split}_data_processed.jsonl"), encoding="utf-8") as f:
        lines = [line for line in f if json.loads(line).get("rec")]
    if percent is not None:
        targets = round(sum(len(json.loads(line)["rec"]) for line in lines) * percent / 100)
    order = list(range(len(lines)))
    rng.shuffle(order)
    chosen, n = [], 0
    for i in order:
        if n >= targets:
            break
        chosen.append(i)
        n += len(json.loads(lines[i])["rec"])
    return [lines[i] for i in sorted(chosen)], n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-targets", type=int, default=1000, help="test recommendations to keep (full: 4810)")
    ap.add_argument("--valid-targets", type=int, default=160, help="validation recommendations to keep (full: 3733)")
    ap.add_argument("--percent", type=float, help="keep this %% of the validation and test sets (overrides targets)")
    ap.add_argument("--seed", type=int, default=0, help="which random sample")
    args = ap.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if not os.path.isfile(os.path.join(SRC, "test_data_processed.jsonl")):
        raise SystemExit(f"{SRC} has no processed data yet: run the README's preprocessing steps first")
    os.makedirs(DST, exist_ok=True)

    processed = {f"{s}_data_processed.jsonl" for s in SPLITS}
    for name in os.listdir(SRC):
        link = os.path.join(DST, name)
        if name in processed or os.path.lexists(link):
            continue
        os.symlink(os.path.join("..", "redial_gen", name), link)

    rng = random.Random(args.seed)
    budget = {"train": 32, "valid": args.valid_targets, "test": args.test_targets}
    for split in SPLITS:
        lines, n = sample_split(split, budget[split], None if split == "train" else args.percent, rng)
        with open(os.path.join(DST, f"{split}_data_processed.jsonl"), "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"{split:<5} {len(lines):>5} dialogue turns, {n:>5} recommendations")
    print(f"-> {DST}; run train_rec.py with --dataset redial_gen_quick")


if __name__ == "__main__":
    main()
