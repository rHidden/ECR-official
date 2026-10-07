"""Collects the results of train_rec.py runs into one table.
"""

import argparse
import csv
import glob
import os
import re

METRICS = [("R@1", "recall@1"), ("R@10", "recall@10"), ("R@50", "recall@50"),
           ("RT@1", "recall_true@1"), ("RT@10", "recall_true@10"), ("RT@50", "recall_true@50"),
           ("AUC", "auc"), ("NDCG@10", "ndcg@10"), ("MRR@10", "mrr@10"), ("loss", "loss")]
COLUMNS = ["started", "run", "mode", "seed", "dataset", "train_examples", "epoch", "split"] + [m for m, _ in METRICS] + ["best"]


def setting(text, key):
    m = re.search(rf"'{key}': '?([^,'}}]*)'?", text)
    return m.group(1) if m else ""


def parse_log(path):
    """One dict per (epoch, split) with the run's settings and metrics, in log order."""
    text = open(path, encoding="utf-8", errors="replace").read()
    mode = "eval" if setting(text, "test") == "True" else "train"
    examples = re.search(r"Num examples = (\d+)", text)
    run = {
        "started": os.path.basename(path)[:-len(".log")] if re.match(r"\d{4}-", os.path.basename(path)) else "",
        "run": setting(text, "output_dir").rstrip("/").split("/")[-1],
        "mode": mode,
        "seed": setting(text, "seed"),
        "dataset": setting(text, "dataset"),
        "train_examples": examples.group(1) if mode == "train" and examples else "",
    }
    rows, best_epoch, last_valid_epoch = [], None, None
    for line in text.splitlines():
        if "new best model" in line:
            best_epoch = last_valid_epoch
        report = re.search(r"\{'(valid|test)/.*\}", line)
        if not report:
            continue
        split = report.group(1)
        values = dict(re.findall(rf"'{split}/([a-z_@0-9]+)': ([-0-9.e+]+)", report.group(0)))
        epoch = int(setting(report.group(0), "epoch") or 0)
        if split == "valid":
            last_valid_epoch = epoch
        row = dict(run, epoch=epoch + 1 if mode == "train" else "", split=split, _epoch=epoch)
        row.update({name: f"{float(values[key]):.4f}" if key in values else "" for name, key in METRICS})
        rows.append(row)
    for row in rows:
        row["best"] = "yes" if mode == "train" and row.pop("_epoch") == best_epoch else ""
        row.pop("_epoch", None)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="*", help="log files (default: src_emo/log/*.log)")
    ap.add_argument("--out", default=os.path.join("src_emo", "save", "results"), help="output path without extension")
    args = ap.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    paths = args.logs or sorted(glob.glob(os.path.join("src_emo", "log", "*.log")))
    rows = [row for path in paths for row in parse_log(path)]
    if not rows:
        raise SystemExit("no results found in: " + ", ".join(paths or ["src_emo/log/ (empty)"]))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    table = [COLUMNS] + [[str(row[c]) for c in COLUMNS] for row in rows]
    with open(args.out + ".md", "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(table[0]) + " |\n|" + "|".join("---" for _ in COLUMNS) + "|\n")
        f.writelines("| " + " | ".join(r) + " |\n" for r in table[1:])

    widths = [max(len(r[i]) for r in table) for i in range(len(COLUMNS))]
    for r in table:
        print("  ".join(cell.ljust(w) for cell, w in zip(r, widths)))
    runs = len({(row["started"], row["run"]) for row in rows})
    print(f"\n{runs} run(s), {len(rows)} rows -> {args.out}.csv and {args.out}.md")


if __name__ == "__main__":
    main()
