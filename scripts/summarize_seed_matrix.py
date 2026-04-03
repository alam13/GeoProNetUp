#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from engine.seed_report import summarize_seed_runs, summary_to_json, write_markdown_summary


def main():
    parser = argparse.ArgumentParser(description="Summarize multi-seed metrics jsonl files")
    parser.add_argument("--metrics_files", nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--title", default="Seed summary")
    args = parser.parse_args()

    summary = summarize_seed_runs(args.metrics_files)
    with open(args.output_json, "w") as f:
        json.dump(summary_to_json(summary), f, indent=2)
        f.write("\n")
    write_markdown_summary(summary, args.output_md, title=args.title)


if __name__ == "__main__":
    main()
