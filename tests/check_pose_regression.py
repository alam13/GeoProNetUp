#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_last(path: str):
    lines = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"Empty metrics file: {path}")
    return json.loads(lines[-1])


def main():
    parser = argparse.ArgumentParser(description="Regression checks for pose metrics")
    parser.add_argument("--baseline_metrics", required=True)
    parser.add_argument("--candidate_metrics", required=True)
    parser.add_argument("--thresholds_json", required=True)
    args = parser.parse_args()

    baseline = load_last(args.baseline_metrics)
    candidate = load_last(args.candidate_metrics)
    thresholds = json.loads(Path(args.thresholds_json).read_text())

    failures = []
    for metric, cfg in thresholds.items():
        if metric not in baseline or metric not in candidate:
            continue
        base = baseline[metric]
        cand = candidate[metric]
        if base is None or cand is None:
            continue

        mode = cfg.get("mode", "max_drop")
        margin = float(cfg.get("margin", 0.0))

        if mode == "max_drop":
            if cand < base - margin:
                failures.append(f"{metric}: candidate {cand:.6f} < baseline {base:.6f} - {margin}")
        elif mode == "max_increase":
            if cand > base + margin:
                failures.append(f"{metric}: candidate {cand:.6f} > baseline {base:.6f} + {margin}")
        else:
            raise ValueError(f"Unsupported mode for {metric}: {mode}")

    if failures:
        print("Regression check FAILED")
        for row in failures:
            print(f" - {row}")
        raise SystemExit(1)

    print("Regression check PASSED")


if __name__ == "__main__":
    main()
