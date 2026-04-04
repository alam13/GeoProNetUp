#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_last_jsonl(path):
    rows = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    if not rows:
        raise ValueError(f"Empty metrics file: {path}")
    return json.loads(rows[-1])


def _resolve_optional_artifact(path: str, pattern: str, label: str):
    requested = Path(path)
    if requested.exists():
        return requested

    search_roots = []
    parent = requested.parent if str(requested.parent) else Path(".")
    search_roots.append(parent)
    project_results = Path("results")
    if project_results != parent and project_results.exists():
        search_roots.append(project_results)

    candidates = []
    for root in search_roots:
        candidates.extend(root.glob(f"**/{pattern}"))
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        chosen = candidates[0]
        print(f"warning: {label} not found at {requested}; using latest discovered file: {chosen}")
        return chosen

    print(f"warning: {label} not found at {requested}; skipping optional hash alignment checks")
    return None


def _check(candidate, baseline, thresholds):
    failures = []
    for metric, cfg in thresholds.items():
        if metric not in candidate or metric not in baseline:
            continue
        cand = candidate[metric]
        base = baseline[metric]
        if cand is None or base is None:
            continue
        mode = cfg.get("mode", "max_drop")
        margin = float(cfg.get("margin", 0.0))
        if mode == "max_drop" and cand < base - margin:
            failures.append(f"{metric}: {cand:.6f} < {base:.6f} - {margin}")
        if mode == "max_increase" and cand > base + margin:
            failures.append(f"{metric}: {cand:.6f} > {base:.6f} + {margin}")
    return failures


def main():
    p = argparse.ArgumentParser(description="CI regression gate using versioned baseline snapshot")
    p.add_argument("--baseline_snapshot", required=True)
    p.add_argument("--candidate_metrics_jsonl", required=True)
    p.add_argument("--thresholds_json", required=True)
    p.add_argument("--candidate_model_file", default=None, help="Optional model file for version/hash alignment checks")
    p.add_argument("--data_id", default=None, help="Optional dataset id for version/hash alignment checks")
    args = p.parse_args()

    baseline_payload = json.loads(Path(args.baseline_snapshot).read_text())
    baseline = baseline_payload["metrics"]
    baseline_version = baseline_payload.get("version")
    baseline_data_id = baseline_payload.get("data_id")
    baseline_model_hash = baseline_payload.get("model_sha256")
    candidate = _read_last_jsonl(args.candidate_metrics_jsonl)
    thresholds = json.loads(Path(args.thresholds_json).read_text())

    if args.data_id is not None and baseline_data_id is not None and args.data_id != baseline_data_id:
        raise SystemExit(f"data_id mismatch: baseline={baseline_data_id}, candidate={args.data_id}")

    if args.candidate_model_file is not None:
        candidate_model_path = _resolve_optional_artifact(args.candidate_model_file, "best_model.pt", "candidate model file")
        if candidate_model_path is None:
            candidate_model_hash = None
        else:
            candidate_model_hash = _sha256_file(candidate_model_path)
        if baseline_model_hash not in (None, "example") and candidate_model_hash != baseline_model_hash:
            print(f"warning: model hash differs from baseline ({candidate_model_hash} != {baseline_model_hash})")
        if args.data_id is not None and baseline_version is not None and candidate_model_hash is not None:
            expected = hashlib.sha256(f"{args.data_id}:{candidate_model_hash}".encode("utf-8")).hexdigest()[:16]
            if baseline_version != expected:
                print(f"warning: baseline version key ({baseline_version}) does not match candidate hash/data_id ({expected})")

    failures = _check(candidate, baseline, thresholds)
    if failures:
        print("CI regression gate FAILED")
        for row in failures:
            print(f" - {row}")
        raise SystemExit(1)

    print("CI regression gate PASSED")


if __name__ == "__main__":
    main()
