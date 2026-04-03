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


def _last_jsonl(path):
    rows = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    if not rows:
        raise ValueError(f"Empty metrics file: {path}")
    return json.loads(rows[-1])


def main():
    p = argparse.ArgumentParser(description="Create versioned baseline metrics snapshot")
    p.add_argument("--metrics_jsonl", required=True)
    p.add_argument("--model_file", required=True)
    p.add_argument("--data_id", required=True, help="Stable dataset identifier (e.g., pdbbind_rmsd_srand_coor2_v1)")
    p.add_argument("--output_dir", default="ci/baselines")
    args = p.parse_args()

    model_hash = _sha256_file(args.model_file)
    baseline_metrics = _last_jsonl(args.metrics_jsonl)
    version_key = hashlib.sha256(f"{args.data_id}:{model_hash}".encode("utf-8")).hexdigest()[:16]

    payload = {
        "version": version_key,
        "data_id": args.data_id,
        "model_sha256": model_hash,
        "metrics": baseline_metrics,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"baseline_{version_key}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(out_path)


if __name__ == "__main__":
    main()
