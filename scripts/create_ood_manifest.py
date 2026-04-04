#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
import sys
import csv

sys.path.append(str(Path(__file__).resolve().parents[1]))


def _hash_bucket(key: str, num_buckets: int = 10) -> int:
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % num_buckets


def _pdb_code(sample):
    pdb = getattr(sample, "pdb", "unknown")
    if isinstance(pdb, (list, tuple)):
        pdb = pdb[0]
    if isinstance(pdb, bytes):
        pdb = pdb.decode("utf-8")
    return str(pdb)


def _family_proxy(pdb_code: str) -> str:
    return pdb_code[:2] if len(pdb_code) >= 2 else pdb_code


def _scaffold_proxy(sample) -> str:
    flex = int(sample.flexible_idx.sum().item()) if hasattr(sample, "flexible_idx") else 0
    bonds = int(sample.bonds.shape[0]) if hasattr(sample, "bonds") else 0
    return f"flex{flex}_bond{bonds}"


def assign_split(key: str, heldout_bucket: int) -> str:
    return "ood_test" if _hash_bucket(key) == heldout_bucket else "id_train"


def _load_map_file(path, key_col, value_col):
    if path is None:
        return {}
    out = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = str(row.get(key_col, "")).strip()
            v = str(row.get(value_col, "")).strip()
            if k and v:
                out[k] = v
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate OOD split manifests (family/scaffold proxies)")
    parser.add_argument("--data_root", required=True, help="Dataset root path used by PDBBindCoor")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--heldout_bucket", type=int, default=0)
    parser.add_argument("--protein_family_csv", default=None, help="CSV with columns: pdb,protein_family")
    parser.add_argument("--scaffold_csv", default=None, help="CSV with columns: pdb,scaffold_id")
    parser.add_argument("--strict_annotations", action="store_true", help="fail if annotation is missing")
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    try:
        from dataset import PDBBindCoor
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing runtime dependency while loading dataset. "
            "Install project training dependencies (torch + torch_geometric) before running this command."
        ) from exc

    ds = PDBBindCoor(root=args.data_root, split=args.split)
    family_map = _load_map_file(args.protein_family_csv, "pdb", "protein_family")
    scaffold_map = _load_map_file(args.scaffold_csv, "pdb", "scaffold_id")
    manifest = {
        "source": args.data_root,
        "dataset_split": args.split,
        "heldout_bucket": args.heldout_bucket,
        "annotation_mode": "true_annotations" if (family_map or scaffold_map) else "proxy_fallback",
        "splits": [],
    }

    for idx in range(len(ds)):
        sample = ds[idx]
        pdb = _pdb_code(sample)
        family_key = family_map.get(pdb, _family_proxy(pdb))
        scaffold_key = scaffold_map.get(pdb, _scaffold_proxy(sample))
        if args.strict_annotations and (pdb not in family_map or pdb not in scaffold_map):
            raise ValueError(f"Missing protein family/scaffold annotation for pdb '{pdb}'")
        manifest["splits"].append(
            {
                "index": idx,
                "pdb": pdb,
                "protein_family": family_key,
                "family_split": assign_split(family_key, args.heldout_bucket),
                "scaffold_id": scaffold_key,
                "scaffold_split": assign_split(scaffold_key, args.heldout_bucket),
            }
        )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(manifest['splits'])} rows to {args.output_json}")


if __name__ == "__main__":
    main()
