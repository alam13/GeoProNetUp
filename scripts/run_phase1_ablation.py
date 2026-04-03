#!/usr/bin/env python3
import argparse
import json
import shlex
import subprocess
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from engine.seed_report import summarize_seed_runs, summary_to_json, write_markdown_summary


ABLATIONS = {
    "baseline_edges": {
        "edge_dim": 3,
        "flags": [],
    },
    "novel_edges": {
        "edge_dim": 13,
        "flags": ["--use_novel_features"],
    },
    "novel_plus_alpha": {
        "edge_dim": 14,
        "flags": ["--use_novel_features", "--use_alpha_channel"],
    },
}


def _run_command(cmd, dry_run=False):
    pretty = " ".join(shlex.quote(c) for c in cmd)
    print(f"[phase1] {pretty}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Phase-1 ablation runner for GeoProNetUp")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_root", default="results/phase1")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--model_type", default="Net_coor")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--tot_seed", type=int, default=1, help="dataset grouping seed count expected by evaluator")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--adaptive_loss_weights", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    aggregate = {}
    for ablation_name, cfg in ABLATIONS.items():
        metrics_files = []
        ablation_dir = output_root / ablation_name
        ablation_dir.mkdir(parents=True, exist_ok=True)

        for seed in args.seeds:
            run_dir = ablation_dir / f"seed_{seed}"
            model_dir = run_dir / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            metrics_file = run_dir / "metrics.jsonl"
            metrics_files.append(str(metrics_file))

            cmd = [
                "python",
                "train_coor.py",
                f"--gpu_id={args.gpu_id}",
                f"--epoch={args.epochs}",
                "--start_epoch=1",
                "--flexible",
                f"--model_dir={model_dir}",
                f"--data_path={args.data_path}",
                "--heads=1",
                f"--batch_size={args.batch_size}",
                f"--model_type={args.model_type}",
                "--residue",
                f"--edge_dim={cfg['edge_dim']}",
                "--loss_reduction=mean",
                f"--output={run_dir / 'train.log'}",
                f"--tot_seed={args.tot_seed}",
                f"--seed={seed}",
                "--lambda_align=0.5",
                "--lambda_coord=0.5",
                "--lambda_steric=0.05",
                "--lambda_torsion=0.02",
                "--lambda_dihedral=0.1",
                f"--metrics_file={metrics_file}",
            ] + cfg["flags"]

            if args.adaptive_loss_weights:
                cmd.append("--adaptive_loss_weights")

            _run_command(cmd, dry_run=args.dry_run)

        if args.dry_run:
            continue

        summary = summarize_seed_runs(metrics_files)
        aggregate[ablation_name] = summary_to_json(summary)
        write_markdown_summary(
            summary,
            str(output_root / f"summary_{ablation_name}.md"),
            title=f"Phase-1 {ablation_name} summary",
        )

    if not args.dry_run:
        (output_root / "summary_all.json").write_text(json.dumps(aggregate, indent=2) + "\n")
        print(f"[phase1] wrote aggregate summary to {output_root / 'summary_all.json'}")


if __name__ == "__main__":
    main()
