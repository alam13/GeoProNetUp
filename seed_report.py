import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DEFAULT_METRICS = [
    "rmsd_mean_A",
    "rmsd_median_A",
    "sr_2a",
    "sr_5a",
    "clash_rate",
    "torsion_mae_rad",
    "torsion_angular_rmse_rad",
]


@dataclass
class MetricSummary:
    metric: str
    n: int
    mean: float
    std: float
    ci_low: float
    ci_high: float


def _latest_payload(metrics_file: Path) -> Dict:
    lines = [line.strip() for line in metrics_file.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Metrics file is empty: {metrics_file}")
    return json.loads(lines[-1])


def _bootstrap_ci(values: List[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 42) -> List[float]:
    rng = random.Random(seed)
    if len(values) == 1:
        return [values[0], values[0]]
    boots = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(0, len(values))] for _ in range(len(values))]
        boots.append(sum(sample) / len(sample))
    boots.sort()
    low_idx = max(0, int((alpha / 2.0) * (n_boot - 1)))
    high_idx = min(n_boot - 1, int((1.0 - alpha / 2.0) * (n_boot - 1)))
    return [boots[low_idx], boots[high_idx]]


def summarize_seed_runs(
    metrics_files: Iterable[str],
    metrics: Optional[Iterable[str]] = None,
    bootstrap_samples: int = 2000,
    alpha: float = 0.05,
    bootstrap_seed: int = 42,
) -> Dict[str, MetricSummary]:
    metric_names = list(metrics) if metrics is not None else DEFAULT_METRICS
    payloads = [_latest_payload(Path(f)) for f in metrics_files]

    summary: Dict[str, MetricSummary] = {}
    for metric in metric_names:
        vals = [p[metric] for p in payloads if metric in p and p[metric] is not None]
        if not vals:
            continue
        mean = float(sum(vals) / len(vals))
        std = float(math.sqrt(sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)))
        ci_low, ci_high = _bootstrap_ci(vals, n_boot=bootstrap_samples, alpha=alpha, seed=bootstrap_seed)
        summary[metric] = MetricSummary(
            metric=metric,
            n=len(vals),
            mean=mean,
            std=std,
            ci_low=float(ci_low),
            ci_high=float(ci_high),
        )
    return summary


def write_markdown_summary(summary: Dict[str, MetricSummary], output_file: str, title: str) -> None:
    lines = [f"# {title}", "", "| Metric | N | Mean | Std | 95% CI |", "|---|---:|---:|---:|---:|"]
    for metric in sorted(summary.keys()):
        s = summary[metric]
        lines.append(
            f"| {metric} | {s.n} | {s.mean:.6f} | {s.std:.6f} | [{s.ci_low:.6f}, {s.ci_high:.6f}] |"
        )
    Path(output_file).write_text("\n".join(lines) + "\n")


def summary_to_json(summary: Dict[str, MetricSummary]) -> Dict[str, Dict]:
    return {
        k: {
            "n": v.n,
            "mean": v.mean,
            "std": v.std,
            "ci_low": v.ci_low,
            "ci_high": v.ci_high,
        }
        for k, v in summary.items()
    }
