#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))


def _as_namespace(payload):
    return SimpleNamespace(**payload)


def main():
    parser = argparse.ArgumentParser(description="Config-driven training entrypoint")
    parser.add_argument("--config", required=True, help="JSON config file")
    args = parser.parse_args()

    import torch
    from torch_geometric.data import DataLoader
    from dataset import PDBBindCoor
    from engine.eval_loop import evaluate
    from engine.train_loop import train_one_epoch
    from losses.geometric_losses import AdaptiveGeoLoss
    from models.factory import build_model

    cfg = _as_namespace(json.loads(Path(args.config).read_text()))
    device = torch.device(cfg.device if hasattr(cfg, "device") else ("cuda" if torch.cuda.is_available() else "cpu"))

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    train_ds = PDBBindCoor(root=cfg.data_path, split="train")
    test_ds = PDBBindCoor(root=cfg.data_path, split="test")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1)

    model = build_model(cfg.model_type, train_ds.num_features, cfg, device)

    adaptive_loss = None
    opt_params = list(model.parameters())
    if getattr(cfg, "adaptive_loss_weights", False):
        adaptive_loss = AdaptiveGeoLoss(
            {
                "align": cfg.lambda_align,
                "coord": cfg.lambda_coord,
                "steric": cfg.lambda_steric,
                "torsion": cfg.lambda_torsion,
                "dihedral": cfg.lambda_dihedral,
            }
        ).to(device)
        opt_params += list(adaptive_loss.parameters())

    optimizer = torch.optim.Adam(opt_params, lr=cfg.lr)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = output_dir / "metrics.jsonl"

    best_rmsd = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, cfg, adaptive_loss=adaptive_loss)
        metrics = evaluate(model, test_loader, device, cfg, adaptive_loss=adaptive_loss)
        metrics.update({"epoch": epoch, "train_loss": train_loss})

        if adaptive_loss is not None:
            for k, p in adaptive_loss.log_vars.items():
                metrics[f"adaptive_weight_{k}"] = float(torch.exp(-p).detach().cpu().item())

        with metrics_file.open("a") as f:
            f.write(json.dumps(metrics) + "\n")

        if metrics["rmsd_mean_A"] < best_rmsd:
            best_rmsd = metrics["rmsd_mean_A"]
            torch.save(model.state_dict(), output_dir / "best_model.pt")

        print(f"Epoch {epoch}: train_loss={train_loss:.6f} rmsd_mean_A={metrics['rmsd_mean_A']:.4f}")


if __name__ == "__main__":
    main()
