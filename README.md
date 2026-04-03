mv GeoProNetUp_tmp/pdbbind_rmsd_srand_coor2/test/ GeoProNetUp_tmp/pdbbind_rmsd_srand_coor2/raw/

mv GeoProNetUp_tmp/pdbbind_rmsd_srand_coor2/train/ GeoProNetUp_tmp/pdbbind_rmsd_srand_coor2/raw/

# Then train the protein-ligand pose prediction model with:
python train_coor.py --gpu_id=0 --n_graph_layer=4 --d_graph_layer=256 --start_epoch=1 --epoch=350 --flexible --model_dir=GeoProNetUp_tmp/models_4_256_atom_hinge0 --data_path=GeoProNetUp_tmp/pdbbind_rmsd_srand_coor2 --heads=1 --batch_size=1 --model_type=Net_coor --residue --edge_dim=3 --loss_reduction=mean --output=GeoProNetUp_tmp/output_4_256_atom_hinge0 --hinge=0 --tot_seed=1




## GeoProNet novelty switches (paper-aligned)

Generate data with geometric novelty (direction + RBF + torsion-aware edge signals):

```bash
python convert_data_to_disk.py --cv=0 --input_list=data/pdb_list_ --output_file=pdbbind_rmsd_srand_coor2 --thread_num=1 --use_new_data --bond_th=6 --pocket_th=12 --groundtruth_dir=data/pdbbind/ --pdbbind_dir=data/medusadock_output --label_list_file=GeoProNetUp_tmp --dataset=coor2 --pdb_version=2016 --use_novel_features
```

Train with decomposition-style objective (alignment-normalized + torsion/steric constraints):

```bash
python train_coor.py --gpu_id=0 --n_graph_layer=4 --d_graph_layer=256 --start_epoch=1 --epoch=350 --flexible --model_dir=GeoProNetUp_tmp/models_4_256_atom_hinge0 --data_path=GeoProNetUp_tmp/pdbbind_rmsd_srand_coor2 --heads=1 --batch_size=1 --model_type=Net_coor --residue --edge_dim=13 --loss_reduction=mean --output=GeoProNetUp_tmp/output_4_256_atom_hinge0 --hinge=0 --tot_seed=1 --lambda_align=0.5 --lambda_coord=0.5 --lambda_steric=0.05 --lambda_torsion=0.02 --use_novel_features
```

To enable **adaptive multi-objective balancing** (learned uncertainty-style loss weights):

```bash
python train_coor.py --gpu_id=0 --n_graph_layer=4 --d_graph_layer=256 --start_epoch=1 --epoch=350 --flexible --model_dir=GeoProNetUp_tmp/models_adaptive --data_path=GeoProNetUp_tmp/pdbbind_rmsd_srand_coor2 --heads=1 --batch_size=1 --model_type=Net_coor --residue --edge_dim=13 --loss_reduction=mean --output=GeoProNetUp_tmp/output_adaptive --tot_seed=1 --lambda_align=0.5 --lambda_coord=0.5 --lambda_steric=0.05 --lambda_torsion=0.02 --lambda_dihedral=0.1 --use_novel_features --adaptive_loss_weights
```

Gap analysis + holistic upgrade blueprint is documented in `docs/pose_upgrade_roadmap.md`.
Follow-up deep audit and remaining upgrade priorities are documented in `docs/deep_upgrade_audit.md`.

## Modular trainer (config-driven, single entrypoint)

Use the new modular training stack (`engine/train_loop.py`, `engine/eval_loop.py`, `losses/geometric_losses.py`, `models/factory.py`) with one config file:

```bash
python scripts/train_from_config.py --config configs/modular_train_example.json
```

## Implementation (modular ablation + seed-robust reporting)

Run standardized Phase-1 feature ablations (baseline vs novel vs novel+alpha):

```bash
python scripts/run_phase1_ablation.py --data_path=GeoProNetUp_tmp/pdbbind_rmsd_srand_coor2 --output_root=results/phase1 --epochs=50 --seeds 1 2 3 --tot_seed=1
```

Add `--adaptive_loss_weights` to test adaptive reweighting in the same matrix.

Summarize arbitrary multi-seed runs (mean/std + bootstrap CI):

```bash
python scripts/summarize_seed_metrics.py --metrics_files results/phase1/novel_edges/seed_1/metrics.jsonl results/phase1/novel_edges/seed_2/metrics.jsonl results/phase1/novel_edges/seed_3/metrics.jsonl --output_json results/phase1/novel_edges_summary.json --output_md results/phase1/novel_edges_summary.md --title "Novel edges seed summary"
```

## Implementation (coarse-to-fine + neighborhood schedule + ranking head)

Train the new two-stage model with radius curriculum and ranking auxiliary loss:

```bash
python train_coor.py --gpu_id=0 --n_graph_layer=4 --d_graph_layer=256 --start_epoch=1 --epoch=350 --flexible --model_dir=GeoProNetUp_tmp/models_two_stage --data_path=GeoProNetUp_tmp/pdbbind_rmsd_srand_coor2 --heads=1 --batch_size=1 --model_type=Net_coor_two_stage --residue --edge_dim=13 --loss_reduction=mean --output=GeoProNetUp_tmp/output_two_stage --tot_seed=1 --lambda_align=0.5 --lambda_coord=0.5 --lambda_steric=0.05 --lambda_torsion=0.02 --lambda_dihedral=0.1 --lambda_rank=0.1 --rank_good_th=2.0 --radius_start=0.06 --radius_end=0.12 --use_novel_features --metrics_file=GeoProNetUp_tmp/metrics_two_stage.jsonl
```

If `data.pose_rmsd` is provided by preprocessing, ranking supervision uses those true per-pose RMSD labels; otherwise it falls back to displacement-derived labels.

## Implementation (OOD protocol + inference package + regression suite)

Generate OOD split manifests (family/scaffold proxies):

```bash
python scripts/create_ood_manifest.py --data_root=GeoProNetUp_tmp/pdbbind_rmsd_srand_coor2 --split=train --heldout_bucket=0 --protein_family_csv=configs/phase3/protein_family_annotations.csv --scaffold_csv=configs/phase3/scaffold_annotations.csv --output_json=results/phase3/ood_train_manifest.json
```

Use the inference package entrypoint (`inference/PoseInferencePipeline`) for deterministic checkpoint inference.

Run regression gates against a known baseline:

```bash
python tests/check_pose_regression.py --baseline_metrics=results/baseline/metrics.jsonl --candidate_metrics=results/candidate/metrics.jsonl --thresholds_json=configs/phase3/regression_thresholds.json
```

Generate a versioned baseline snapshot (model hash + data id) for CI:

```bash
python scripts/make_baseline_snapshot.py --metrics_jsonl=results/baseline/metrics.jsonl --model_file=results/baseline/best_model.pt --data_id=golden_pose_set_v1 --output_dir=ci/baselines
```

Run CI-style regression gate (strict pass/fail):

```bash
python scripts/ci_regression_gate.py --baseline_snapshot=ci/baselines/baseline_example.json --candidate_metrics_jsonl=ci/golden/candidate_metrics.jsonl --thresholds_json=configs/phase3/regression_thresholds.json
```

For hash/data alignment checks, also pass `--candidate_model_file` and `--data_id`.

Ranking calibration metrics are also reported: `rank_ece` and `rank_brier`.


For explicit torsion updates with rotatable-bond supervision, use:

```bash
python train_coor.py --gpu_id=0 --n_graph_layer=4 --d_graph_layer=256 --start_epoch=1 --epoch=350 --flexible --model_dir=GeoProNetUp_tmp/models_torsion --data_path=GeoProNetUp_tmp/pdbbind_rmsd_srand_coor2 --heads=1 --batch_size=1 --model_type=Net_coor_torsion --residue --edge_dim=13 --loss_reduction=mean --output=GeoProNetUp_tmp/output_torsion --tot_seed=1 --lambda_align=0.5 --lambda_coord=0.5 --lambda_steric=0.05 --lambda_torsion=0.02 --lambda_dihedral=0.1 --torsion_iters=2 --use_novel_features
```

To include explicit angle side-channel (`alpha_ijk`) in message passing, add `--use_alpha_channel` and set `--edge_dim=14`.
