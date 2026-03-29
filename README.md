# To train the model by your self, you need to first generated the dataset:
python convert_data_to_disk.py --cv=0 --input_list=data/pdb_list_ --output_file=pdbbind_rmsd_srand_coor2 --thread_num=1 --use_new_data --bond_th=6 --pocket_th=12 --groundtruth_dir=data/pdbbind/ --pdbbind_dir=data/medusadock_output --label_list_file=KtransPose_tmp --dataset=coor2 --pdb_version=2016

mkdir KtransPose_tmp/pdbbind_rmsd_srand_coor2/raw

mv KtransPose_tmp/pdbbind_rmsd_srand_coor2/test/ KtransPose_tmp/pdbbind_rmsd_srand_coor2/raw/

mv KtransPose_tmp/pdbbind_rmsd_srand_coor2/train/ KtransPose_tmp/pdbbind_rmsd_srand_coor2/raw/

# Then train the protein-ligand pose prediction model with:
python train_coor.py --gpu_id=0 --n_graph_layer=4 --d_graph_layer=256 --start_epoch=1 --epoch=350 --flexible --model_dir=KtransPose_tmp/models_4_256_atom_hinge0 --data_path=KtransPose_tmp/pdbbind_rmsd_srand_coor2 --heads=1 --batch_size=1 --model_type=Net_coor --residue --edge_dim=3 --loss_reduction=mean --output=KtransPose_tmp/output_4_256_atom_hinge0 --hinge=0 --tot_seed=1




## GeoProNet novelty switches (paper-aligned)

Generate data with geometric novelty (direction + RBF + torsion-aware edge signals):

```bash
python convert_data_to_disk.py --cv=0 --input_list=data/pdb_list_ --output_file=pdbbind_rmsd_srand_coor2 --thread_num=1 --use_new_data --bond_th=6 --pocket_th=12 --groundtruth_dir=data/pdbbind/ --pdbbind_dir=data/medusadock_output --label_list_file=KtransPose_tmp --dataset=coor2 --pdb_version=2016 --use_novel_features
```

Train with decomposition-style objective (alignment-normalized + torsion/steric constraints):

```bash
python train_coor.py --gpu_id=0 --n_graph_layer=4 --d_graph_layer=256 --start_epoch=1 --epoch=350 --flexible --model_dir=KtransPose_tmp/models_4_256_atom_hinge0 --data_path=KtransPose_tmp/pdbbind_rmsd_srand_coor2 --heads=1 --batch_size=1 --model_type=Net_coor --residue --edge_dim=13 --loss_reduction=mean --output=KtransPose_tmp/output_4_256_atom_hinge0 --hinge=0 --tot_seed=1 --lambda_align=0.5 --lambda_coord=0.5 --lambda_steric=0.05 --lambda_torsion=0.02 --use_novel_features
```


For explicit torsion updates with rotatable-bond supervision, use:

```bash
python train_coor.py --gpu_id=0 --n_graph_layer=4 --d_graph_layer=256 --start_epoch=1 --epoch=350 --flexible --model_dir=KtransPose_tmp/models_torsion --data_path=KtransPose_tmp/pdbbind_rmsd_srand_coor2 --heads=1 --batch_size=1 --model_type=Net_coor_torsion --residue --edge_dim=13 --loss_reduction=mean --output=KtransPose_tmp/output_torsion --tot_seed=1 --lambda_align=0.5 --lambda_coord=0.5 --lambda_steric=0.05 --lambda_torsion=0.02 --lambda_dihedral=0.1 --torsion_iters=2 --use_novel_features
