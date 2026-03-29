import importlib.util
import json
import os
import tempfile
import unittest


class TestNovelStack(unittest.TestCase):
    def test_train_has_fail_fast_check(self):
        with open('train_coor.py', 'r') as f:
            txt = f.read()
        self.assertIn('expects --edge_dim=13', txt)
        self.assertIn('expects --edge_dim=14', txt)

    def test_edge_dims_baseline_and_novel(self):
        if importlib.util.find_spec('numpy') is None:
            self.skipTest('numpy not installed in environment')
        from data_utils import gen_3D_2_pose_atomwise

        Atoms = ['N', 'C', 'O', 'S', 'Br', 'Cl', 'P', 'F', 'I']
        Bonds = ['1', '2', 'ar', 'am']
        protein = [('CA', 0.0, 0.0, 0.0, 'C', 1), ('CA', 2.0, 0.0, 0.0, 'C', 2)]
        ligand = [('C1', 0.5, 0.0, 0.0, 'C'), ('C2', 1.0, 0.0, 0.0, 'C'), ('O1', 1.5, 0.5, 0.0, 'O')]
        edge_gt = {(0, 1), (1, 0), (1, 2), (2, 1)}

        with tempfile.TemporaryDirectory() as d:
            fp = os.path.join(d, 'a')
            gen_3D_2_pose_atomwise(protein, ligand, Atoms, Bonds, edge_gt, 6, fp, use_novel_features=False)
            dist = json.loads(open(fp + '_data-G.json').read().splitlines()[2])
            self.assertEqual(len(dist[0]), 3)

            fp = os.path.join(d, 'b')
            gen_3D_2_pose_atomwise(protein, ligand, Atoms, Bonds, edge_gt, 6, fp, use_novel_features=True)
            dist = json.loads(open(fp + '_data-G.json').read().splitlines()[2])
            self.assertEqual(len(dist[0]), 13)


if __name__ == '__main__':
    unittest.main()
