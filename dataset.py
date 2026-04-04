import os
import ast
import glob
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, Data

SPACE = 100
BOND_TH = 6.0


def _row_idx_from_node_index(node_index, num_edges):
    if node_index.numel() == 0:
        return torch.empty((0,), dtype=torch.long)

    counts = torch.empty_like(node_index)
    counts[0] = node_index[0]
    if node_index.numel() > 1:
        counts[1:] = node_index[1:] - node_index[:-1]

    rows = torch.arange(node_index.numel(), dtype=torch.long)
    row_idx = torch.repeat_interleave(rows, counts.clamp(min=0))

    if row_idx.numel() != num_edges:
        if row_idx.numel() > num_edges:
            row_idx = row_idx[:num_edges]
        else:
            pad = torch.full((num_edges - row_idx.numel(),), rows[-1], dtype=torch.long)
            row_idx = torch.cat([row_idx, pad], dim=0)
    return row_idx


class PDBBindCoor(InMemoryDataset):

    def __init__(self, root, subset=False, split='train', data_type='coor2',
                 transform=None, pre_transform=None, pre_filter=None):
        self.subset = subset
        self.split = split
        self.data_type = data_type
        super().__init__(root, transform, pre_transform, pre_filter)

        path = os.path.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path, weights_only=False)

    @property
    def raw_file_names(self):
        return [self.split]

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        return [self.split + '.pt']

    def download(self):
        pass

    def process(self):

        split = self.split
        dataset_dir = os.path.join(self.raw_dir, split)

        files_num = len(glob.glob(os.path.join(dataset_dir, '*_data-G.json')))

        data_list = []
        pbar = tqdm(total=files_num)
        pbar.set_description(f'Processing {split} dataset')

        for f in range(files_num):

            with open(os.path.join(dataset_dir, f'{f}_data-G.json')) as gf:
                graphs = gf.readlines()

            num_graphs_per_file = len(graphs) // 3
            pbar.total = num_graphs_per_file * files_num
            pbar.refresh()

            feat_file = open(os.path.join(dataset_dir, f'{f}_data-feats'), 'rb')
            label_file = open(os.path.join(dataset_dir, f'{f}_label'), 'rb')

            for idx in range(num_graphs_per_file):

                features = np.load(feat_file)

                indptr = ast.literal_eval(graphs[3 * idx])
                indices = ast.literal_eval(graphs[3 * idx + 1])
                dist = ast.literal_eval(graphs[3 * idx + 2])

                flexible_len = np.load(label_file)
                labels = np.load(label_file)
                bonds = np.load(label_file)
                pdb = np.load(label_file)

                indptr = torch.LongTensor(indptr)
                indices = torch.LongTensor(indices)
                dist = torch.tensor(dist, dtype=torch.float)

                row_idx = _row_idx_from_node_index(indptr, len(indices))
                edge_index = torch.stack((row_idx, indices), dim=0)

                x = torch.tensor(features, dtype=torch.float)

                # ⚠️ ORIGINAL BUG SOURCE
                # bonds could be string → crash here
                bonds = torch.tensor(bonds)  

                dist = dist.reshape(dist.size()[0], -1)

                flexible_idx = torch.arange(features.shape[0]) < flexible_len[0]
                flexible_len = torch.tensor(flexible_len)

                y = torch.tensor(labels, dtype=torch.float)
                y = y - x[flexible_idx, -3:]

                data = Data(x=x, edge_index=edge_index, y=y)
                data.bonds = bonds
                data.dist = dist
                data.pdb = pdb
                data.flexible_idx = flexible_idx
                data.flexible_len = flexible_len

                data_list.append(data)
                pbar.update(1)

            feat_file.close()
            label_file.close()

        pbar.close()

        torch.save(self.collate(data_list),
                   os.path.join(self.processed_dir, f'{split}.pt'))
