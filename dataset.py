import os
import ast
import argparse
import glob
import torch
import torch.nn.functional as F
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import distance

from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, Data, DataLoader

import time
from model import Net_coor

SPACE = 100
BOND_TH = 6.0

def _read_optional_scalar(np_file_handle):
    """
    Attempt to read one extra scalar payload from label stream.
    Returns None when payload is absent or malformed.
    """
    try:
        arr = np.load(np_file_handle)
    except Exception:
        return None
    arr = np.asarray(arr).reshape(-1)
    if arr.size == 0:
        return None
    try:
        return float(arr[0])
    except Exception:
        return None

def _row_idx_from_node_index(node_index, num_edges):
    """
    Convert per-node cumulative edge counts (without leading 0) to row indices.
    This is a local replacement for torch_sparse dependency calls.
    """
    if node_index.numel() == 0:
        return torch.empty((0,), dtype=torch.long)
    counts = torch.empty_like(node_index)
    counts[0] = node_index[0]
    if node_index.numel() > 1:
        counts[1:] = node_index[1:] - node_index[:-1]
    rows = torch.arange(node_index.numel(), dtype=torch.long)
    row_idx = torch.repeat_interleave(rows, counts.clamp(min=0))
    if row_idx.numel() != num_edges:
        # fallback-safe: trim/pad to expected edge length
        if row_idx.numel() > num_edges:
            row_idx = row_idx[:num_edges]
        else:
            pad = torch.full((num_edges - row_idx.numel(),), rows[-1], dtype=torch.long)
            row_idx = torch.cat([row_idx, pad], dim=0)
    return row_idx


def _compute_edge_alpha(edge_index, coords):
    """Per-edge angle feature alpha_ijk using one local triplet at source node."""
    src = edge_index[0]
    dst = edge_index[1]
    num_edges = src.size(0)
    alpha = torch.zeros((num_edges, 1), dtype=coords.dtype)
    for e in range(num_edges):
        i = src[e].item()
        j = dst[e].item()
        neigh = dst[src == i]
        neigh = neigh[neigh != j]
        if neigh.numel() == 0:
            continue
        k = neigh[0].item()
        v1 = coords[j] - coords[i]
        v2 = coords[k] - coords[i]
        n1 = torch.norm(v1) + 1e-8
        n2 = torch.norm(v2) + 1e-8
        cosang = torch.clamp(torch.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        alpha[e, 0] = torch.acos(cosang)
    return alpha

def _read_coor_label_record(label_file):
    """
    Read one coordinate-label record without drifting file offsets.
    Expected payload count per record is 4, but field order can vary between
    dumps (notably bonds/pdb position).
    """
    payloads = []
    for _ in range(4):
        try:
            payloads.append(np.load(label_file, allow_pickle=True))
        except EOFError:
            break

    if len(payloads) < 3:
        # Caller can stop iterating this file safely.
        return None

    pdb = None
    bonds = None
    flexible_len = None
    labels = None

    for arr in payloads:
        arr_np = np.asarray(arr)
        if arr_np.dtype.kind in {"U", "S", "O"} and arr_np.size > 0 and pdb is None:
            pdb = arr
            continue
        if arr_np.ndim == 2 and arr_np.shape[1] == 2 and bonds is None:
            bonds = arr
            continue
        if flexible_len is None and arr_np.size == 1:
            flexible_len = arr
            continue
        if labels is None:
            labels = arr

    # Safe fallback if heuristic matching is incomplete.
    if flexible_len is None:
        flexible_len = payloads[0]
    if labels is None:
        labels = payloads[1]
    if bonds is None:
        # pick first numeric 2D payload
        for arr in payloads:
            arr_np = np.asarray(arr)
            if arr_np.ndim == 2 and arr_np.dtype.kind in {"i", "u", "f"}:
                bonds = arr
                break
    if pdb is None:
        pdb = payloads[-1]

    return flexible_len, labels, bonds, pdb


class PDBBind(InMemoryDataset):

    def __init__(self, root, subset=False, split='train', transform=None,
                 pre_transform=None, pre_filter=None):
        self.subset = subset
        # super(PDBBind, self).__init__(root, transform, pre_transform, pre_filter)
        super().__init__(root, transform, pre_transform, pre_filter)
        path = os.path.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path, weights_only=False)

    @property
    def raw_file_names(self):
        return [
            'train', 'test'
        ]

    @property
    def processed_dir(self):
        # name = 'subset' if self.subset else 'full'
        # return os.path.join(self.root, name, 'processed')
        return os.path.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        return ['train.pt', 'test.pt']

    # TODO: implement after uploading the entire dataset somewhere
    def download(self):
        print('Hello Download')
        pass

    def process(self):
        for split in ['train', 'test']:

            dataset_dir = os.path.join(self.raw_dir, f'{split}')
            files_num = len(
                glob.glob(os.path.join(dataset_dir, '*_data-G.json')))

            data_list = []
            graph_idx = 0

            pbar = tqdm(total=files_num)
            pbar.set_description(f'Processing {split} dataset')
        

            for f in range(files_num):

                with open(os.path.join(dataset_dir, f'{f}_data-G.json')) as gf:
                    graphs = gf.readlines()
                num_graphs_per_file = len(graphs)//3

                pbar.total = num_graphs_per_file * files_num
                pbar.refresh()

                # features = []
                # with open(os.path.join(dataset_dir, f'{f}_data-feats'), 'rb') as ff:
                #     for _ in range(num_graphs_per_file):
                #         features.append(np.load(ff))

                feat_file = open(os.path.join(dataset_dir, f'{f}_data-feats'), 'rb')

                labels = np.loadtxt(os.path.join(dataset_dir, f'{f}_label'))
                rmsds = labels[:, 1]
                energies = labels[:, 2]
                labels = labels[:, 0]

                for idx in range(num_graphs_per_file):

                    # it's slow because of these!
                    features = np.load(feat_file)
                    indptr = ast.literal_eval(graphs[3*idx])
                    indices = ast.literal_eval(graphs[3*idx+1])
                    dist = ast.literal_eval(graphs[3*idx+2])
                    
                    # using scipy, probably slow
                    # value = np.ones_like(indices)
                    # # add 0 to indptr for scipy csr
                    # num_nodes = len(indptr)
                    # indptr.insert(0, 0)
                    # coo_graph = csr_matrix((value, indices, indptr), shape=(
                    #     num_nodes, num_nodes)).tocoo()
                    # per graph data
                    # edge_index = torch.Tensor(
                    #     [coo_graph.row, coo_graph.col]).to(torch.long)


                    indptr = torch.LongTensor(indptr)
                    indices = torch.LongTensor(indices)
                    dist = torch.tensor(dist, dtype=torch.float)
                    row_idx = _row_idx_from_node_index(indptr, len(indices))
                    edge_index = torch.stack((row_idx, indices), dim=0)
                    
                    # No edge attr to save space
                    # edge_attr = torch.Tensor(coo_graph.data).to(torch.long)
                    
                    # x = torch.Tensor(features[idx])
                    x = torch.Tensor(features)
                    y = torch.LongTensor([labels[idx].astype(int)])
                    dist = dist.reshape(dist.size()[0], 1)
                    # e =

                    data = Data(x=x, edge_index=edge_index, y=y)
                    data.rmsd = torch.tensor([rmsds[idx]], dtype=torch.float)
                    data.energy = torch.tensor([energies[idx]], dtype=torch.float)
                    data.dist = dist

                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    data_list.append(data)
                
                    pbar.update(1)
            
            pbar.close()
            torch.save(self.collate(data_list),
                       os.path.join(self.processed_dir, f'{split}.pt'))


class PDBBind_test(InMemoryDataset):

    def __init__(self, root, subset=False, split='test', transform=None,
                 pre_transform=None, pre_filter=None, protein = None):
        self.subset = subset
        self.protein = protein
        self.protein_dir = os.path.join(root, self.protein)
        # super(PDBBind, self).__init__(root, transform, pre_transform, pre_filter)
        super().__init__(root, transform, pre_transform, pre_filter)
        path = os.path.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path, weights_only=False)

    @property
    def raw_file_names(self):
        return [
            # 'train', 'test'
            'test'
        ]

    @property
    def processed_dir(self):
        # name = 'subset' if self.subset else 'full'
        # return os.path.join(self.root, name, 'processed')
        return os.path.join(self.protein_dir, 'processed')

    @property
    def processed_file_names(self):
        # return ['train.pt', 'test.pt']
        return ['test.pt']

    # TODO: implement after uploading the entire dataset somewhere
    def download(self):
        # print('Hello Download')
        pass

    def process(self):

        dataset_dir = self.protein_dir
        files_num = 1

        data_list = []
        graph_idx = 0

        pbar = tqdm(total=files_num)
        pbar.set_description(f'Processing {self.protein} dataset')
        

        with open(os.path.join(dataset_dir, f'{self.protein}_data-G.json')) as gf:
            graphs = gf.readlines()
        num_graphs_per_file = len(graphs)//3

        pbar.total = num_graphs_per_file * files_num
        pbar.refresh()

        # features = []
        # with open(os.path.join(dataset_dir, f'{f}_data-feats'), 'rb') as ff:
        #     for _ in range(num_graphs_per_file):
        #         features.append(np.load(ff))

        feat_file = open(os.path.join(dataset_dir, f'{self.protein}_data-feats'), 'rb')

        labels = np.loadtxt(os.path.join(dataset_dir, f'{self.protein}_label'))
        rmsds = labels[:, 1]
        energies = labels[:, 2]
        labels = labels[:, 0]

        for idx in range(num_graphs_per_file):

            # it's slow because of these!
            features = np.load(feat_file)
            indptr = ast.literal_eval(graphs[3*idx])
            indices = ast.literal_eval(graphs[3*idx+1])
            dist = ast.literal_eval(graphs[3*idx+2])
            
            # using scipy, probably slow
            # value = np.ones_like(indices)
            # # add 0 to indptr for scipy csr
            # num_nodes = len(indptr)
            # indptr.insert(0, 0)
            # coo_graph = csr_matrix((value, indices, indptr), shape=(
            #     num_nodes, num_nodes)).tocoo()
            # per graph data
            # edge_index = torch.Tensor(
            #     [coo_graph.row, coo_graph.col]).to(torch.long)


            indptr = torch.LongTensor(indptr)
            indices = torch.LongTensor(indices)
            dist = torch.tensor(dist, dtype=torch.float)
            row_idx = _row_idx_from_node_index(indptr, len(indices))
            edge_index = torch.stack((row_idx, indices), dim=0)
            
            # No edge attr to save space
            # edge_attr = torch.Tensor(coo_graph.data).to(torch.long)
            
            # x = torch.Tensor(features[idx])
            x = torch.Tensor(features)
            y = torch.LongTensor([labels[idx].astype(int)])
            dist = dist.reshape(dist.size()[0], 1)
            # e =

            data = Data(x=x, edge_index=edge_index, y=y)
            data.rmsd = torch.tensor([rmsds[idx]], dtype=torch.float)
            data.energy = torch.tensor([energies[idx]], dtype=torch.float)
            data.dist = dist

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
        
            pbar.update(1)
        
        pbar.close()
        torch.save(self.collate(data_list),
                   os.path.join(self.processed_dir, 'test.pt'))


class PDBBind(InMemoryDataset):

    def __init__(self, root, subset=False, split='train', transform=None,
                 pre_transform=None, pre_filter=None):
        self.subset = subset
        # super(PDBBind, self).__init__(root, transform, pre_transform, pre_filter)
        super().__init__(root, transform, pre_transform, pre_filter)
        path = os.path.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path, weights_only=False)

    @property
    def raw_file_names(self):
        return [
            'train', 'test'
        ]

    @property
    def processed_dir(self):
        # name = 'subset' if self.subset else 'full'
        # return os.path.join(self.root, name, 'processed')
        return os.path.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        return ['train.pt', 'test.pt']
class PDBBindCoor(InMemoryDataset):

                pbar.total = num_graphs_per_file * files_num
                pbar.refresh()

                # features = []
                # with open(os.path.join(dataset_dir, f'{f}_data-feats'), 'rb') as ff:
                #     for _ in range(num_graphs_per_file):
                #         features.append(np.load(ff))

                feat_file = open(os.path.join(dataset_dir, f'{f}_data-feats'), 'rb')
                label_file = open(os.path.join(dataset_dir, f'{f}_label'), 'rb')

                # labels = np.loadtxt(os.path.join(dataset_dir, f'{f}_label'))
                # rmsds = labels[:, 1]
                # energies = labels[:, 2]
                # labels = labels[:, 0]

                for idx in range(num_graphs_per_file):

                    # it's slow because of these!
                    features = np.load(feat_file)
                    # features[:,-3:] *= 100
                    indptr = ast.literal_eval(graphs[3*idx])
                    indices = ast.literal_eval(graphs[3*idx+1])
                    dist = ast.literal_eval(graphs[3*idx+2])
                    
                    record = _read_coor_label_record(label_file)
                    if record is None:
                        break
                    flexible_len, labels, bonds, pdb = record


                    indptr = torch.LongTensor(indptr)
                    indices = torch.LongTensor(indices)
                    # dist[:, :3] *= 10
                    dist = torch.tensor(dist, dtype=torch.float)
                    row_idx = _row_idx_from_node_index(indptr, len(indices))
                    edge_index = torch.stack((row_idx, indices), dim=0)
                    

                    x = torch.Tensor(features)
                    y = torch.Tensor(labels)
                    
                    bonds_np = np.asarray(bonds)
                    if bonds_np.dtype.kind not in {"i", "u", "f"}:
                        raise TypeError(f"Invalid bonds payload dtype {bonds_np.dtype}; check label file field order.")
                    bonds = torch.Tensor(bonds_np)
                    dist = dist.reshape(dist.size()[0], -1)
                    # flexible_idx = torch.tensor([(F.mse_loss(x[i][-3:], y[i]).item() > 0.000000001) for i in range(y.size()[0])])
                    
                    flexible_idx = torch.arange(features.shape[0], dtype=torch.long) < int(flexible_len[0])
                    # flexible_idy = torch.tensor(torch.LongTensor(range(y.size()[0])) < flexible_len[0])
                    flexible_len = torch.tensor(flexible_len)
                    # flexible_idx = torch.tensor(torch.LongTensor(range(features.shape[0])) < labels[1])
                    # print(flexible_idx)

                    # e =

                    # only flexible has
                    flex_count = int(flexible_idx.sum().item())
                    if y.dim() == 1:
                        y = y.reshape(-1, 1)
                    if y.size(0) != flex_count:
                        if y.size(0) > flex_count:
                            y = y[:flex_count]
                        else:
                            pad_rows = flex_count - y.size(0)
                            y = torch.cat([y, torch.zeros((pad_rows, y.size(1)), dtype=y.dtype)], dim=0)
                    y = y - x[flexible_idx, -3:]
                    data = Data(x=x, edge_index=edge_index, y=y)
                    # data.rmsd = torch.tensor([rmsds[idx]], dtype=torch.float)
                    # data.energy = torch.tensor([energies[idx]], dtype=torch.float)
                    data.bonds = bonds
                    data.dist = dist
                    
                    data.alpha = _compute_edge_alpha(edge_index, x[:, -3:])
                    data.pdb = pdb
                    data.flexible_idx = flexible_idx
                    # data.flexible_idy = flexible_idy
                    data.flexible_len = flexible_len
                    if y.dim() == 2 and y.size(1) > 3:
                        data.pose_rmsd = y[:, 3].abs().mean().reshape(1)


                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    data_list.append(data)
                
                    pbar.update(1)
            
            pbar.close()
            torch.save(self.collate(data_list),
                       os.path.join(self.processed_dir, f'{split}.pt'))


class PDBBindNextStep(InMemoryDataset):

    def __init__(self, root, model_dir, pre_root, gpu_id=0, subset=False, split='train', transform=None,
                 pre_transform=None, pre_filter=None):
        self.subset = subset
        self.model_dir = model_dir
        self.pre_root = pre_root

        gpu_id_ = str(gpu_id)
        device_str = 'cuda:' + gpu_id_ if torch.cuda.is_available() else 'cpu'
        # device_str = 'cpu'
        self.device = torch.device(device_str)

        super().__init__(root, transform, pre_transform, pre_filter)
        path = os.path.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path, weights_only=False)

    @property
    def raw_file_names(self):
        return [
            'train', 'test'
        ]

    @property
    def processed_dir(self):
        # name = 'subset' if self.subset else 'full'
        # return os.path.join(self.root, name, 'processed')
        return os.path.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        return ['train.pt', 'test.pt']

    # TODO: implement after uploading the entire dataset somewhere
    def download(self):
        print('Hello Download')
        pass

    @torch.no_grad()
    def process(self):
        for split in ['train', 'test']:
            pre_dataset = PDBBindCoor(root=self.pre_root, split=split)
            pre_loader=DataLoader(pre_dataset, batch_size=1)
            model = torch.load(self.model_dir, weights_only=False).to(self.device)
            model.eval()

            pbar = tqdm(total=len(pre_dataset))
            pbar.set_description(f'Processing {split} dataset')
            tot = 0
            data_list = []
            pbar.total = len(pre_dataset)
            pbar.refresh()
            loss_op = torch.nn.MSELoss()
            
            for data2 in pre_loader:
                data3 = data2.to(self.device)
                out1 = model(data3.x, data3.edge_index, data3.dist)
                out = out1.cpu().numpy()
                del out1
                del data3
                data = data2.to('cpu')
                # loss = loss_op(out, data.y.to(self.device))
                #data = data2.to('cpu')
                
                x = data.x.numpy()

                # maxx = 0.0
                idx = (data.flexible_idx==1).nonzero(as_tuple=True)[0].numpy()
                for i in idx:
                    x[i, -3:] += out[i]
                    data.y[i] -= out[i]

                    # if np.sqrt(np.sum(np.square(out[i]))) > maxx:
                    #     maxx = np.sqrt(np.sum(np.square(out[i])))
                    # data.x[i.item(), -3:] = out[i]
                # print(f'max movement: {maxx}')
                coor = x[:, -3:]
                x = torch.Tensor(x)
                
                # edge_index = data.edge_index.numpy()

                # dist = data.dist.numpy()
                distt = []
                edges = [[], []]
                protein_nodes = torch.max(data.edge_index[0, (data.dist[:, 0] > 0)]).item()
                # print(protein_nodes)
                # print(coor.shape, x.size())
                # print(data.edge_index.size())
                # print(data.edge_index.tolist())

                l1 = (data.flexible_idx[data.edge_index[0]]==0).nonzero(as_tuple=True)[0].numpy()
                l2 = (data.flexible_idx[data.edge_index[1]]==0).nonzero(as_tuple=True)[0].numpy()
                fix_idx = np.intersect1d(l1, l2)
                edge_index = data.edge_index[:, fix_idx]
                dist = data.dist[fix_idx]


                for i in range(x.size()[0]):
                    for j in idx:
                        if i <= j:
                            continue
                        c_i = coor[i]
                        c_j = coor[j]
                        dis = distance.euclidean(c_i, c_j)
                        dis = round(dis*100000) / 100000
                        if dis * SPACE < BOND_TH:
                            edges[0].append(i)
                            edges[1].append(j)
                            edges[0].append(j)
                            edges[1].append(i)
                            if i < protein_nodes and j < protein_nodes:
                                distt.append([dis, 0.0, 0.0])
                                distt.append([dis, 0.0, 0.0])
                            elif i >= protein_nodes and j >= protein_nodes:
                                distt.append([0.0, 0.0, dis])
                                distt.append([0.0, 0.0, dis])
                            else:
                                distt.append([0.0, dis, 0.0])
                                distt.append([0.0, dis, 0.0])
                edge_index = torch.cat((edge_index, torch.LongTensor(edges)), 1)
                dist = torch.cat((dist, torch.Tensor(distt)), 0)
                # print('wo cao!')
                # print(edge_index.size())
                # print('wo cao2!')
                # print(dist.size())
                # assert(edge_index.size()[1] == dist.size()[0])
                # break

                '''
                l1 = (data.flexible_idx[data.edge_index[0]]==1).nonzero(as_tuple=True)[0]
                l2 = (data.flexible_idx[data.edge_index[1]]==1).nonzero(as_tuple=True)[0]
                idx = torch.cat((l1, l2), 0).unique().numpy()
                for i in idx:
                    a = edge_index[0][i]
                    b = edge_index[1][i]
                    # dis = distance.euclidean(out[a], out[b])
                    dis = distance.euclidean(x[a, -3:], x[b, -3:])
                    dis = round(dis*100000) / 100000
                    for t in range(3):
                        if dist[i, t] != 0:
                            dist[i, t] = dis
                dist = torch.Tensor(dist)
                '''

                data_new = Data(x=x, edge_index=edge_index, y=data.y)
                # y = torch.LongTensor([data.y[0].item()])
                # data_new = Data(x=x, edge_index=edge_index, y=y)
                data_new.dist = dist
                data_new.flexible_idx = data.flexible_idx

                data_list.append(data_new)

                pbar.update(1)
            
            # for data in pre_loader:
            #     Data(x=x, edge_index=edge_index, y=y)
            
            pbar.close()
            torch.save(self.collate(data_list),
                       os.path.join(self.processed_dir, f'{split}.pt'))


class PDBBindNextStep2(InMemoryDataset):

    def __init__(self, root, model_dir, pre_root, gpu_id=0, subset=False, split='train', transform=None,
                 pre_transform=None, pre_filter=None):
        self.subset = subset
        self.model_dir = model_dir
        self.pre_root = pre_root
        self.split = split

        gpu_id_ = str(gpu_id)
        device_str = 'cuda:' + gpu_id_ if torch.cuda.is_available() else 'cpu'
        # device_str = 'cpu'
        self.device = torch.device(device_str)

        super().__init__(root, transform, pre_transform, pre_filter)
        path = os.path.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path, weights_only=False)

    @property
    def raw_file_names(self):
        return [
            'train', 'test'
        ]

    @property
    def processed_dir(self):
        # name = 'subset' if self.subset else 'full'
        # return os.path.join(self.root, name, 'processed')
        return os.path.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        return ['train.pt', 'test.pt']

    # TODO: implement after uploading the entire dataset somewhere
    def download(self):
        print('Hello Download')
        pass

    @torch.no_grad()
    def process(self):
        # for split in ['train', 'test']:
        for split in [self.split]:
            pre_dataset = PDBBindCoor(root=self.pre_root, split=split)
            pre_loader=DataLoader(pre_dataset, batch_size=1)
            model = torch.load(self.model_dir, weights_only=False).to(self.device)
            model.eval()

            pbar = tqdm(total=len(pre_dataset))
            pbar.set_description(f'Processing {split} dataset')
            tot = 0
            data_list = []
            pbar.total = len(pre_dataset)
            pbar.refresh()
            loss_op = torch.nn.MSELoss()
            
            for data2 in pre_loader:
                data3 = data2.to(self.device)
                out1 = model(data3.x, data3.edge_index, data3.dist)
                out = out1.cpu().numpy()
                del out1
                del data3
                data = data2.to('cpu')
                
                x = data.x.numpy()

                idx = (data.flexible_idx==1).nonzero(as_tuple=True)[0].numpy()
                flexible_len = data.flexible_len.item()
                for i in idx:
                    x[i, -3:] += out[i]
                    data.y[i] -= out[i]
                coor = x[:, -3:]
                x = torch.Tensor(x)
                
                # edge_index = data.edge_index.numpy()

                # dist = data.dist.numpy()
                distt = []
                edges = [[], []]
                # protein_nodes = torch.max(data.edge_index[0, (data.dist[:, 0] > 0)]).item()


                l1 = (data.flexible_idx[data.edge_index[0]]==0).nonzero(as_tuple=True)[0].numpy()
                l2 = (data.flexible_idx[data.edge_index[1]]==0).nonzero(as_tuple=True)[0].numpy()
                fix_idx = np.intersect1d(l1, l2)
                edge_index = data.edge_index[:, fix_idx]
                dist = data.dist[fix_idx]
                bond_idx = (data.dist[:, 0] != 0).nonzero(as_tuple=True)[0]
                bonds = data.edge_index[:, bond_idx].cpu().numpy()

                # between protein and ligand
                for i in range(flexible_len, x.size()[0]):
                    for j in idx:
                        if i <= j:
                            continue
                        c_i = coor[i]
                        c_j = coor[j]
                        dis = distance.euclidean(c_i, c_j)
                        dis = round(dis*100000) / 100000
                        if dis * SPACE < BOND_TH:
                            edges[0].append(i)
                            edges[1].append(j)
                            edges[0].append(j)
                            edges[1].append(i)
                            distt.append([0.0, dis, 0.0])
                            distt.append([0.0, dis, 0.0])
                            '''
                            if i < protein_nodes and j < protein_nodes:
                                distt.append([dis, 0.0, 0.0])
                                distt.append([dis, 0.0, 0.0])
                            elif i >= protein_nodes and j >= protein_nodes:
                                distt.append([0.0, 0.0, dis])
                                distt.append([0.0, 0.0, dis])
                            else:
                                distt.append([0.0, dis, 0.0])
                                distt.append([0.0, dis, 0.0])
                            '''

                # between ligand and ligand
                # bonds = data.bonds.cpu().numpy()
                for i in range(bonds.shape[1]):
                    ii = int(bonds[0][i])
                    jj = int(bonds[1][i])
                    #print(ii,jj)
                    edges[0].append(ii)
                    edges[1].append(jj)
                    # edges[0].append(jj)
                    # edges[1].append(ii)
                    c_i = coor[ii]
                    c_j = coor[jj]
                    dis = distance.euclidean(c_i, c_j)
                    dis = round(dis*100000) / 100000
                    distt.append([dis, 0.0, 0.0])
                    # distt.append([dis, 0.0, 0.0])
                edge_index = torch.cat((edge_index, torch.LongTensor(edges)), 1)
                dist = torch.cat((dist, torch.Tensor(distt)), 0)


                data_new = Data(x=x, edge_index=edge_index, y=data.y)
                # y = torch.LongTensor([data.y[0].item()])
                # data_new = Data(x=x, edge_index=edge_index, y=y)
                data_new.dist = dist
                data_new.pdb = data.pdb
                data_new.flexible_idx = data.flexible_idx
                data_new.bonds = data.bonds
                data_new.flexible_len = data.flexible_len

                data_list.append(data_new)

                pbar.update(1)
            
            # for data in pre_loader:
            #     Data(x=x, edge_index=edge_index, y=y)
            
            pbar.close()
            torch.save(self.collate(data_list),
                       os.path.join(self.processed_dir, f'{split}.pt'))


class PDBBindScreen(InMemoryDataset):

    def __init__(self, root, model_dir, gpu_id=0, subset=False, split='train', data_type='screen', transform=None,
                 pre_transform=None, pre_filter=None):
        self.subset = subset
        self.model_dir = model_dir
        self.data_type = data_type

        gpu_id_ = str(gpu_id)
        device_str = 'cuda:' + gpu_id_ if torch.cuda.is_available() else 'cpu'
        # device_str = 'cpu'
        self.device = torch.device(device_str)

        super().__init__(root, transform, pre_transform, pre_filter)
        path = os.path.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path, weights_only=False)

    @property
    def raw_file_names(self):
        if self.data_type == 'muv':
            return ['test']
        return [
            'train', 'test'
        ]

    @property
    def processed_dir(self):
        # name = 'subset' if self.subset else 'full'
        # return os.path.join(self.root, name, 'processed')
        return os.path.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        if self.data_type == 'muv':
            return ['test.pt']
        return ['train.pt', 'test.pt']
        

    # TODO: implement after uploading the entire dataset somewhere
    def download(self):
        print('Hello Download')
        pass

    @torch.no_grad()
    def process(self):
        if self.data_type == 'muv':
            splits = ['test']
        else:
            splits = ['train', 'test']
        # for split in ['train', 'test']:
        for split in splits:

            dataset_dir = os.path.join(self.raw_dir, f'{split}')
            files_num = len(
                glob.glob(os.path.join(dataset_dir, '*_data-G.json')))

            data_list = []
            graph_idx = 0

            pbar = tqdm(total=files_num)
            pbar.set_description(f'Processing {split} dataset')
            
            if self.model_dir == 'None':
                model = None
            else:
                model = torch.load(self.model_dir, weights_only=False).to(self.device)
                model.eval()

            for f in range(files_num):

                with open(os.path.join(dataset_dir, f'{f}_data-G.json')) as gf:
                    graphs = gf.readlines()
                num_graphs_per_file = len(graphs)//3

                pbar.total = num_graphs_per_file * files_num
                pbar.refresh()

                # features = []
                # with open(os.path.join(dataset_dir, f'{f}_data-feats'), 'rb') as ff:
                #     for _ in range(num_graphs_per_file):
                #         features.append(np.load(ff))

                feat_file = open(os.path.join(dataset_dir, f'{f}_data-feats'), 'rb')
                label_file = open(os.path.join(dataset_dir, f'{f}_label'), 'rb')

                # labels = np.loadtxt(os.path.join(dataset_dir, f'{f}_label'))
                # rmsds = labels[:, 1]
                # energies = labels[:, 2]
                # labels = labels[:, 0]

                for idx in range(num_graphs_per_file):

                    # it's slow because of these!
                    features = np.load(feat_file)
                    # features[:,-3:] *= 100
                    indptr = ast.literal_eval(graphs[3*idx])
                    indices = ast.literal_eval(graphs[3*idx+1])
                    dist = ast.literal_eval(graphs[3*idx+2])
                    labels = np.load(label_file) # label and num of flexible
                    flexible = int(labels[1])


                    x = torch.Tensor(features)
                    dist = torch.tensor(dist, dtype=torch.float)
                    dist = dist.reshape(dist.size()[0], -1)
                    indptr = torch.LongTensor(indptr)
                    indices = torch.LongTensor(indices)
                    row_idx = _row_idx_from_node_index(indptr, len(indices))
                    edge_index = torch.stack((row_idx, indices), dim=0)
                    
                    if model is not None:
                        out = model(x.to(self.device), edge_index.to(self.device), dist.to(self.device))[:flexible, :].cpu()
                        x[:flexible, -3:] += out
                        coor = x[:, -3:].numpy()
    
                        distt = []
                        edges = [[], []]
                        ligand_nodes = flexible
    
                        l1 = (edge_index[0] >= flexible).nonzero(as_tuple=True)[0].numpy()
                        l2 = (edge_index[1] >= flexible).nonzero(as_tuple=True)[0].numpy()
                        fix_idx = np.intersect1d(l1, l2)
                        edge_index = edge_index[:, fix_idx]
                        dist = dist[fix_idx]
    
    
                        for i in range(x.size()[0]):
                            for j in range(ligand_nodes):
                                if i <= j:
                                    continue
                                c_i = coor[i]
                                c_j = coor[j]
                                dis = distance.euclidean(c_i, c_j)
                                dis = round(dis*100000) / 100000
                                if dis * SPACE < BOND_TH:
                                    edges[0].append(i)
                                    edges[1].append(j)
                                    edges[0].append(j)
                                    edges[1].append(i)
                                    if i < ligand_nodes and j < ligand_nodes:
                                        distt.append([dis, 0.0, 0.0])
                                        distt.append([dis, 0.0, 0.0])
                                    elif i >= ligand_nodes and j >= ligand_nodes:
                                        distt.append([0.0, 0.0, dis])
                                        distt.append([0.0, 0.0, dis])
                                    else:
                                        distt.append([0.0, dis, 0.0])
                                        distt.append([0.0, dis, 0.0])
                        edge_index = torch.cat((edge_index, torch.LongTensor(edges)), 1)
                        dist = torch.cat((dist, torch.Tensor(distt)), 0)

                    y = torch.LongTensor([labels[0].astype(int)])


                    data = Data(x=x, edge_index=edge_index, y=y)
                    # data.rmsd = torch.tensor([rmsds[idx]], dtype=torch.float)
                    # data.energy = torch.tensor([energies[idx]], dtype=torch.float)
                    data.dist = dist
                    flexible_idx = torch.arange(features.shape[0], dtype=torch.long) < int(flexible)
                    data.flexible_idx = flexible_idx
                    if labels.shape[0] > 2:
                        energy = labels[2]
                        data.energy = torch.tensor([energy], dtype=torch.float)
                    if labels.shape[0] > 3:
                        energy = labels[3]
                        data.tot_energy = torch.tensor([energy], dtype=torch.float)
                    if labels.shape[0] > 4:
                        energy = labels[4]
                        data.lig_idx = torch.tensor([energy], dtype=torch.int)

                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    data_list.append(data)
                
                    pbar.update(1)
            
            pbar.close()
            torch.save(self.collate(data_list),
                       os.path.join(self.processed_dir, f'{split}.pt'))


class PDBBindScreen2(InMemoryDataset):

    def __init__(self, root, model_dir, gpu_id=0, subset=False, split='train', data_type='screen', transform=None,
                 pre_transform=None, pre_filter=None):
        self.subset = subset
        self.model_dir = model_dir
        self.data_type = data_type

        gpu_id_ = str(gpu_id)
        device_str = 'cuda:' + gpu_id_ if torch.cuda.is_available() else 'cpu'
        # device_str = 'cpu'
        self.device = torch.device(device_str)

        super().__init__(root, transform, pre_transform, pre_filter)
        path = os.path.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path, weights_only=False)

    @property
    def raw_file_names(self):
        if self.data_type == 'muv':
            return ['test']
        return [
            'train', 'test'
        ]

    @property
    def processed_dir(self):
        # name = 'subset' if self.subset else 'full'
        # return os.path.join(self.root, name, 'processed')
        return os.path.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        if self.data_type == 'muv':
            return ['test.pt']
        return ['train.pt', 'test.pt']
        

    # TODO: implement after uploading the entire dataset somewhere
    def download(self):
        print('Hello Download')
        pass

    @torch.no_grad()
    def process(self):
        if self.data_type == 'muv':
            splits = ['test']
        else:
            splits = ['train', 'test']
        # for split in ['train', 'test']:
        for split in splits:

            dataset_dir = os.path.join(self.raw_dir, f'{split}')
            files_num = len(
                glob.glob(os.path.join(dataset_dir, '*_data-G.json')))

            data_list = []
            graph_idx = 0

            pbar = tqdm(total=files_num)
            pbar.set_description(f'Processing {split} dataset')
            
            if self.model_dir == 'None':
                model = None
            else:
                model = torch.load(self.model_dir, weights_only=False).to(self.device)
                model.eval()

            for f in range(files_num):

                with open(os.path.join(dataset_dir, f'{f}_data-G.json')) as gf:
                    graphs = gf.readlines()
                num_graphs_per_file = len(graphs)//3

                pbar.total = num_graphs_per_file * files_num
                pbar.refresh()

                # features = []
                # with open(os.path.join(dataset_dir, f'{f}_data-feats'), 'rb') as ff:
                #     for _ in range(num_graphs_per_file):
                #         features.append(np.load(ff))

                feat_file = open(os.path.join(dataset_dir, f'{f}_data-feats'), 'rb')
                label_file = open(os.path.join(dataset_dir, f'{f}_label'), 'rb')

                # labels = np.loadtxt(os.path.join(dataset_dir, f'{f}_label'))
                # rmsds = labels[:, 1]
                # energies = labels[:, 2]
                # labels = labels[:, 0]

                for idx in range(num_graphs_per_file):

                    # it's slow because of these!
                    features = np.load(feat_file)
                    # features[:,-3:] *= 100
                    indptr = ast.literal_eval(graphs[3*idx])
                    indices = ast.literal_eval(graphs[3*idx+1])
                    dist = ast.literal_eval(graphs[3*idx+2])
                    labels = np.load(label_file) # label and num of flexible
                    flexible = int(labels[1])
                    bonds = np.load(label_file)
                    pose_rmsd = _read_optional_scalar(label_file)

                    x = torch.Tensor(features)
                    dist = torch.tensor(dist, dtype=torch.float)
                    dist = dist.reshape(dist.size()[0], -1)
                    indptr = torch.LongTensor(indptr)
                    indices = torch.LongTensor(indices)
                    row_idx = _row_idx_from_node_index(indptr, len(indices))
                    edge_index = torch.stack((row_idx, indices), dim=0)
                    
                    if model is not None:
                        out = model(x.to(self.device), edge_index.to(self.device), dist.to(self.device))[:flexible, :].cpu()
                        x[:flexible, -3:] += out
                        coor = x[:, -3:].numpy()
    
                        distt = []
                        edges = [[], []]
                        ligand_nodes = flexible
    
                        l1 = (edge_index[0] >= flexible).nonzero(as_tuple=True)[0].numpy()
                        l2 = (edge_index[1] >= flexible).nonzero(as_tuple=True)[0].numpy()
                        fix_idx = np.intersect1d(l1, l2)
                        edge_index = edge_index[:, fix_idx]
                        dist = dist[fix_idx]

                        # between protein and ligand
                        for i in range(flexible, x.size()[0]):
                            for j in range(ligand_nodes):
                                if i <= j:
                                    continue
                                c_i = coor[i]
                                c_j = coor[j]
                                dis = distance.euclidean(c_i, c_j)
                                dis = round(dis*100000) / 100000
                                if dis * SPACE < BOND_TH:
                                    edges[0].append(i)
                                    edges[1].append(j)
                                    edges[0].append(j)
                                    edges[1].append(i)
                                    distt.append([0.0, dis, 0.0])
                                    distt.append([0.0, dis, 0.0])
                                    '''
                                    if i < protein_nodes and j < protein_nodes:
                                        distt.append([dis, 0.0, 0.0])
                                        distt.append([dis, 0.0, 0.0])
                                    elif i >= protein_nodes and j >= protein_nodes:
                                        distt.append([0.0, 0.0, dis])
                                        distt.append([0.0, 0.0, dis])
                                    else:
                                        distt.append([0.0, dis, 0.0])
                                        distt.append([0.0, dis, 0.0])
                                    '''

                        # between ligand and ligand
                        for i in range(bonds.shape[0]):
                            ii = int(bonds[i][0])
                            jj = int(bonds[i][1])
                            #print(ii,jj)
                            edges[0].append(ii)
                            edges[1].append(jj)
                            edges[0].append(jj)
                            edges[1].append(ii)
                            c_i = coor[ii]
                            c_j = coor[jj]
                            dis = distance.euclidean(c_i, c_j)
                            dis = round(dis*100000) / 100000
                            distt.append([dis, 0.0, 0.0])
                            distt.append([dis, 0.0, 0.0])
                        edge_index = torch.cat((edge_index, torch.LongTensor(edges)), 1)
                        dist = torch.cat((dist, torch.Tensor(distt)), 0)

                    y = torch.LongTensor([labels[0].astype(int)])


                    data = Data(x=x, edge_index=edge_index, y=y)
                    # data.rmsd = torch.tensor([rmsds[idx]], dtype=torch.float)
                    # data.energy = torch.tensor([energies[idx]], dtype=torch.float)
                    data.dist = dist
                    flexible_idx = torch.arange(features.shape[0], dtype=torch.long) < int(flexible)
                    data.flexible_idx = flexible_idx
                    if labels.shape[0] > 2:
                        energy = labels[2]
                        data.energy = torch.tensor([energy], dtype=torch.float)
                    if labels.shape[0] > 3:
                        energy = labels[3]
                        data.tot_energy = torch.tensor([energy], dtype=torch.float)
                    if labels.shape[0] > 4:
                        energy = labels[4]
                        data.lig_idx = torch.tensor([energy], dtype=torch.int)
                    bonds = torch.Tensor(bonds)
                    data.bonds = bonds
                    if labels.shape[0] > 2:
                        data.pose_rmsd = torch.tensor([float(labels[2])], dtype=torch.float)
                    if labels.shape[0] > 2:
                        data.pose_rmsd = torch.tensor([float(labels[2])], dtype=torch.float)
                    if pose_rmsd is not None:
                        data.pose_rmsd = torch.tensor([pose_rmsd], dtype=torch.float)
                        
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    data_list.append(data)
                
                    pbar.update(1)
            
            pbar.close()
            torch.save(self.collate(data_list),
                       os.path.join(self.processed_dir, f'{split}.pt'))



"""
This dataset is only for test, please fell free to modify it.
"""
class PDBBindCoorTest(InMemoryDataset):

    def __init__(self, root, model_dir, pre_root, gpu_id=0, subset=False, split='train', transform=None,
                 pre_transform=None, pre_filter=None):
        self.subset = subset
        self.model_dir = model_dir
        self.pre_root = pre_root

        gpu_id_ = str(gpu_id)
        device_str = 'cuda:' + gpu_id_ if torch.cuda.is_available() else 'cpu'
        # device_str = 'cpu'
        self.device = torch.device(device_str)

        super().__init__(root, transform, pre_transform, pre_filter)
        path = os.path.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path, weights_only=False)

    @property
    def raw_file_names(self):
        return [
            'train', 'test'
        ]

    @property
    def processed_dir(self):
        # name = 'subset' if self.subset else 'full'
        # return os.path.join(self.root, name, 'processed')
        return os.path.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        return ['train.pt', 'test.pt']

    # TODO: implement after uploading the entire dataset somewhere
    def download(self):
        print('Hello Download')
        pass

    @torch.no_grad()
    def process(self):
        for split in ['train', 'test']:
            pre_dataset = PDBBindCoor(root=self.pre_root, split=split)
            pre_loader=DataLoader(pre_dataset, batch_size=1)
            # model = torch.load(self.model_dir).to(self.device)

            pbar = tqdm(total=len(pre_dataset))
            pbar.set_description(f'Processing {split} dataset')
            tot = 0
            data_list = []
            pbar.total = len(pre_dataset)
            pbar.refresh()
            loss_op = torch.nn.MSELoss()
            
            for data in pre_loader:

                y = (data.y - data.x[:, -3:])
                data_new = Data(x=data.x, edge_index=data.edge_index, y=y)
                data_new.dist = data.dist
                data_new.flexible_idx = data.flexible_idx

                data_list.append(data_new)

                pbar.update(1)
            
            # for data in pre_loader:
            #     Data(x=x, edge_index=edge_index, y=y)
            
            pbar.close()
            torch.save(self.collate(data_list),
                       os.path.join(self.processed_dir, f'{split}.pt'))
