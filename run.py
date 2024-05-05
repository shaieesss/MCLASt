import random
import os
import torch
import argparse
from tqdm import tqdm
import dgl
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
# Set argument
parser = argparse.ArgumentParser(description='SAMCL')
# Set argument
parser = argparse.ArgumentParser(description='CNCL-GAD')
parser.add_argument('--dataset', type=str, default='cora')  # 'BlogCatalog'  'Flickr'  'ACM'  'cora'  'citeseer'  'pubmed'
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=3)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--readout', type=str, default='avg')  # max min avg  weighted_sum
parser.add_argument('--auc_test_rounds', type=int, default=256)
parser.add_argument('--negsamp_ratio', type=int, default=1)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--alpha', type=float)
parser.add_argument('--beta', type=float)
parser.add_argument('--gama', type=float, default=1)
parser.add_argument('--Sinkhorn_iter_times', type=int, default=5)
parser.add_argument('--Sinkhorn_lamb', type=int, default=20)
parser.add_argument('--topo_t', type=int, default=10, help='temperature for sigmoid in topology dist')
parser.add_argument('--temperature', type=float, default=3, help='temperature for fx')
parser.add_argument('--rectified', type=bool, help='use rectified cost matrix', default=True)
parser.add_argument('--have_neg', type=bool, help='anomaly score and LOSS contain negtive pairs OT', default=True)
parser.add_argument('--neg_top_k', type=float, help='top max k of OT to select negtive pairs', default=20)

parser.add_argument('--K_1', type=int, help='view 1')
parser.add_argument('--K_2', type=int, help='view 2')
parser.add_argument('--restart_prob_1', type=float, help='RWR restart probability on view 1', default=0.9)
parser.add_argument('--restart_prob_2', type=float, help='RWR restart probability on view 2', default=0.3)
parser.add_argument('--subgraph_mode', type=str, default='1+2')

args = parser.parse_args(args=[])


if args.lr is None:
    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        args.lr = 2e-3
    elif args.dataset == 'BlogCatalog':
        args.lr = 1e-2
    elif args.dataset == 'ACM':
        args.lr = 5e-3

if args.beta is None:
    if args.dataset == 'cora':
        args.beta = 0.4
    elif args.dataset in ['BlogCatalog', 'citeseer']:
        args.beta = 0.8
    elif args.dataset == 'ACM':
        args.beta = 0.1
    elif args.dataset == 'pubmed':
        args.beta = 0.6

if args.alpha is None:
    if args.dataset in ['cora', 'citeseer']:
        args.alpha = 0.5
    elif args.dataset in ['BlogCatalog', 'pubmed']:
        args.alpha = 0.7
    elif args.dataset == 'ACM':
        args.alpha = 0.3

if args.K_1 is None:
    if args.dataset == 'cora':
        args.K_1 = 2
    elif args.dataset in ['BlogCatalog', 'pubmed']:
        args.K_1 = 4
    elif args.dataset in ['citeseer', 'ACM']:
        args.K_1 = 6

if args.K_2 is None:
    if args.dataset == 'citeseer':
        args.K_2 = 4
    elif args.dataset in ['cora', 'ACM', 'BlogCatalog', 'pubmed']:
        args.K_2 = 8

AUC_list = []

alpha_recon = args.beta
alpha_inter = args.alpha
alpha_intra = args.gama
batch_size = args.batch_size
subgraph_size_1 = args.K_1
subgraph_size_2 = args.K_2
print('Dataset: ', args.dataset)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
seed = args.seed
dgl.random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import networkx as nx
import scipy.sparse as sp
import scipy.io as sio
import random
import torch
# Load and preprocess data
def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
def load_mat(dataset, train_rate=0.3, val_rate=0.1):
    """Load .mat dataset."""
    data = sio.loadmat("./dataset/{}.mat".format(dataset))
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']
    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)
    labels = np.squeeze(np.array(data['Class'], dtype=np.int64) - 1)
    num_classes = np.max(labels) + 1
    labels = dense_to_one_hot(labels, num_classes)
    ano_labels = np.squeeze(np.array(label))

    if 'str_anomaly_label' in data:
        str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
        attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
    else:
        str_ano_labels = None
        attr_ano_labels = None

    num_node = adj.shape[0]
    num_train = int(num_node * train_rate)
    num_val = int(num_node * val_rate)
    all_idx = list(range(num_node))
    random.shuffle(all_idx)
    idx_train = all_idx[: num_train]
    idx_val = all_idx[num_train: num_train + num_val]
    idx_test = all_idx[num_train + num_val:]

    return adj, feat, labels, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels
adj, features, labels, idx_train, idx_val, \
idx_test, ano_label, str_ano_label, attr_ano_label = load_mat(args.dataset)
A = adj
degree = np.sum(adj, axis=0)
degree_ave = np.mean(degree)
dgl_graph = dgl.from_scipy(adj)
raw_feature = features.todense()
def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape
    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx
def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)
features, _ = preprocess_features(features)
nb_nodes = features.shape[0]
ft_size = features.shape[1]
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    adj_raw = adj
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo(), adj_raw
adj, adj_raw = normalize_adj(adj)
adj = (adj + sp.eye(adj.shape[0])).todense()
adj_raw = adj_raw.todense()
features = torch.FloatTensor(features[np.newaxis])
raw_feature = torch.FloatTensor(raw_feature[np.newaxis])
adj = torch.FloatTensor(adj[np.newaxis])
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
def register_topology(dataset, adj):
    MAX_HOP = 100
    topo_file = f"./dataset/topology_dist/{dataset.lower()}_padding.pt"
    exist = os.path.isfile(topo_file)
    if not exist:
        nx_graph = nx.from_scipy_sparse_array(adj)
        node_num = adj.shape[0]
        generator = dict(nx.shortest_path_length(nx_graph))
        topology_dist = torch.zeros((node_num, node_num)) # we shift the node index with 1, in order to store 0-index for padding nodes
        mask = torch.zeros((node_num, node_num)).bool()
        for i in tqdm(range(0, node_num)):
            # print(f"processing {i}-th node")
            for j in range(0, node_num):
                if j in generator[i].keys():
                    topology_dist[i][j] = generator[i][j]
                else:
                    topology_dist[i][j] = MAX_HOP
                    mask[i][j] = True # record nodes that do not have connections
        torch.save(topology_dist, topo_file)
    else:
        topology_dist = torch.load(topo_file)
    return topology_dist
def gen_batch_topology_dist(full_topology_dist, node_idx1, node_idx2):
    batch_size = node_idx1.shape[0]
    batch_subpology_dist = [full_topology_dist.index_select(dim=0, index=node_idx1[i]).
                            index_select(dim=1, index=node_idx2[i]) for i in range(batch_size)]
    batch_subpology_dist = torch.stack(batch_subpology_dist)
    return batch_subpology_dist
class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)

class Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_round):
        super(Discriminator, self).__init__()
#         self.fc = nn.Linear(n_h * 2, n_h)
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)
        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    def forward(self, c, h_pl):
        scs = []
        # positive
        scs.append(self.f_k(h_pl, c))
        # negative
        c_mi = c
        for _ in range(self.negsamp_round):
            c_mi = torch.cat((c_mi[-2:-1, :], c_mi[:-1, :]), 0)
            scs.append(self.f_k(h_pl, c_mi))
        logits = torch.cat(tuple(scs))
        return logits

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)


class MaxReadout(nn.Module):
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq, 1).values


class MinReadout(nn.Module):
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values

class Decoder(nn.Module):
    def __init__(self, n_in, n_h, hidden_size = 128):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.network1 = nn.Sequential(
            nn.Linear(n_h , self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, n_in),
            nn.PReLU()
        )
        self.network2 = nn.Sequential(
            nn.Linear(n_h * 2, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, n_in),
            nn.PReLU()
        )
        self.network3 = nn.Sequential(
            nn.Linear(n_h * 3, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, n_in),
            nn.PReLU()
        )
        self.network4 = nn.Sequential(
            nn.Linear(n_h * 4, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, n_in),
            nn.PReLU()
        )
        self.network5 = nn.Sequential(
            nn.Linear(n_h * 5, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, n_in),
            nn.PReLU()
        )
        self.network6 = nn.Sequential(
            nn.Linear(n_h * 6, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, n_in),
            nn.PReLU()
        )
        self.network7 = nn.Sequential(
            nn.Linear(n_h * 7, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, n_in),
            nn.PReLU()
        )
    def forward(self, h_raw, subgraph_size):
        sub_size = h_raw.shape[1]
        batch_size = h_raw.shape[0]
        sub_node = h_raw[:, :sub_size - 2, :]
        input_res = sub_node.reshape(batch_size, -1)
        if subgraph_size == 1:
            node_recons = self.network1(input_res)
        elif subgraph_size == 2:
            node_recons = self.network2(input_res)
        elif subgraph_size == 3:
            node_recons = self.network3(input_res)
        elif subgraph_size == 4:
            node_recons = self.network4(input_res)
        elif subgraph_size == 5:
            node_recons = self.network5(input_res)
        elif subgraph_size == 6:
            node_recons = self.network6(input_res)
        elif subgraph_size == 7:
            node_recons = self.network7(input_res)
        return node_recons
def matrix_diag(diagonal):
    N = diagonal.shape[-1]
    shape = diagonal.shape[:-1] + (N, N)
    device, dtype = diagonal.device, diagonal.dtype
    result = torch.zeros(shape, dtype=dtype, device=device)
    indices = torch.arange(result.numel(), device=device).reshape(shape)
    indices = indices.diagonal(dim1=-2, dim2=-1)
    result.view(-1)[indices] = diagonal
    return result


def similarity( reps1, reps2 ):
    reps1_unit = F.normalize(reps1, dim=-1)
    reps2_unit = F.normalize(reps2, dim=-1)
    if len(reps1.shape) == 2:
        sim_mat = torch.einsum("ik,jk->ij", [reps1_unit, reps2_unit])
    elif len(reps1.shape) == 3:
        sim_mat = torch.einsum('bik,bjk->bij', [reps1_unit, reps2_unit])
    else:
        print(f"{len(reps1.shape)} dimension tensor is not supported for this function!")
    return sim_mat


def Sinkhorn( out1, avg_out1, out2, avg_out2,
             Sinkhorn_iter_times=5, lamb=20, rescale_ratio=None):
    cost_matrix = 1 - similarity(out1, out2)
    if rescale_ratio is not None:
        cost_matrix = cost_matrix * rescale_ratio
    # Sinkhorn iteration
    with torch.no_grad():
        r = torch.bmm(out1, avg_out2.transpose(1, 2))
        r[r <= 0] = 1e-8
        r = r / r.sum(dim=1, keepdim=True)
        c = torch.bmm(out2, avg_out1.transpose(1, 2))
        c[c <= 0] = 1e-8
        c = c / c.sum(dim=1, keepdim=True)
        P = torch.exp(-1 * lamb * cost_matrix)
        u = (torch.ones_like(c) / c.shape[1])
        for i in range(Sinkhorn_iter_times):
            v = torch.div(r, torch.bmm(P, u))
            u = torch.div(c, torch.bmm(P.transpose(1, 2), v))
        u = u.squeeze(dim=-1)
        v = v.squeeze(dim=-1)
        transport_matrix = torch.bmm(torch.bmm(matrix_diag(v), P), matrix_diag(u))
    assert cost_matrix.shape == transport_matrix.shape
    # Eq.10
    S = torch.mul(transport_matrix, 1 - cost_matrix).sum(dim=1).sum(dim=1, keepdim=True)

    return S

class Model(nn.Module):
    def __init__(self, n_in, n_h, activation, negsamp_round, readout, have_neg = False, hidden_size = 128,
                 temperature=0.4, Sinkhorn_iter_times = 5, lamb=20, is_rectified=True, topo_t=2, neg_top_k=50):
        super(Model, self).__init__()
        self.read_mode = readout
        self.hidden_size = hidden_size
        self.gcn = GCN(n_in, n_h, activation)
        self.decoder = Decoder(n_in, n_h, self.hidden_size)
        self.temperature = temperature
        self.Sinkhorn_iter_times = Sinkhorn_iter_times
        self.lamb = lamb
        self.rectified = is_rectified
        self.topo_t = topo_t
        self.have_neg = have_neg
        self.neg_top_k = neg_top_k
        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()
        self.discriminator = Discriminator(n_h, negsamp_round)
    # --------------------Inter-view / Cross-view / RoSA loss---------------#
    def InterViewLoss(self, h1, h2, rescale_ratio, have_neg = False, neg_top_k = 50):
        h1_new = h1.clone()
        h2_new = h2.clone()
        h1_new[:, [-2, -1], :] = h1_new[:, [-1, -2], :]
        h2_new[:, [-2, -1], :] = h2_new[:, [-1, -2], :]
        h_graph_1 = h1_new[:, : -1, :]  # (B, subgraph_size, D)
        h_node_1 = h1_new[:, -1, :][:, None, :]  # (B,1,D)
        h_graph_2 = h2_new[:, : -1, :]  # (B, subgraph_size, D)
        h_node_2 = h2_new[:, -1, :][:, None, :]  # (B,1,D)
        # positive
        fx = lambda x: torch.exp(x / self.temperature)
        if rescale_ratio is not None:
            # rescale_ratio about 0.5+-
            sim_pos = Sinkhorn(h_graph_1, h_node_1, h_graph_2, h_node_2, self.Sinkhorn_iter_times, self.lamb, rescale_ratio)
            loss_pos = fx( sim_pos ) * 2
        else:
            sim_pos = Sinkhorn(h_graph_1, h_node_1, h_graph_2, h_node_2, self.Sinkhorn_iter_times, self.lamb)
            loss_pos = fx( sim_pos )
        # negative
        if have_neg:
            neg_sim_list = []
            loss_neg_total = 0
            sim_neg = 0
            batch_size = h_node_1.shape[0]
            # 初始化负样本索引
            neg_index = list(range(batch_size))
            # 对于每个节点
            for i in range((batch_size - 1)):
                # 将负样本索引最后一个索引换成0
                neg_index.insert(0, neg_index.pop(-1))
                out1_perm = h_graph_1[neg_index].clone()
                out2_perm = h_graph_2[neg_index].clone()
                avg_out1_perm = h_node_1[neg_index].clone()
                avg_out2_perm = h_node_2[neg_index].clone()
                # torch.Size([128, 1])
                sim_neg1 = Sinkhorn(h_graph_1, h_node_1, out1_perm, avg_out1_perm, self.Sinkhorn_iter_times, self.lamb)
                sim_neg2 = Sinkhorn(h_graph_1, h_node_1, out2_perm, avg_out2_perm, self.Sinkhorn_iter_times, self.lamb)
                sim_neg += (sim_neg1 + sim_neg2) / 2
                loss_neg_total += fx(sim_neg1) + fx(sim_neg2)
                # top k neg
                neg_sim_list.append(torch.squeeze(sim_neg1).detach().cpu().numpy())
                neg_sim_list.append(torch.squeeze(sim_neg2).detach().cpu().numpy())
            # top max k as negative pairs
            # torch.Size([254, 128])
            neg_sim = torch.tensor(np.array(neg_sim_list), requires_grad=True)
            neg_sim = neg_sim.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
            neg_sim = torch.sort(neg_sim, descending=False, dim=0)[0]
            loss_neg_top_k = neg_sim[:neg_top_k, :]
            loss_neg_total = torch.mean(loss_neg_top_k, dim=0)
            loss_neg_total = torch.unsqueeze(loss_neg_total, 1)
            loss = -torch.log((loss_pos) / (loss_neg_total + loss_pos)) #!!!
            sim_neg = sim_neg / (batch_size - 1)
            sim_all = sim_pos - sim_neg
        else:
            loss = -torch.log( loss_pos )
            sim_all = sim_pos
        return loss, sim_all, sim_pos
    def node_node(self, h1, h2, neg_top_k = 50):
        # h :batch_size * 1 * feature_dim
        h1_new = h1.clone().unsqueeze(dim=1)
        h2_new = h2.clone().unsqueeze(dim=1)
        # positive
        fx = lambda x: torch.exp(x / self.temperature)
        
        sim_pos = F.pairwise_distance(h1_new, h2_new)
        loss_pos = fx( sim_pos ) * 2
        # negative

        neg_sim_list = []
        loss_neg_total = 0
        sim_neg = 0
        batch_size = h1_new.shape[0]
        # 初始化负样本索引
        neg_index = list(range(batch_size))
        # 对于每个节点
        for i in range((batch_size - 1)):
            # 将负样本索引最后一个索引换成0
            neg_index.insert(0, neg_index.pop(-1))
            # h :batch_size * 1 * feature_dim
            avg_out1_perm = h1_new[neg_index].clone()
            avg_out2_perm = h2_new[neg_index].clone()
            # torch.Size([128, 1])
            sim_neg1 = F.pairwise_distance(h1_new, avg_out1_perm)
            sim_neg2 = F.pairwise_distance(h1_new, avg_out2_perm)
            sim_neg += (sim_neg1 + sim_neg2) / 2
            loss_neg_total += fx(sim_neg1) + fx(sim_neg2)
            # top k neg
            neg_sim_list.append(torch.squeeze(sim_neg1).detach().cpu().numpy())
            neg_sim_list.append(torch.squeeze(sim_neg2).detach().cpu().numpy())
        # top max k as negative pairs
        # torch.Size([254, 128])
        neg_sim = torch.tensor(np.array(neg_sim_list), requires_grad=True)
        neg_sim = neg_sim.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        neg_sim = torch.sort(neg_sim, descending=False, dim=0)[0]
        loss_neg_top_k = neg_sim[:neg_top_k, :]
        loss_neg_total = torch.mean(loss_neg_top_k, dim=0)
        loss_neg_total = torch.unsqueeze(loss_neg_total, 1)
        loss = -torch.log((loss_pos) / (loss_neg_total + loss_pos)) #!!!
        sim_neg = sim_neg / (batch_size - 1)
        sim_all = sim_pos - sim_neg

        return loss, sim_all, sim_pos
    def forward(self, feature1, adj1, raw1, size1, feature2, adj2, raw2, size2,
                full_topology_dist, batch_g_idx, batch_g_idx_2, sparse=False):
        # batch_size * sub_graph_size+1 * feature_dim
        h1 = self.gcn(feature1, adj1, sparse)
        h2 = self.gcn(feature2, adj2, sparse)
        h_raw_1 = self.gcn(raw1, adj1, sparse)
        h_raw_2 = self.gcn(raw2, adj2, sparse)
        # 加载Eq. (9)
        if self.rectified:
            topology_dist = gen_batch_topology_dist(full_topology_dist, batch_g_idx, batch_g_idx_2)
            rescale_ratio = torch.sigmoid(topology_dist / self.topo_t)
        else:
            rescale_ratio = None
        # --------------------recon loss--------------#
        # assert h_raw_1.shape == h_raw_2.shape
        node_recons_1 = self.decoder(h_raw_1, size1)
        node_recons_2 = self.decoder(h_raw_2, size2)

        # --------------------node-subgraph alignment loss---------------#
        # h_graph_read_1：[128, 64] batch_size * features_dim
        h_node_1 = h1[:, -1, :]
        h_graph_read_1 = self.read(h1[:, : -1, :])
        h_node_2 = h2[:, -1, :]
        h_graph_read_2 = self.read(h2[:, : -1, :])
        disc_1 = self.discriminator(h_graph_read_1, h_node_1)
        disc_2 = self.discriminator(h_graph_read_2, h_node_2)

        # --------------------node-node alignment loss--------------#
        Intra_loss_1, sim_all_node1, sim_pos_node1 = self.node_node(h_node_1, h_node_2)
        Intra_loss_2, sim_all_node2, sim_pos_node2 = self.node_node(h_node_2, h_node_1)
        # --------------------subgraph-subgraph alignment loss---------------#
        if self.rectified:
            Inter_loss_1, sim_all_1, sim_pos_1 = self.InterViewLoss(h1, h2, rescale_ratio, self.have_neg, self.neg_top_k)
            Inter_loss_2, sim_all_2, sim_pos_2 = self.InterViewLoss(h2, h1, rescale_ratio.transpose(1,2), self.have_neg, self.neg_top_k)
        else:
            Inter_loss_1, sim_all_1, sim_pos_1 = self.InterViewLoss(h1, h2, None, self.have_neg, self.neg_top_k)
            Inter_loss_2, sim_all_2, sim_pos_2 = self.InterViewLoss(h2, h1, None, self.have_neg, self.neg_top_k)
        return node_recons_1, node_recons_2, disc_1, disc_2, Inter_loss_1, Inter_loss_2, Intra_loss_1, Intra_loss_2, sim_all_1, sim_all_2, \
    sim_pos_1, sim_pos_2, sim_all_node1, sim_pos_node1, sim_all_node2, sim_pos_node2, h1[:, -1, :], h2[:, -1, :]

model = Model(n_in=ft_size, n_h=args.embedding_dim, activation='prelu', negsamp_round=args.negsamp_ratio,
            readout=args.readout, hidden_size=args.hidden_size, temperature=args.temperature,
            Sinkhorn_iter_times = args.Sinkhorn_iter_times, lamb=args.Sinkhorn_lamb, is_rectified=args.rectified,
            topo_t = args.topo_t, have_neg = args.have_neg, neg_top_k = args.neg_top_k)
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
full_topology_dist = register_topology(args.dataset, adj)
if torch.cuda.is_available():
    print('Using CUDA')
    full_topology_dist = full_topology_dist.to(device)
    model.to(device)
    features = features.to(device)
    raw_feature = raw_feature.to(device)
    adj = adj.to(device)
    b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).to(device))
else:
    b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]))
xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 1e9
best_t = 0
mse_loss = nn.MSELoss(reduction='mean')
if nb_nodes % batch_size == 0:
    batch_num = nb_nodes // batch_size
else:
    batch_num = nb_nodes // batch_size + 1
def get_first_adj(dgl_graph, adj, subgraph_size):
    """Generate the first view's subgraph with the first-order neighbor."""
    all_idx = list(range(dgl_graph.number_of_nodes()))
    subgraphs = []
    adj = np.array(adj.todense()).squeeze()
    for node_id in all_idx:
        first_adj = np.where(adj[node_id] == 1)
        first_adj = list(first_adj[0])
        if len(first_adj) < subgraph_size - 1:
            subgraphs.append(first_adj)
            first_adj.append(node_id) #自己也可以被循环选择
            subgraphs[node_id].extend(
                list(np.random.choice(first_adj, subgraph_size - len(first_adj) - 1, replace=True)))
        else:
            subgraphs.append(list(np.random.choice(first_adj, subgraph_size - 1, replace=False)))
        subgraphs[node_id].append(node_id)
    return subgraphs
def get_second_adj(dgl_graph, adj, subgraph_size):
    """Generate the second view's subgraph with the 1/2 first-order and 1/2 second-order neighbor. """
    all_idx = list(range(dgl_graph.number_of_nodes()))
    subgraphs = []
    adj_2 = adj.dot(adj)
    adj = np.array(adj.todense())
    adj_2 = np.array(adj_2.todense())
    row, col = np.diag_indices_from(adj_2)
    zeros = np.zeros(adj_2.shape[0])
    adj_2[row, col] = np.array(zeros)
    adj = adj.squeeze()
    adj_2 = adj_2.squeeze()
    for node_id in all_idx:
        first_adj = np.where(adj[node_id] == 1)
        second_adj = np.where(adj_2[node_id] != 0)
        first_adj = first_adj[0].tolist()
        second_adj = second_adj[0].tolist()
        if len(first_adj) < subgraph_size // 2:
            subgraphs.append(list(np.random.choice(first_adj, subgraph_size // 2, replace=True)))
            if len(second_adj) == 0:
                first_adj.append(node_id)
                subgraphs[node_id].extend(list(np.random.choice(first_adj, (subgraph_size - 1) // 2, replace=True)))
            elif len(second_adj) < (subgraph_size - 1) // 2:
                subgraphs[node_id].extend(list(np.random.choice(second_adj, (subgraph_size - 1) // 2, replace=True)))
            else:
                subgraphs[node_id].extend(list(np.random.choice(second_adj, (subgraph_size - 1) // 2, replace=False)))
        else:
            if len(second_adj) == 0:
                first_adj.append(node_id)
                if len(first_adj) < subgraph_size - 1:
                    subgraphs.append(list(np.random.choice(first_adj, (subgraph_size - 1), replace=True)))
                else:
                    subgraphs.append(list(np.random.choice(first_adj, (subgraph_size - 1), replace=False)))
            elif len(second_adj) < (subgraph_size - 1) // 2 :
                subgraphs.append(list(np.random.choice(first_adj, subgraph_size // 2, replace=False)))
                subgraphs[node_id].extend(list(np.random.choice(second_adj, (subgraph_size - 1) // 2, replace=True)))
            else:
                subgraphs.append(list(np.random.choice(first_adj, subgraph_size // 2, replace=False)))
                subgraphs[node_id].extend(list(np.random.choice(second_adj, (subgraph_size - 1) // 2, replace=False)))
        subgraphs[node_id].append(node_id)
    return subgraphs
def generate_subgraph(args, dgl_graph, A, subgraph_size_1, subgraph_size_2):
    """Generate subgraph with RWR/first & second -neiborhood algorithm."""
    restart_prob_1 = args.restart_prob_1
    restart_prob_2 = args.restart_prob_2
    if args.subgraph_mode == '1+2':
        subgraphs_1 = get_first_adj(dgl_graph, A, subgraph_size_1)
        subgraphs_2 = get_second_adj(dgl_graph, A, subgraph_size_2)
    else:
        raise NotImplementedError
    return subgraphs_1, subgraphs_2

with tqdm(total=args.epochs) as pbar:
    pbar.set_description('Training')
    for epoch in range(args.epochs):
        model.train()
        all_idx = list(range(nb_nodes))
        random.shuffle(all_idx)
        loss_1 = 0.
        loss_2 = 0.
        loss_3 = 0.
        loss_record = 0.
        total_loss = 0.
        subgraphs_1, subgraphs_2 = generate_subgraph(args, dgl_graph, A, subgraph_size_1, subgraph_size_2)
        for batch_idx in range(batch_num):
            optimiser.zero_grad()
            is_final_batch = (batch_idx == (batch_num - 1))
            if not is_final_batch:
                idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            else:
                idx = all_idx[batch_idx * batch_size:]
            cur_batch_size = len(idx)
            lbl = torch.unsqueeze(
                torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio))), 1)
            ba = []
            ba_2 = []
            bf = []
            bf_2 = []
            raw = []
            raw_2 = []
            subgraph_idx = []
            subgraph_idx_2 = []
            Z_l = torch.full((cur_batch_size,), 1.)
            added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size_1))
            added_adj_zero_row_2 = torch.zeros((cur_batch_size, 1, subgraph_size_2))
            added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size_1 + 1, 1))
            added_adj_zero_col_2 = torch.zeros((cur_batch_size, subgraph_size_2 + 1, 1))
            added_adj_zero_col[:, -1, :] = 1.
            added_adj_zero_col_2[:, -1, :] = 1.
            added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))
            if torch.cuda.is_available():
                Z_l = Z_l.to(device)
                lbl = lbl.to(device)
                added_adj_zero_row = added_adj_zero_row.to(device)
                added_adj_zero_col = added_adj_zero_col.to(device)
                added_adj_zero_row_2 = added_adj_zero_row_2.to(device)
                added_adj_zero_col_2 = added_adj_zero_col_2.to(device)
                added_feat_zero_row = added_feat_zero_row.to(device)
            for i in idx:
                cur_adj = adj[:, subgraphs_1[i], :][:, :, subgraphs_1[i]]
                cur_adj_2 = adj[:, subgraphs_2[i], :][:, :, subgraphs_2[i]]
                cur_feat = features[:, subgraphs_1[i], :]
                cur_feat_2 = features[:, subgraphs_2[i], :]
                raw_f = raw_feature[:, subgraphs_1[i], :]
                raw_f_2 = raw_feature[:, subgraphs_2[i], :]
                ba.append(cur_adj)
                ba_2.append(cur_adj_2)
                bf.append(cur_feat)
                bf_2.append(cur_feat_2)
                raw.append(raw_f)
                raw_2.append(raw_f_2)
                subgraph_idx.append(subgraphs_1[i])
                subgraph_idx_2.append(subgraphs_2[i])
            ba = torch.cat(ba)
            ba_2 = torch.cat(ba_2)
            ba = torch.cat((ba, added_adj_zero_row), dim=1)
            ba = torch.cat((ba, added_adj_zero_col), dim=2)
            ba_2 = torch.cat((ba_2, added_adj_zero_row_2), dim=1)
            ba_2 = torch.cat((ba_2, added_adj_zero_col_2), dim=2)

            bf = torch.cat(bf)
            bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)
            bf_2 = torch.cat(bf_2)
            bf_2 = torch.cat((bf_2[:, :-1, :], added_feat_zero_row, bf_2[:, -1:, :]), dim=1)

            raw = torch.cat(raw)
            raw = torch.cat((raw[:, :-1, :], added_feat_zero_row, raw[:, -1:, :]), dim=1)
            raw_2 = torch.cat(raw_2)
            raw_2 = torch.cat((raw_2[:, :-1, :], added_feat_zero_row, raw_2[:, -1:, :]), dim=1)
            
            subgraph_idx = torch.Tensor(subgraph_idx)
            subgraph_idx_2 = torch.Tensor(subgraph_idx_2)
            subgraph_idx = subgraph_idx.int()
            subgraph_idx_2 = subgraph_idx_2.int()
            if torch.cuda.is_available():
                subgraph_idx = subgraph_idx.to(device)
                subgraph_idx_2 = subgraph_idx_2.to(device)
            #/---------------------MODEL-----------------------/#
            node_recons_1, node_recons_2, disc_1, disc_2, inter_loss_1, inter_loss_2, Intra_loss_1, Intra_loss_2, _, _, _, _, _, _, _, _, _, _ = \
                model(bf, ba, raw, subgraph_size_1 - 1, bf_2, ba_2, raw_2, subgraph_size_2 - 1,
                        full_topology_dist, subgraph_idx, subgraph_idx_2)
            loss_recon = 0.5 * (mse_loss(node_recons_1, raw[:, -1, :]) + mse_loss(node_recons_2, raw_2[:, -1, :]))
            intra_loss_1 = b_xent(disc_1, lbl)
            intra_loss_2 = b_xent(disc_2, lbl)
            loss_intra = torch.mean((intra_loss_1 + intra_loss_2) / 2)
            loss_inter = torch.mean((inter_loss_1 + inter_loss_2) / 2)
            loss_node = torch.mean((Intra_loss_1 + Intra_loss_2) / 2)
            loss = alpha_recon * loss_recon + alpha_inter * loss_inter + alpha_intra * loss_intra + loss_node

            loss.backward()
            optimiser.step()
            loss = loss.detach().cpu().numpy()
            if not is_final_batch:
                total_loss += loss
        mean_loss = (total_loss * batch_size + loss * cur_batch_size) / nb_nodes
        if mean_loss < best:
            best = mean_loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'pkl/model(cora)' + args.dataset + '.pkl')
        else:
            cnt_wait += 1
        pbar.set_postfix(loss=mean_loss)
        pbar.update(1)
path = 'pkl/model(cora)' + args.dataset + '.pkl'
model.load_state_dict(torch.load(path))
multi_round_attr_ano_score = np.zeros((args.auc_test_rounds, nb_nodes))
# batch批次个元素
nodes_embed1 = torch.zeros([nb_nodes, args.embedding_dim], dtype=torch.float).cuda()
nodes_embed2 = torch.zeros([nb_nodes, args.embedding_dim], dtype=torch.float).cuda()

with tqdm(total=args.auc_test_rounds) as pbar_test:
    pbar_test.set_description('Testing')
    for round in range(args.auc_test_rounds):
        all_idx = list(range(nb_nodes))
        random.shuffle(all_idx)
        subgraphs_1, subgraphs_2 = generate_subgraph(args, dgl_graph, A,subgraph_size_1, subgraph_size_2)
        for batch_idx in range(batch_num):
            optimiser.zero_grad()
            is_final_batch = (batch_idx == (batch_num - 1))
            if not is_final_batch:
                idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            else:
                idx = all_idx[batch_idx * batch_size:]
            cur_batch_size = len(idx)
            ba = []
            bf = []
            bf_2 = []
            ba_2 = []
            raw = []
            raw_2 = []
            subgraph_idx = []
            subgraph_idx_2 = []
            added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size_1))
            added_adj_zero_row_2 = torch.zeros((cur_batch_size, 1, subgraph_size_2))
            added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size_1 + 1, 1))
            added_adj_zero_col_2 = torch.zeros((cur_batch_size, subgraph_size_2 + 1, 1))
            added_adj_zero_col[:, -1, :] = 1.
            added_adj_zero_col_2[:, -1, :] = 1.
            added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))
            if torch.cuda.is_available():
                added_adj_zero_row = added_adj_zero_row.to(device)
                added_adj_zero_row_2 = added_adj_zero_row_2.to(device)
                added_adj_zero_col = added_adj_zero_col.to(device)
                added_adj_zero_col_2 = added_adj_zero_col_2.to(device)
                added_feat_zero_row = added_feat_zero_row.to(device)
            for i in idx:
                cur_adj = adj[:, subgraphs_1[i], :][:, :, subgraphs_1[i]]
                cur_adj2 = adj[:, subgraphs_2[i], :][:, :, subgraphs_2[i]]
                cur_feat = features[:, subgraphs_1[i], :]
                raw_f = raw_feature[:, subgraphs_1[i], :]
                cur_feat_2 = features[:, subgraphs_2[i], :]
                raw_f_2 = raw_feature[:, subgraphs_2[i], :]
                ba.append(cur_adj)
                ba_2.append(cur_adj2)
                bf.append(cur_feat)
                bf_2.append(cur_feat_2)
                raw.append(raw_f)
                raw_2.append(raw_f_2)
                subgraph_idx.append(subgraphs_1[i])
                subgraph_idx_2.append(subgraphs_2[i])
            ba = torch.cat(ba)
            ba = torch.cat((ba, added_adj_zero_row), dim=1)
            ba = torch.cat((ba, added_adj_zero_col), dim=2)
            ba_2 = torch.cat(ba_2)
            ba_2 = torch.cat((ba_2, added_adj_zero_row_2), dim=1)
            ba_2 = torch.cat((ba_2, added_adj_zero_col_2), dim=2)
            bf = torch.cat(bf)
            bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)
            bf_2 = torch.cat(bf_2)
            bf_2 = torch.cat((bf_2[:, :-1, :], added_feat_zero_row, bf_2[:, -1:, :]), dim=1)
            raw = torch.cat(raw)
            raw = torch.cat((raw[:, :-1, :], added_feat_zero_row, raw[:, -1:, :]), dim=1)
            raw_2 = torch.cat(raw_2)
            raw_2 = torch.cat((raw_2[:, :-1, :], added_feat_zero_row, raw_2[:, -1:, :]), dim=1)
            subgraph_idx = torch.Tensor(subgraph_idx)
            subgraph_idx_2 = torch.Tensor(subgraph_idx_2)
            subgraph_idx = subgraph_idx.int()
            subgraph_idx_2 = subgraph_idx_2.int()
            if torch.cuda.is_available():
                subgraph_idx = subgraph_idx.to(device)
                subgraph_idx_2 = subgraph_idx_2.to(device)
            # /---------------------MODEL-----------------------/#
            with torch.no_grad():
                node_res_1, node_res_2, logits_1, logits_2, inter_loss_1, inter_loss_2, Intra_loss_1, Intra_loss_2, sim_all_1, sim_all_2, \
                sim_pos_1, sim_pos_2,  sim_all_node1, sim_pos_node1, sim_all_node2, sim_pos_node2, batch_embed1, batch_embed2 = model(bf, ba, raw, subgraph_size_1 - 1, bf_2, ba_2, raw_2, subgraph_size_2 - 1,
                                                full_topology_dist, subgraph_idx, subgraph_idx_2)
                logits_1 = torch.squeeze(logits_1)
                logits_1 = torch.sigmoid(logits_1)

                logits_2 = torch.squeeze(logits_2)
                logits_2 = torch.sigmoid(logits_2)
                if round == args.auc_test_rounds - 1:
                    nodes_embed1[idx] = batch_embed1
                    nodes_embed2[idx] = batch_embed2
            pdist = nn.PairwiseDistance(p=2)
            scaler1 = MinMaxScaler()
            scaler2 = MinMaxScaler()
            scaler3 = MinMaxScaler()
            scaler4 = MinMaxScaler()
            score_co1 = - (logits_1[:cur_batch_size] - logits_1[cur_batch_size:]).cpu().numpy()
            score_co2 = - (logits_2[:cur_batch_size] - logits_2[cur_batch_size:]).cpu().numpy()
            score_co = (score_co1 + score_co2) / 2
            score_re = (pdist(node_res_1, raw[:, -1, :]) + pdist(node_res_2, raw_2[:, -1, :])) / 2
            score_re = score_re.cpu().numpy()
            score_ot = - (sim_pos_1 + sim_pos_2) / 2
            score_ot = score_ot.cpu().numpy()
            score_otx = - (sim_pos_node1 + sim_pos_node2) / 2
            score_otx = score_otx.cpu().numpy()
            #nomalize
            ano_score_co = scaler1.fit_transform(score_co.reshape(-1, 1)).reshape(-1)
            ano_score_re = scaler2.fit_transform(score_re.reshape(-1, 1)).reshape(-1)
            ano_score_ot = scaler3.fit_transform(score_ot.reshape(-1, 1)).reshape(-1)
            ano_score_otx = scaler4.fit_transform(score_otx.reshape(-1, 1)).reshape(-1)
            ano_scores = ano_score_co + alpha_recon * ano_score_re + alpha_inter * ano_score_ot + ano_score_otx  # anomaly score have ot(pos)

            multi_round_attr_ano_score[round, idx] = ano_scores
        pbar_test.update(1)
attr_ano_score_final = np.mean(multi_round_attr_ano_score, axis=0)
attr_scaler = MinMaxScaler()
attr_ano_score_final = attr_scaler.fit_transform(attr_ano_score_final.reshape(-1, 1)).reshape(-1)
    
# topology anomaly scores
# torch.Size([128, 3, 64])
# batch_size * sub_graph_size+1 * feature_dim
features1_norm = F.normalize(nodes_embed1, p=2, dim=1)
features1_similarity = torch.matmul(features1_norm, features1_norm.transpose(0, 1)).squeeze(0).cpu()
features2_norm = F.normalize(nodes_embed2, p=2, dim=1)
features2_similarity = torch.matmul(features2_norm, features2_norm.transpose(0, 1)).squeeze(0).cpu()
k_init = int(degree_ave)
net = nx.from_numpy_matrix(adj_raw)
net.remove_edges_from(nx.selfloop_edges(net))
adj_raw = nx.to_numpy_matrix(net)
multi_round_stru_ano_score = []
while 1:
    list_temp = list(nx.k_core(net, k_init))
    if list_temp == []:
        break
    else:
        core_adj = adj_raw[list_temp, :][:, list_temp]
        core_graph = nx.from_numpy_matrix(core_adj)
        list_temp = np.array(list_temp)
        for i in nx.connected_components(core_graph):
            core_temp = list(i)
            core_temp = list_temp[core_temp]
            core_temp_size = len(core_temp)
            similar_temp = 0
            similar_num = 0
            scores_temp = np.zeros(nb_nodes)
            for idx in core_temp:
                for idy in core_temp:
                    if idx != idy:
                        similar_temp += (features1_similarity[idx][idy] + features2_similarity[idx][idy]) / 2
                        similar_num += 1
            scores_temp[core_temp] = core_temp_size * 1 / (similar_temp / similar_num)
            multi_round_stru_ano_score.append(scores_temp)
        k_init += 1
if list_temp != []:
    multi_round_stru_ano_score = np.array(multi_round_stru_ano_score)
    multi_round_stru_ano_score = np.mean(multi_round_stru_ano_score, axis=0)
    stru_scaler = MinMaxScaler()
    stru_ano_score_final = stru_scaler.fit_transform(multi_round_stru_ano_score.reshape(-1, 1)).reshape(-1)
    alpha_list = list(np.arange(0, 1, 0.2))
    rate_auc = []
    for alpha in alpha_list:
        final_scores_rate = alpha * attr_ano_score_final + (1 - alpha) * stru_ano_score_final
        auc_temp = roc_auc_score(ano_label, final_scores_rate)
        rate_auc.append(auc_temp)
    max_alpha = alpha_list[rate_auc.index(max(rate_auc))]
    final_scores_rate = max_alpha * attr_ano_score_final + (1 - max_alpha) * stru_ano_score_final
    best_auc = roc_auc_score(ano_label, final_scores_rate)
    print('Alpha: ', max_alpha)
    print('AUC:{:.4f}'.format(best_auc))
    print('\n')
else:
    final_scores_rate = attr_ano_score_final
    best_auc = roc_auc_score(ano_label, final_scores_rate)
    print('AUC:{:.4f}'.format(best_auc))
    print('\n')