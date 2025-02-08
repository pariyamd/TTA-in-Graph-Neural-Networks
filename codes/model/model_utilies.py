import logging
import torch
from datetime import datetime
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch_sparse import SparseTensor
from sklearn.metrics import f1_score
import numpy as np

import os.path as osp
import torch
import numpy as np
from torch_geometric.data import (InMemoryDataset, Data)

from model.Data import DomainData
from torch_geometric.io import read_txt_array
from torch_geometric.utils import remove_self_loops

class CoraData(InMemoryDataset):
    def __init__(self,
                 root,
                 valid_ratio=0.2,
                 noise_ratio=0.0,  # Amount of noise to add (0.0 to 1.0)
                 noise_type='gaussian',  # Type of noise: 'gaussian', 'uniform', or 'flip'
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        
        self.valid_ratio = valid_ratio
        self.noise_ratio = noise_ratio
        self.noise_type = noise_type
        print(self.noise_ratio)
        super(CoraData, self).__init__(root, transform, pre_transform, pre_filter)
        print(self.noise_ratio)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['cora.content', 'cora.cites']

    @property
    def processed_file_names(self):
        # Different processed file name based on noise configuration
        if self.noise_ratio > 0:
            return [f'data_noise_{self.noise_type}_{self.noise_ratio}.pt']
        return ['data.pt']

    def add_noise(self, x):
        """Add noise to features of selected nodes.
        
        Args:
            x (torch.Tensor): Feature matrix
        """
        # Make a copy to avoid modifying original
        x_noisy = x.clone()
        
        if self.noise_type == 'gaussian':
            # Add Gaussian noise with same std as original features
            std = x.std(dim=0)
            noise = torch.randn_like(x) * std * self.noise_ratio
            x_noisy += noise
            
        elif self.noise_type == 'uniform':
            # Add uniform noise between -range and +range
            range_per_feature = (x.max(dim=0)[0] - x.min(dim=0)[0])
            noise = (torch.rand_like(x) * 2 - 1) * range_per_feature * self.noise_ratio
            x_noisy += noise
            
        elif self.noise_type == 'flip':
            # Randomly flip features (for binary features)
            flip_mask = torch.rand_like(x) < self.noise_ratio
            x_noisy[flip_mask] = 1 - x[flip_mask]
            
        return x_noisy

    def download(self):
        pass

    def process(self):
        # Read node features and labels
        content_path = osp.join(self.raw_dir, 'cora.content')
        data = np.genfromtxt(content_path, dtype=str)
        
        # Get indices
        indices = np.array(data[:, 0], dtype=np.int32)
        
        # Get features
        features = np.array(data[:, 1:-1], dtype=np.float32)
        x = torch.from_numpy(features)
        
        # Get labels
        labels = np.array(data[:, -1], dtype=str)
        unique_labels = np.unique(labels)
        label_dict = {label: i for i, label in enumerate(unique_labels)}
        labels = np.array([label_dict[label] for label in labels])
        y = torch.from_numpy(labels).to(torch.int64)

        # Read edge index
        edge_path = osp.join(self.raw_dir, 'cora.cites')
        edge_index = read_txt_array(edge_path, sep='\t', dtype=torch.long).t()
        
        # Map edge indices to our new node indices
        idx_dict = {old_idx: new_idx for new_idx, old_idx in enumerate(indices)}
        edge_index = torch.tensor([[idx_dict[edge_index[0, i].item()], 
                                  idx_dict[edge_index[1, i].item()]]
                                 for i in range(edge_index.size(1))], dtype=torch.long).t()
        
        # Remove self-loops
        edge_index, _ = remove_self_loops(edge_index)

        # Create train/val/test masks
        num_nodes = y.shape[0]
        random_node_indices = np.random.permutation(num_nodes)
        
        train_size = int(num_nodes * (1 - self.valid_ratio))
        val_size = int(num_nodes * self.valid_ratio)
        
        train_idx = random_node_indices[:train_size]
        val_idx = random_node_indices[train_size:]
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True

        # Add noise to test node features if specified
        print(self.noise_ratio)
        if self.noise_ratio > 0:
            print("Adding noise to node features...")
            after_noise = self.add_noise(x)
            print((after_noise-x).mean())
            x = after_noise

        # Create data object
        data = Data(x=x, 
                   edge_index=edge_index, 
                   y=y,
                   train_mask=train_mask,
                   val_mask=val_mask,
                   noise_applied=self.noise_ratio > 0)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save((data, self.collate([data])[1]), self.processed_paths[0])

def init_layer(args, params, conf_params):
    layer_unit_count_list = [args.num_features] 
    
    if args.num_layers == 2:
        layer_unit_count_list.extend([100])
    elif args.num_layers == 3:
        layer_unit_count_list.extend([256, 128])
    elif args.num_layers == 4:
        layer_unit_count_list.extend([256, 128, 64])
    elif args.num_layers == 5:
        layer_unit_count_list.extend([32, 32, 32, 32])
    elif args.num_layers == 7:
        layer_unit_count_list.extend([256, 128, 64, 32, 32,16])
    elif args.num_layers == 6:
        layer_unit_count_list.extend([256, 256, 256, 256, 256])
    elif args.num_layers == 9: 
        layer_unit_count_list.extend([256, 128, 64, 64, 32, 32, 16, 16]) 
    layer_unit_count_list.append(args.num_label)
    
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vars(args)["device"] = device

    vars(args)["time"] = datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
    vars(args)["layer_unit_count_list"] = layer_unit_count_list
    
class ArrayLogger:
    def __init__(self, logger):
        self.logger = logger
        self.values = {"Loss": [], "Macro F1": [], "Micro F1": [], "Loss Encoder": [], "Loss Augmenter": []}
        self.logs = []
    
    def info(self, message, key=None,*args):
        formatted_message = message % args if args else message
        if key:
            self.values[key].append(formatted_message)
        self.logs.append(formatted_message)
        self.logger.info(formatted_message)


def init_log(args):
    # Your existing logger initialization code
    with open("./record/" + args.model_name+f"/{args.adaptation_method}_{args.random_seed}.log", 'w') as f:
        f.truncate()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("./record/" + args.model_name+f"/{args.adaptation_method}_{args.random_seed}.log")
    fh.setLevel(logging.INFO)
    # ch = logging.StreamHandler(sys.stdout)
    # ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    # ch.setFormatter(formatter)
    logger.addHandler(fh)
    # logger.addHandler(ch)
    logger.info("logger name:%s", args.model_name + ".log")
    # vars(args)["logger"] = logger
    
    array_logger = ArrayLogger(logger)
    vars(args)["logger"] = array_logger
    return array_logger


def save_model(args, prefix, model):
    torch.save({'model_state_dict': model.state_dict()}, f"record/{args.model_name}/{prefix}_{args.random_seed}.pkl")

def load_model(args, prefix, model):
    state_dict = torch.load(f"record/{args.model_name}/{prefix}_{args.random_seed}.pkl", map_location=args.device)["model_state_dict"]
    model.load_state_dict(state_dict)
    return model

def Entropy(input):
    batch_size, num_feature = input.size()
    epsilon = 1e-5
    entropy = -input * torch.log(input + epsilon)
    entropy = torch.sum(entropy, dim=1)

    return entropy 

def predict(output):
    labels = output.argmax(dim=1)
    
    return labels


def evaluate(output, labels, metric):
    preds = predict(output)
    corrects = preds.eq(labels)
    labels = labels.cpu().numpy()
    num_labels = np.max(labels) + 1
    preds = torch.argmax(output, dim = 1).cpu().numpy()
    macro_score = f1_score(labels, preds, average='macro')
    micro_score = f1_score(labels, preds, average='micro')

        
    if metric == "micro":
        score = micro_score
    elif metric == "macro":
        score  = macro_score
    else:
        print("wrong!")
        exit()

    return score


def cos_distance(input1, input2):
    norm1 = torch.norm(input1, dim = -1)
    norm2 = torch.norm(input2, dim = -1)
        
    norm1 = torch.unsqueeze(norm1, dim = 1)
    norm2 = torch.unsqueeze(norm2, dim = 0)

    cos_matrix = torch.matmul(input1, input2.t())
        
    cos_matrix /= norm1
    cos_matrix /= norm2

    return cos_matrix

def test(model, args, data, criterion, mode = 'valid'):
    outputs = model(data)

    if mode == 'valid':
        outputs = outputs[data.val_mask]
        labels = data.y[data.val_mask]
    else:
        labels = data.y

    loss = criterion(outputs, labels)
    micro = evaluate(outputs, labels, "micro")
    macro = evaluate(outputs, labels, "macro")

    return micro, macro

def generate_one_hot_label(labels):
    num_labels = torch.max(labels).item() + 1
    num_nodes = labels.shape[0]
    label_onehot = torch.zeros((num_nodes, num_labels)).cuda()
    label_onehot = F.one_hot(labels, num_labels).float().squeeze(1) 

    return label_onehot

def generate_normalized_adjs(adj, D_isqrt):
    DAD = D_isqrt.view(-1,1)*adj*D_isqrt.view(1,-1)
    DA = D_isqrt.view(-1,1) * D_isqrt.view(-1,1)*adj
    AD = adj*D_isqrt.view(1,-1) * D_isqrt.view(1,-1)
    return DAD, DA, AD

def process_adj(data):
    N = data.num_nodes
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    row, col = data.edge_index

    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    
    return adj, deg_inv_sqrt