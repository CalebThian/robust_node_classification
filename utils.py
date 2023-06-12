import os
import argparse
import random
import psutil
import yaml
import logging
from functools import partial
from tensorboardX import SummaryWriter
import wandb

import numpy as np
import torch
import torch.nn as nn
from torch import optim as optim


import dgl
import dgl.function as fn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def accuracy(y_pred, y_true):
    """Return accuracy of output compared to labels.
    Parameters
    ----------
    y_pred : torch.Tensor
        output from model
    y_true : torch.Tensor or numpy.array
        node labels
    Returns
    -------
    float
        accuracy
    """
    if not hasattr(y_true, '__len__'):
        y_true = [y_true]
    if type(y_true) is not torch.Tensor:
        y_true = torch.LongTensor(y_true)
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True


def get_current_lr(optimizer):
    return optimizer.state_dict()["param_groups"][0]["lr"]


def build_args():
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max_epoch", type=int, default=500,
                        help="number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=-1)

    parser.add_argument("--num_heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num_dec_layers", type=int, default=1)
    parser.add_argument("--num_remasking", type=int, default=3)    
    parser.add_argument("--num_hidden", type=int, default=512,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in_drop", type=float, default=.2,
                        help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=.1,
                        help="attention dropout")
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="weight decay")
    parser.add_argument("--negative_slope", type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument("--activation", type=str, default="prelu")
    parser.add_argument("--mask_rate", type=float, default=0.5)
    parser.add_argument("--remask_rate", type=float, default=0.5)
    parser.add_argument("--remask_method", type=str, default="random")
    parser.add_argument("--mask_type", type=str, default="mask",
                        help="`mask` or `drop`")
    parser.add_argument("--mask_method", type=str, default="random")
    parser.add_argument("--drop_edge_rate", type=float, default=0.0)
    parser.add_argument("--drop_edge_rate_f", type=float, default=0.0)

    parser.add_argument("--encoder", type=str, default="gat")
    parser.add_argument("--decoder", type=str, default="gat")
    parser.add_argument("--loss_fn", type=str, default="sce")
    parser.add_argument("--alpha_l", type=float, default=2)
    parser.add_argument("--optimizer", type=str, default="adam")
    
    parser.add_argument("--max_epoch_f", type=int, default=300)
    parser.add_argument("--lr_f", type=float, default=0.01)
    parser.add_argument("--weight_decay_f", type=float, default=0.0)
    parser.add_argument("--linear_prob", action="store_true", default=False)

    
    parser.add_argument("--no_pretrain", action="store_true")
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--logging", action="store_true")
    parser.add_argument("--scheduler", action="store_true", default=False)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--batch_size_f", type=int, default=128)
    parser.add_argument("--sampling_method", type=str, default="saint", help="sampling method, `lc` or `saint`")

    #parser.add_argument("--label_rate", type=float, default=1.0)
    parser.add_argument("--ego_graph_file_path", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="data")

    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--full_graph_forward", action="store_true", default=False)
    parser.add_argument("--delayed_ema_epoch", type=int, default=0)
    parser.add_argument("--replace_rate", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.996)
    
    # Attack
    parser.add_argument('--attack', type=str, default='random',
                    choices=['meta', 'random', 'nettack','none'])
    parser.add_argument("--label_rate", type=float, default=0.01, 
                        help='rate of labeled data')
    parser.add_argument('--ptb_rate', type=float, default=0.15, 
                        help="noise ptb_rate")
    args = parser.parse_args()
    return args
    
def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name == "selu":
        return nn.SELU()
    elif name == "elu":
        return nn.ELU()
    elif name == "silu":
        return nn.SiLU()
    elif name is None:
        return nn.Identity()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def identity_norm(x):
    def func(x):
        return x
    return func

def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "identity":
        return identity_norm
    else:
        # print("Identity norm")
        return None


def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()
    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]

    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        raise NotImplementedError("Invalid optimizer")

    return optimizer


def show_occupied_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2


# -------------------
def mask_edge(graph, mask_prob):
    E = graph.num_edges()

    mask_rates = torch.ones(E) * mask_prob
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


def drop_edge(graph, drop_rate, return_edges=False):
    if drop_rate <= 0:
        return graph

    graph = graph.remove_self_loop()

    n_node = graph.num_nodes()
    edge_mask = mask_edge(graph, drop_rate)
    src, dst = graph.edges()

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
    ng = ng.add_self_loop()

    return ng


def visualize(x, y, method="tsne"):
    if torch.is_tensor(x):
        x = x.cpu().numpy()
        
    if torch.is_tensor(y):
        y = y.cpu().numpy()
    
    if method == "tsne":
        func = TSNE(n_components=2)
    else:
        func = PCA(n_components=2)
    out = func.fit_transform(x)
    plt.scatter(out[:, 0], out[:, 1], c=y)
    plt.savefig("vis.png")
    

def load_best_configs(args):
    dataset_name = args.dataset
    config_path = os.path.join("configs", f"{dataset_name}.yaml")
    with open(config_path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    logging.info(f"----- Using best configs from {config_path} -----")

    return args



def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    scheduler = np.concatenate((warmup_schedule, schedule))
    assert len(scheduler) == epochs * niter_per_ep
    return scheduler

    

# ------ logging ------

class TBLogger(object):
    def __init__(self, log_path="./logging_data", name="run"):
        super(TBLogger, self).__init__()

        if not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok=True)

        self.last_step = 0
        self.log_path = log_path
        raw_name = os.path.join(log_path, name)
        name = raw_name
        for i in range(1000):
            name = raw_name + str(f"_{i}")
            if not os.path.exists(name):
                break
        self.writer = SummaryWriter(logdir=name)

    def note(self, metrics, step=None):
        if step is None:
            step = self.last_step
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
        self.last_step = step

    def finish(self):
        self.writer.close()


class WandbLogger(object):
    def __init__(self, log_path, project, args):
        self.log_path = log_path
        self.project = project
        self.args = args
        self.last_step = 0
        self.project = project
        self.start()

    def start(self):
        self.run = wandb.init(config=self.args, project=self.project)

    def log(self, metrics, step=None):
        if not hasattr(self, "run"):
            self.start()
        if step is None:
            step = self.last_step
        self.run.log(metrics)
        self.last_step = step

    def finish(self):
        self.run.finish()
        
# ------ NR-GNN ------
def build_uniform_P(size, noise):
    """ The noise matrix flips any class to any other with probability
    noise / (#class - 1).
    """

    assert(noise >= 0.) and (noise <= 1.)

    P = np.float64(noise) / np.float64(size - 1) * np.ones((size, size))
    np.fill_diagonal(P, (np.float64(1)-np.float64(noise))*np.ones(size))
    
    diag_idx = np.arange(size)
    P[diag_idx,diag_idx] = P[diag_idx,diag_idx] + 1.0 - P.sum(0)
    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P

def build_pair_p(size, noise):
    assert(noise >= 0.) and (noise <= 1.)
    P = (1.0 - np.float64(noise)) * np.eye(size)
    for i in range(size):
        P[i,i-1] = np.float64(noise)
    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P


def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """

    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    sparserow=torch.LongTensor(sparse_mx.row).unsqueeze(1)
    sparsecol=torch.LongTensor(sparse_mx.col).unsqueeze(1)
    sparseconcat=torch.cat((sparserow, sparsecol),1)
    sparsedata=torch.FloatTensor(sparse_mx.data)
    return torch.sparse.FloatTensor(sparseconcat.t(),sparsedata,torch.Size(sparse_mx.shape))

def noisify(y, p_minus, p_plus=None, random_state=0):
    """ Flip labels with probability p_minus.
    If p_plus is given too, the function flips with asymmetric probability.
    """

    assert np.all(np.abs(y) == 1)

    m = y.shape[0]
    new_y = y.copy()
    coin = np.random.RandomState(random_state)

    if p_plus is None:
        p_plus = p_minus

    # This can be made much faster by tossing all the coins and completely
    # avoiding the loop. Although, it is not simple to write the asymmetric
    # case then.
    for idx in np.arange(m):
        if y[idx] == -1:
            if coin.binomial(n=1, p=p_minus, size=1) == 1:
                new_y[idx] = -new_y[idx]
        else:
            if coin.binomial(n=1, p=p_plus, size=1) == 1:
                new_y[idx] = -new_y[idx]

    return new_y

def noisify_with_P(y_train, nb_classes, noise, random_state=None,  noise_type='uniform'):

    if noise > 0.0:
        if noise_type=='uniform':
            print('Uniform noise')
            P = build_uniform_P(nb_classes, noise)
        elif noise_type == 'pair':
            print('Pair noise')
            P = build_pair_p(nb_classes, noise)
        else:
            print('Noise type have implemented')
        # seed the random numbers with #run
        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    else:
        P = np.eye(nb_classes)

    return y_train, P

def to_onehot(labels):
    class_size = labels.max() + 1
    onehot = np.eye(class_size)
    
    return onehot[labels]

# %%
import os
def load_emd(path, dataset):

    graph_embedding = np.genfromtxt(
            os.path.join(path,"{}.emb".format(dataset)),
            skip_header=1,
            dtype=float)
    embedding = np.zeros([graph_embedding.shape[0],graph_embedding.shape[1]-1])

    for i in range(graph_embedding.shape[0]):
        embedding[int(graph_embedding[i,0])] = graph_embedding[i,1:]
    
    return embedding

# ------ RS-GNN ------
def idx_to_mask(indices, n):
    mask = torch.zeros(n, dtype=torch.bool)
    mask[indices] = True
    return mask