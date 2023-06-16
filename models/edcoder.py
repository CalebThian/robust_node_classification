from itertools import chain

from typing import Optional
import torch
import torch.nn as nn
from functools import partial

from .gat import GAT
from .GCN import GCN

from .loss_func import sce_loss

import torch_geometric.utils as utils


def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead, nhead_out, attn_drop, negative_slope=0.2, concat_out=True, **kwargs) -> nn.Module:
    if m_type in ("gat", "tsgat"):
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
            **kwargs,
        )
    elif m_type == "mlp":
        # * just for decoder 
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden * 2),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden * 2, out_dim)
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError
    
    return mod

class PreModel(nn.Module):
    def __init__(
            self,
            args,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            num_dec_layers: int,
            num_remasking: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            mask_rate: float = 0.3,
            remask_rate: float = 0.5,
            remask_method: str = "random",
            mask_method: str = "random",
            encoder_type: str = "gat",
            decoder_type: str = "gat",
            loss_fn: str = "byol",
            drop_edge_rate: float = 0.0,
            alpha_l: float = 2,
            lam: float = 1.0,
            delayed_ema_epoch: int = 0,
            momentum: float = 0.996,
            replace_rate: float = 0.0,
            zero_init: bool = False,
            graph = None,
            x = None,
            # need to add graph information for edge predictor construction
         ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate
        self._remask_rate = remask_rate
        self._mask_method = mask_method
        self._alpha_l = alpha_l
        self._delayed_ema_epoch = delayed_ema_epoch

        self.num_remasking = num_remasking
        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._momentum = momentum
        self._replace_rate = replace_rate
        self._num_remasking = num_remasking
        self._remask_method = remask_method

        self._token_rate = 1 - self._replace_rate
        self._lam = lam

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat",):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead if decoder_type in ("gat",) else num_hidden 

        # edge predictor
        if graph != None and x != None:
            self.args = args
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            self.model = GCN(nfeat=enc_num_hidden,
                nhid=args.hidden,
                nclass=graph.ndata['labels'].max().item() + 1,
                self_loop=True,
                dropout=args.dropout, device=device).to(device)

            if args.estimator=='MLP':
                estimator = nn.Sequential(nn.Linear(enc_num_hidden,args.mlp_hidden),
                                        nn.ReLU(),
                                        nn.Linear(args.mlp_hidden,args.mlp_hidden))
            else:
                estimator = GCN(enc_num_hidden, args.mlp_hidden, args.mlp_hidden,dropout=0.0,device=device)
            self.estimator = EstimateAdj(estimator, args, device=self.device).to(self.device)

            self.optimizer_adj = optim.Adam(self.estimator.parameters(),lr=args.lr_adj, weight_decay=args.weight_decay)
        ##
        
        # build encoder
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

        self.decoder = setup_module(
                m_type=decoder_type,
                enc_dec="decoding",
                in_dim=dec_in_dim,
                num_hidden=dec_num_hidden,
                out_dim=in_dim,
                nhead_out=nhead_out,
                num_layers=num_dec_layers,
                nhead=nhead,
                activation=activation,
                dropout=feat_drop,
                attn_drop=attn_drop,
                negative_slope=negative_slope,
                residual=residual,
                norm=norm,
                concat_out=True,
            )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        self.dec_mask_token = nn.Parameter(torch.zeros(1, num_hidden))

        self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)
        
        if not zero_init:
            self.reset_parameters_for_token()


        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)
        
        self.projector = nn.Sequential(
            nn.Linear(num_hidden, 256),
            nn.PReLU(),
            nn.Linear(256, num_hidden),
        )
        self.projector_ema = nn.Sequential(
            nn.Linear(num_hidden, 256),
            nn.PReLU(),
            nn.Linear(256, num_hidden),
        )
        self.predictor = nn.Sequential(
            nn.PReLU(),
            nn.Linear(num_hidden, num_hidden)
        )
        
        self.encoder_ema = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )
        self.encoder_ema.load_state_dict(self.encoder.state_dict())
        self.projector_ema.load_state_dict(self.projector.state_dict())

        for p in self.encoder_ema.parameters():
            p.requires_grad = False
            p.detach_()
        for p in self.projector_ema.parameters():
            p.requires_grad = False
            p.detach_()
        
        self.print_num_parameters()

    def print_num_parameters(self):
        num_encoder_params = [p.numel() for p in self.encoder.parameters() if  p.requires_grad]
        num_decoder_params = [p.numel() for p in self.decoder.parameters() if  p.requires_grad]
        num_params = [p.numel() for p in self.parameters() if p.requires_grad]

        print(f"num_encoder_params: {sum(num_encoder_params)}, num_decoder_params: {sum(num_decoder_params)}, num_params_in_total: {sum(num_params)}")

    def reset_parameters_for_token(self):
        nn.init.xavier_normal_(self.enc_mask_token)
        nn.init.xavier_normal_(self.dec_mask_token)
        nn.init.xavier_normal_(self.encoder_to_decoder.weight, gain=1.414)

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            print(f"=== Use mse_loss ===")
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            print(f"=== Use sce_loss and alpha_l={alpha_l} ===")
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def forward(self, g, x, targets=None, epoch=0, drop_g1=None, drop_g2=None):        # ---- attribute reconstruction ----
        loss = self.mask_attr_prediction(g, x, targets, epoch, drop_g1, drop_g2)

        return loss

    def mask_attr_prediction(self, g, x, targets, epoch, drop_g1=None, drop_g2=None):
        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)
        use_g = drop_g1 if drop_g1 is not None else g

        enc_rep = self.encoder(use_g, use_x,)
        
        # Embeddings
        # ---- Begin: Link Prediction ----
        for i in range(int(args.outer_steps)):
                # train_adj(self, epoch, features, edge_index, labels, idx_train, idx_val)
                ## epoch: i
                ## features: enc_rep
                ## edge_index: edge_index, _ = utils.from_scipy_sparse_matrix(adj)
                ###            edge_index = edge_index.to(self.device)
                ## labels:  graph.ndata['label']
                ## idx_train: graph.ndata['train_mask'].numpy()
                ## idx_val: graph.ndata['val_mask'].numpy()
                adj = utils.from_scipy_sparse_matrix(pre_use_g.adj(scipy_fmt='coo', etype='develops'))
                edge_index, _ = utils.from_scipy_sparse_matrix(adj)
                edge_index = edge_index.to(self.device)
                labels = graph.ndata['train_mask'].numpy()
                idx_train = graph.ndata['train_mask'].numpy()
                idx_val = graph.ndata['val_mask'].numpy()
                self.train_adj(i, enc_rep, edge_index, labels, idx_train, idx_val, g, keep_nodes) 
        # ---- End: Link Prediction ----
        
        with torch.no_grad():
            drop_g2 = drop_g2 if drop_g2 is not None else g
            latent_target = self.encoder_ema(drop_g2, x,)
            if targets is not None:
                latent_target = self.projector_ema(latent_target[targets])
            else:
                latent_target = self.projector_ema(latent_target[keep_nodes])

        if targets is not None:
            latent_pred = self.projector(enc_rep[targets])
            latent_pred = self.predictor(latent_pred)
            loss_latent = sce_loss(latent_pred, latent_target, 1)
        else:
            latent_pred = self.projector(enc_rep[keep_nodes])
            latent_pred = self.predictor(latent_pred)
            loss_latent = sce_loss(latent_pred, latent_target, 1)

        # ---- attribute reconstruction ----
        origin_rep = self.encoder_to_decoder(enc_rep)

        loss_rec_all = 0
        if self._remask_method == "random":
            for i in range(self._num_remasking):
                rep = origin_rep.clone()
                rep, remask_nodes, rekeep_nodes = self.random_remask(use_g, rep, self._remask_rate)
                recon = self.decoder(pre_use_g, rep)

                x_init = x[mask_nodes]
                x_rec = recon[mask_nodes]
                loss_rec = self.criterion(x_init, x_rec)
                loss_rec_all += loss_rec
            loss_rec = loss_rec_all
        elif self._remask_method == "fixed":
            rep = self.fixed_remask(g, origin_rep, mask_nodes)
            x_rec = self.decoder(pre_use_g, rep)[mask_nodes]
            x_init = x[mask_nodes]
            loss_rec = self.criterion(x_init, x_rec)
        else:
            raise NotImplementedError

        loss = loss_rec + self._lam * loss_latent

        if epoch >= self._delayed_ema_epoch:
            self.ema_update()
        return loss

    # ---- Begin: Edge Predictor Training ----
     def fit(self, features, adj, labels, idx_train, idx_val):
        """Train GraphMAE2 with Link Predictor.

        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices
        """
        args = self.args
        edge_index, _ = utils.from_scipy_sparse_matrix(adj)
        edge_index = edge_index.to(self.device)

        if sp.issparse(features):
            features = sparse_mx_to_torch_sparse_tensor(features).to_dense().float()
        else:
            features = torch.FloatTensor(np.array(features))
        features = features.to(self.device)
        labels = torch.LongTensor(np.array(labels)).to(self.device)

        self.features = features
        self.labels = labels


        self.estimator = EstimateAdj(edge_index, features, args, device=self.device).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)
        self.optimizer_adj = optim.Adam(self.estimator.parameters(),
                               lr=args.lr_adj, weight_decay=args.weight_decay)

        # Train model
        t_total = time.time()
        for epoch in range(args.epochs):
            for i in range(int(args.outer_steps)):
                self.train_adj(epoch, features, edge_index, labels,
                        idx_train, idx_val)

            for i in range(int(args.inner_steps)):
                self.train_gcn(epoch, features, edge_index,
                        labels, idx_train, idx_val)

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Testing
        print("picking the best model according to validation performance")
        self.model.load_state_dict(self.weights)

        print("=====validation set accuracy=======")
        self.test(idx_val)
        print("===================================")
    
    def train_adj(self, epoch, features, edge_index, labels, idx_train, idx_val, g, keep_nodes):
        args = self.args
        if args.debug:
            print("\n=== train_adj ===")
        t = time.time()

        self.optimizer_adj.zero_grad()

        rec_loss = self.estimator(edge_index, features)

        #output = self.model(features, self.estimator.poten_edge_index, self.estimator.estimated_weights)
        enc_rep = self.encoder(g, features,)
        with torch.no_grad():
            latent_target = self.encoder_ema(g, features,)
            latent_target = self.projector_ema(latent_target[keep_nodes])
            
            latent_pred = self.projector(enc_rep[keep_nodes])
            latent_pred = self.predictor(latent_pred)
            
            loss_latent = sce_loss(latent_pred, latent_target, 1)
            acc_train = accuracy(latent_pred, latent_target)
            #loss_gcn = F.cross_entropy(output, labels[idx_train])
            #acc_train = accuracy(output, labels[idx_train])
        
        ## Stop here
        loss_label_smooth = self.label_smoothing(self.estimator.poten_edge_index,\
                                                 self.estimator.estimated_weights.detach(),\
                                                 latent_pred, idx_train, self.args.threshold)


        total_loss = loss_latent + args.alpha *rec_loss + loss_label_smooth


        total_loss.backward()

        self.optimizer_adj.step()


        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        #self.model.eval()
        #output = self.model(features, self.estimator.poten_edge_index, self.estimator.estimated_weights.detach())
        with torch.no_grad():
            latent_target = self.encoder_ema(g, features,)
            latent_target = self.projector_ema(latent_target[idx_val])
            
            latent_pred = self.projector(enc_rep[idx_val])
            latent_pred = self.predictor(latent_pred)
            
            loss_val_latent = sce_loss(latent_pred, latent_target, 1)
            acc_val = accuracy(latent_pred, latent_target)
        
        #loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
        #acc_val = accuracy(output[idx_val], labels[idx_val])
        

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = self.estimator.estimated_weights.detach()
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print('\t=== saving current graph/gcn, best_val_acc: %s' % self.best_val_acc.item())


        if args.debug:
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch+1),
                      'loss_latent: {:.4f}'.format(loss_latent.item()),
                      'rec_loss: {:.4f}'.format(rec_loss.item()),
                      'loss_label_smooth: {:.4f}'.format(loss_label_smooth.item()),
                      'loss_total: {:.4f}'.format(total_loss.item()))
                print('Epoch: {:04d}'.format(epoch+1),
                        'acc_train: {:.4f}'.format(acc_train.item()),
                        'loss_val: {:.4f}'.format(loss_val_latent.item()),
                        'acc_val: {:.4f}'.format(acc_val.item()),
                        'time: {:.4f}s'.format(time.time() - t))
                
        if args.debug:
            print("\n=== end train_adj ===")
            
    def label_smoothing(self, edge_index, edge_weight, representations, idx_train, threshold):


        num_nodes = representations.shape[0]
        n_mask = torch.ones(num_nodes, dtype=torch.bool).to(self.device)
        n_mask[idx_train] = 0

        mask = n_mask[edge_index[0]] \
                & (edge_index[0] < edge_index[1])\
                & (edge_weight >= threshold)\
                | torch.bitwise_not(n_mask)[edge_index[1]]

        unlabeled_edge = edge_index[:,mask]
        unlabeled_weight = edge_weight[mask]

        Y = F.softmax(representations)

        loss_smooth_label = unlabeled_weight\
                            @ torch.pow(Y[unlabeled_edge[0]] - Y[unlabeled_edge[1]], 2).sum(dim=1)\
                            / num_nodes

        return loss_smooth_label
    # ---- End: Edge Predictor Training ----
    
    def ema_update(self):
        def update(student, teacher):
            with torch.no_grad():
            # m = momentum_schedule[it]  # momentum parameter
                m = self._momentum
                for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
        update(self.encoder, self.encoder_ema)
        update(self.projector, self.projector_ema)

    def embed(self, g, x):
        rep = self.encoder(g, x)
        return rep

    def get_encoder(self):
        #self.encoder.reset_classifier(out_size)
        return self.encoder
    
    def reset_encoder(self, out_size):
        self.encoder.reset_classifier(out_size)
 
    @property
    def enc_params(self):
        return self.encoder.parameters()
    
    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])

    def output_grad(self):
        grad_dict = {}
        for n, p in self.named_parameters():
            if p.grad is not None:
                grad_dict[n] = p.grad.abs().mean().item()
        return grad_dict
    
    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # exclude isolated nodes
        # isolated_nodes = torch.where(g.in_degrees() <= 1)[0]
        # mask_nodes = perm[: num_mask_nodes]
        # mask_nodes = torch.index_fill(torch.full((num_nodes,), False, device=device), 0, mask_nodes, True)
        # mask_nodes[isolated_nodes] = False
        # keep_nodes = torch.where(~mask_nodes)[0]
        # mask_nodes = torch.where(mask_nodes)[0]
        # num_mask_nodes = mask_nodes.shape[0]

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        out_x = x.clone()
        token_nodes = mask_nodes
        out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)
    
    def random_remask(self,g,rep,remask_rate=0.5):
        
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=rep.device)
        num_remask_nodes = int(remask_rate * num_nodes)
        remask_nodes = perm[: num_remask_nodes]
        rekeep_nodes = perm[num_remask_nodes: ]

        rep = rep.clone()
        rep[remask_nodes] = 0
        rep[remask_nodes] += self.dec_mask_token

        return rep, remask_nodes, rekeep_nodes

    def fixed_remask(self, g, rep, masked_nodes):
        rep[masked_nodes] = 0
        return rep
    
class EstimateAdj(nn.Module):
    """Provide a pytorch parameter matrix for estimated
    adjacency matrix and corresponding operations.
    """

    def __init__(self, estimator, args ,device='cuda'):
        super(EstimateAdj, self).__init__()

        '''
        if args.estimator=='MLP':
            self.estimator = nn.Sequential(nn.Linear(features.shape[1],args.mlp_hidden),
                                    nn.ReLU(),
                                    nn.Linear(args.mlp_hidden,args.mlp_hidden))
        else:
            self.estimator = GCN(features.shape[1], args.mlp_hidden, args.mlp_hidden,dropout=0.0,device=device)
        '''
        self.estimator = estimator
        self.device = device
        self.args = args
        self.estimated_weights = None


    def get_poten_edge(self, edge_index, features, n_p):
        if n_p == 0:
            return edge_index

        poten_edges = []
        for i in range(len(features)):
            sim = torch.div(torch.matmul(features[i],features.T), features[i].norm()*features.norm(dim=1))
            _,indices = sim.topk(n_p)
            poten_edges.append([i,i])
            indices = set(indices.cpu().numpy())
            indices.update(edge_index[1,edge_index[0]==i])
            for j in indices:
                if j > i:
                    pair = [i,j]
                    poten_edges.append(pair)
        poten_edges = torch.as_tensor(poten_edges).T
        poten_edges = utils.to_undirected(poten_edges,len(features)).to(self.device)

        return poten_edges
    

    def forward(self, edge_index, features):
        self.poten_edge_index = self.get_poten_edge(edge_index,features,args.n_p)
        self.features_diff = torch.cdist(features,features,2)
        
        if self.args.estimator=='MLP':
            representations = self.estimator(features)
        else:
            representations = self.estimator(features,edge_index,\
                                            torch.ones([edge_index.shape[1]]).to(self.device).float())
        rec_loss = self.reconstruct_loss(edge_index, representations)

        x0 = representations[self.poten_edge_index[0]]
        x1 = representations[self.poten_edge_index[1]]
        output = torch.sum(torch.mul(x0,x1),dim=1)

        self.estimated_weights = F.relu(output)
        self.estimated_weights[self.estimated_weights < self.args.t_small] = 0.0
        

        return rec_loss
    
    def reconstruct_loss(self, edge_index, representations):
        
        num_nodes = representations.shape[0]
        randn = utils.negative_sampling(edge_index,num_nodes=num_nodes, num_neg_samples=self.args.n_n*num_nodes)
        randn = randn[:,randn[0]<randn[1]]

        edge_index = edge_index[:, edge_index[0]<edge_index[1]]
        neg0 = representations[randn[0]]
        neg1 = representations[randn[1]]
        neg = torch.sum(torch.mul(neg0,neg1),dim=1)

        pos0 = representations[edge_index[0]]
        pos1 = representations[edge_index[1]]
        pos = torch.sum(torch.mul(pos0,pos1),dim=1)

        neg_loss = torch.exp(torch.pow(self.features_diff[randn[0],randn[1]]/self.args.sigma,2)) @ F.mse_loss(neg,torch.zeros_like(neg), reduction='none')
        pos_loss = torch.exp(-torch.pow(self.features_diff[edge_index[0],edge_index[1]]/self.args.sigma,2)) @ F.mse_loss(pos, torch.ones_like(pos), reduction='none')

        rec_loss = (pos_loss + neg_loss) \
                    * num_nodes/(randn.shape[1] + edge_index.shape[1]) 
        

        return rec_loss