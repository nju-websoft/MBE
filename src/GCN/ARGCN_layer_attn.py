import torch
from torch.nn.init import xavier_normal_
from torch_scatter import scatter_mean, scatter_max, scatter_sum

class ARGCN(torch.nn.Module):
    def __init__(self, args, data_box):
        super(ARGCN, self).__init__()
        self.args = args
        self.data_box = data_box
        torch.cuda.set_device(self.args.gpu)
        self.conv_layer = RelConvLayer(self.args, self.data_box)
        self.ent_conv_layers = torch.nn.ModuleList()
        for i in range(self.args.gcn_layer-1):
            self.ent_conv_layers.append(EntConvLayer(self.args, self.data_box))
        self.rel_embeddings = torch.nn.Embedding(self.data_box['all']['num_rel'], self.args.emb_dim)
        xavier_normal_(self.rel_embeddings.weight)

    def forward(self, batch, query=None):
        res, rel_res = self.conv_layer(batch, self.rel_embeddings.weight, query)
        ent_embeddings, rel_embeddings = res, rel_res
        for conv in self.ent_conv_layers:
            res, rel_res = conv(batch, res, rel_res, query)
            ent_embeddings, rel_embeddings = res, rel_res
        return ent_embeddings, rel_embeddings


class RelConvLayer(torch.nn.Module):
    def __init__(self, args, data_box):
        super(RelConvLayer, self).__init__()
        self.args = args
        self.data_box = data_box
        torch.cuda.set_device(self.args.gpu)

        self.w_rel = torch.nn.Parameter(torch.Tensor(self.args.emb_dim, self.args.emb_dim))
        xavier_normal_(self.w_rel)
        self.w_in, self.w_out = torch.nn.Parameter(torch.Tensor(self.args.emb_dim, self.args.emb_dim)), torch.nn.Parameter(torch.Tensor(self.args.emb_dim, self.args.emb_dim))
        xavier_normal_(self.w_in)
        xavier_normal_(self.w_out)

        self.ent_drop = torch.nn.Dropout(self.args.node_dropout)
        self.neigh_drop = torch.nn.Dropout(self.args.neigh_dropout)
        self.bn = torch.nn.BatchNorm1d(self.args.emb_dim)
        self.tanh = torch.nn.Tanh()

    def forward(self, batch, rel_embed, query=None):
        data = self.data_box[batch]
        if self.args.aug_link:
            edge_index, edge_type = data['edge_index_aug_link'], data['edge_type_aug_link']
        else:
            edge_index, edge_type = data['edge_index'], data['edge_type']
        edge_size = edge_index.size(1)//2
        edge_norm_in = self.compute_norm(edge_index[:, :edge_size], data['num_ent'])
        edge_norm_out = self.compute_norm(edge_index[:, edge_size:], data['num_ent'])
        neighs = torch.index_select(rel_embed, 0, edge_type)
        neighs_in = torch.mm(neighs[:edge_size], self.w_in)
        neighs_out = torch.mm(neighs[edge_size:], self.w_out)
        if query is not None:
            weight = self.args.attn_weight[query] + (1.2 - self.args.attn_reliability[query])
            neighs_in = weight[edge_type[:edge_size]].unsqueeze(0).t() * neighs_in
            neighs_out = weight[edge_type[edge_size:]].unsqueeze(0).t() * neighs_out
        if self.args.rel_agg == 'sum':
            res_in = scatter_sum(src=self.neigh_drop(neighs_in) * edge_norm_in.view(-1, 1), index=edge_index[0][:edge_size], dim=0, dim_size=data['num_ent'])
            res_out = scatter_sum(src=self.neigh_drop(neighs_out) * edge_norm_out.view(-1, 1), index=edge_index[0][edge_size:], dim=0,
                                    dim_size=data['num_ent'])
        elif self.args.rel_agg == 'mean':
            res_in = scatter_mean(src=self.neigh_drop(neighs_in) * edge_norm_in.view(-1, 1),
                                 index=edge_index[0][:edge_size], dim=0, dim_size=data['num_ent'])
            res_out = scatter_mean(src=self.neigh_drop(neighs_out) * edge_norm_out.view(-1, 1),
                                  index=edge_index[0][edge_size:], dim=0,
                                  dim_size=data['num_ent'])
        elif self.args.rel_agg == 'max':
            res_in, _ = scatter_max(src=self.neigh_drop(neighs_in) * edge_norm_in.view(-1, 1),
                                 index=edge_index[0][:edge_size], dim=0, dim_size=data['num_ent'])
            res_out, _ = scatter_max(src=self.neigh_drop(neighs_out) * edge_norm_out.view(-1, 1),
                                  index=edge_index[0][edge_size:], dim=0,
                                  dim_size=data['num_ent'])

        res = self.tanh(self.bn(1/2*self.ent_drop(res_in) + 1/2*self.ent_drop(res_out)))
        return res, rel_embed

    def compute_norm(self, edge_index, num_ent):
        row, col = edge_index
        edge_weight = torch.ones_like(row).float()
        deg = scatter_sum(src=edge_weight, index=row, dim=0, dim_size=num_ent)  # Summing number of weights of the edges
        deg_inv = deg.pow(-0.5)  # D^{-0.5}
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col]  # D^{-0.5}
        return norm

class EntConvLayer(torch.nn.Module):
    def __init__(self, args, data_box):
        super(EntConvLayer, self).__init__()
        self.args = args
        self.data_box = data_box
        torch.cuda.set_device(self.args.gpu)

        self.w_rel = torch.nn.Parameter(torch.Tensor(self.args.emb_dim, self.args.emb_dim))
        xavier_normal_(self.w_rel)
        self.w_in, self.w_out, self.w_loop = torch.nn.Parameter(
            torch.Tensor(self.args.emb_dim, self.args.emb_dim)), torch.nn.Parameter(
            torch.Tensor(self.args.emb_dim, self.args.emb_dim)), torch.nn.Parameter(
            torch.Tensor(self.args.emb_dim, self.args.emb_dim))
        xavier_normal_(self.w_in)
        xavier_normal_(self.w_out)
        xavier_normal_(self.w_loop)

        self.ent_drop = torch.nn.Dropout(self.args.node_dropout)
        self.neigh_drop = torch.nn.Dropout(self.args.neigh_dropout)
        self.bn = torch.nn.BatchNorm1d(self.args.emb_dim)
        self.tanh = torch.nn.Tanh()

    def forward(self, batch, x, rel_embed, query=None):
        data = self.data_box[batch]
        if self.args.aug_link:
            edge_index, edge_type = data['edge_index_aug_link'], data['edge_type_aug_link']
        else:
            edge_index, edge_type = data['edge_index'], data['edge_type']
        edge_size = edge_index.size(1) // 2

        loop_index = torch.arange(data['num_ent']).long().cuda()
        neighs_loop = torch.index_select(x, 0, loop_index)
        edge_norm_in = self.compute_norm(edge_index[:, :edge_size], data['num_ent'])
        edge_norm_out = self.compute_norm(edge_index[:, edge_size:], data['num_ent'])
        neighs = torch.index_select(x, 0, edge_index[0])
        neighs_in = torch.mm(neighs[:edge_size], self.w_in)
        neighs_out = torch.mm(neighs[edge_size:], self.w_out)
        neighs_loop = torch.mm(neighs_loop, self.w_loop)
        if query is not None:
            weight = self.args.attn_weight[query] + (1.0 - self.args.attn_reliability[query])
            neighs_in = weight[edge_type[:edge_size]].unsqueeze(0).t() * neighs_in
            neighs_out = weight[edge_type[edge_size:]].unsqueeze(0).t() * neighs_out

        if self.args.ent_agg == 'sum':
            res_in = scatter_sum(src=self.neigh_drop(neighs_in) * edge_norm_in.view(-1, 1),
                                    index=edge_index[0][:edge_size], dim=0, dim_size=data['num_ent'])
            res_out = scatter_sum(src=self.neigh_drop(neighs_out) * edge_norm_out.view(-1, 1),
                                     index=edge_index[0][edge_size:], dim=0,
                                     dim_size=data['num_ent'])
            res_loop = scatter_sum(src=self.neigh_drop(neighs_loop), index=loop_index, dim=0, dim_size=data['num_ent'])
        elif self.args.ent_agg == 'mean':
            res_in = scatter_mean(src=self.neigh_drop(neighs_in) * edge_norm_in.view(-1, 1),
                                 index=edge_index[0][:edge_size], dim=0, dim_size=data['num_ent'])
            res_out = scatter_mean(src=self.neigh_drop(neighs_out) * edge_norm_out.view(-1, 1),
                                  index=edge_index[0][edge_size:], dim=0,
                                  dim_size=data['num_ent'])
            res_loop = scatter_mean(src=self.neigh_drop(neighs_loop), index=loop_index, dim=0, dim_size=data['num_ent'])
        elif self.args.ent_agg == 'max':
            res_in, _ = scatter_max(src=self.neigh_drop(neighs_in) * edge_norm_in.view(-1, 1),
                                 index=edge_index[0][:edge_size], dim=0, dim_size=data['num_ent'])
            res_out, _ = scatter_max(src=self.neigh_drop(neighs_out) * edge_norm_out.view(-1, 1),
                                  index=edge_index[0][edge_size:], dim=0,
                                  dim_size=data['num_ent'])
            res_loop, _ = scatter_max(src=self.neigh_drop(neighs_loop), index=loop_index, dim=0, dim_size=data['num_ent'])


        res = self.tanh(self.bn(1 / 3 * self.ent_drop(res_in) + 1 / 3 * self.ent_drop(res_out) + + 1 / 3 * self.ent_drop(res_loop)))
        return res, rel_embed

    def compute_norm(self, edge_index, num_ent):
        row, col = edge_index
        edge_weight = torch.ones_like(row).float()
        deg = scatter_sum(src=edge_weight, index=row, dim=0,
                          dim_size=num_ent)  # Summing number of weights of the edges
        deg_inv = deg.pow(-0.5)  # D^{-0.5}
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col]  # D^{-0.5}
        return norm

