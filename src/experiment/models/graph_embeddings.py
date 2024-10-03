from torch_geometric.nn import TopKPooling, SAGPooling
from torch_geometric.nn import GATv2Conv, aggr, MessagePassing, GCNConv
import torch
from torch_geometric.data import Data
from torch_geometric.utils import unbatch
from options import Options


class GraphEmbeddings(torch.nn.Module):
    def __init__(self, options: Options):
        super(GraphEmbeddings, self).__init__()
        embedding_size = options.embedding_size
        num_node_features = options.num_node_features
        edge_dim = options.num_edge_features
        self.emb_node_features = torch.nn.Embedding(options.num_shapes, num_node_features)

        self.emb1 = GATv2Conv(num_node_features, embedding_size//2, edge_dim=edge_dim, heads=1, dropout=0, bias=True)
        self.bn1 = torch.nn.BatchNorm1d(embedding_size//2)
        self.rel1 = torch.nn.LeakyReLU()

        self.emb2 = GATv2Conv(embedding_size//2, embedding_size, edge_dim=edge_dim, heads=1, dropout=0, bias=True)
        self.bn2 = torch.nn.BatchNorm1d(embedding_size)
        self.rel2 = torch.nn.LeakyReLU()

        self.lin1 = torch.nn.Linear(num_node_features, embedding_size)
        self.lin2 = torch.nn.Linear(4, embedding_size//2)

        self.gate_nn = torch.nn.Linear(embedding_size, 1)
        self.global_pool = aggr.AttentionalAggregation(self.gate_nn)

    def forward(self, d: Data, return_node_embs: bool = False):
        node_features, edge_index, edge_attr, batch = d.x, d.edge_index, d.edge_attr, d.batch
        node_features = self.emb_node_features(node_features.argmax(dim=1)).squeeze()


        d1 = self.emb1(node_features, edge_index, edge_attr)
        d1 = self.bn1(d1)
        d1 = self.rel1(d1)

        # edge_attr_transformed = self.lin2(edge_attr.float().squeeze())
        # d1[batch] = d1[batch] + edge_attr_transformed[batch]

        d2 = self.emb2(d1, edge_index, edge_attr)
        d2 = self.bn2(d2)
        d2 = self.rel2(d2)

        global_emb = self.global_pool(d2, batch)
        if return_node_embs:
            return global_emb, unbatch(d2, batch)
        else:
            return global_emb


class EGNNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(EGNNConv, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.lin2 = torch.nn.Linear(4, in_channels)
        self.rel1 = torch.nn.LeakyReLU()

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        edge_attr = self.lin2(edge_attr)
        res = self.lin(x_j + edge_attr)
        res = self.rel1(res)
        return res


class GraphMessageEmbeddings(torch.nn.Module):
    def __init__(self, options: Options):
        super(GraphMessageEmbeddings, self).__init__()
        embedding_size = options.embedding_size
        num_node_features = options.num_node_features
        self.dropout = torch.nn.Dropout(0.2)

        self.pool1 = TopKPooling(embedding_size, ratio=0.8)
        self.pool1 = SAGPooling(embedding_size)

        self.emb_node_features = torch.nn.Embedding(51, num_node_features)
        self.conv1 = EGNNConv(num_node_features, embedding_size)

        self.bn1 = torch.nn.BatchNorm1d(embedding_size)
        self.rel1 = torch.nn.LeakyReLU()

        self.conv2 = EGNNConv(embedding_size, embedding_size)
        self.bn2 = torch.nn.BatchNorm1d(embedding_size)
        self.rel2 = torch.nn.LeakyReLU()

        self.gate_nn = torch.nn.Linear(embedding_size, 1)
        self.global_pool = aggr.AttentionalAggregation(self.gate_nn)

    def forward(self, d, return_node_embs: bool = False):
        node_features, edge_index, edge_attr, batch = d.x, d.edge_index, d.edge_attr, d.batch
        node_features = self.emb_node_features(node_features).squeeze()

        # x = torch.cat([node_features, edge_attr[edge_index[0]]], dim=1)
        x = self.conv1(node_features, edge_index, edge_attr)
        # x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        x = self.bn1(x)
        x = self.rel1(x)
        # x = self.dropout(x)

        x2 = self.conv2(x, edge_index, edge_attr)
        x2 = self.bn2(x2)
        x2 = self.rel2(x2)

        global_emb = self.global_pool(x2, batch)
        if return_node_embs:
            return global_emb, unbatch(x2, batch)
        else:
            return global_emb


class GraphPositionNodeEmbeddings(torch.nn.Module):
    def __init__(self, options: Options):
        super().__init__()
        embedding_size = options.embedding_size
        num_node_features = options.num_node_features
        self.emb_node_features = torch.nn.Linear(
            options.num_shapes + options.num_edge_features,
            num_node_features
        )

        self.emb1 = GCNConv(num_node_features, embedding_size//2)
        self.bn1 = torch.nn.BatchNorm1d(embedding_size//2)
        self.rel1 = torch.nn.LeakyReLU()

        self.emb2 = GCNConv(embedding_size//2, embedding_size)
        self.bn2 = torch.nn.BatchNorm1d(embedding_size)
        self.rel2 = torch.nn.LeakyReLU()

        self.gate_nn = torch.nn.Linear(embedding_size, 1)
        self.global_pool = aggr.AttentionalAggregation(self.gate_nn)

    def forward(self, d: Data, return_node_embs: bool = False):
        node_features, edge_index, batch = d.x, d.edge_index, d.batch
        node_features = self.emb_node_features(node_features).squeeze()

        d1 = self.emb1(node_features, edge_index)
        d1 = self.bn1(d1)
        d1 = self.rel1(d1)

        d2 = self.emb2(d1, edge_index)
        d2 = self.bn2(d2)
        d2 = self.rel2(d2)

        global_emb = self.global_pool(d2, batch)
        if return_node_embs:
            return global_emb, unbatch(d2, batch)
        else:
            return global_emb
