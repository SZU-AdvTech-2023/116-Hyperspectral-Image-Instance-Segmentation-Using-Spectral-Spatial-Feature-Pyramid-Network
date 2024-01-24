from torch_geometric import nn as gnn
import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_mean
import os
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph
from torch_geometric.nn import DataParallel
from utils import printlog

import math
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        h,w,c = x.shape
        mask = torch.ones(size=(1,h,w))

        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).squeeze(0)

        output = torch.cat((x, pos), dim=2)
        return output

from torch_geometric.nn import SAGEConv
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F

class Graph_Classifier_Net(torch.nn.Module):
    def __init__(self,c_in, hidden_size, nc):
        super(Graph_Classifier_Net, self).__init__()
        self.bn_0 = gnn.BatchNorm(c_in)
        self.conv1 = SAGEConv(c_in, hidden_size)
        # 剪枝
        self.pool1 = TopKPooling(hidden_size, ratio=0.5)
        self.conv2 = SAGEConv(hidden_size, 128)
        self.pool2 = TopKPooling(128, ratio=0.5)
        self.conv3 = SAGEConv(128, 64)
        self.pool3 = TopKPooling(64, ratio=0.5)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(640, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(128, nc)
        )


    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
       
        x= self.bn_0(graph.x)

      
        x = x.squeeze(1)
      
        x = F.relu(self.conv1(x, edge_index))
  

        x, edge_index, _, batch, _, _= self.pool1(x, edge_index, None, batch)
 
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))

        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)

        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        # printlog(f"subGragh features shape after gcn3 :{x.shape}", print_signal)
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        # printlog(f"subGragh features  shape after TopKPooling 3 :{x.shape}", print_signal)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        # printlog(f"subGragh features x3 shape after cat[gmp, gap] :{x3.shape}", print_signal)
        # 将三层特征相加
        x = torch.cat([x1, x2, x3], dim=1)

        prob = self.classifier(x).squeeze(1)
        # printlog(f"class prob shape after classifier and sigmoid:{prob.shape}", print_signal)
        return torch.cat([x, prob], dim=1), prob

max_super_pixel = 100
from torch_geometric.nn import GCNConv
class Link_Net(torch.nn.Module):
    def __init__(self, hidden_size, num_class):
        super(Link_Net, self).__init__()

        self.num_class = num_class

        self.bn_0 = gnn.BatchNorm(hidden_size)
        self.gcn_1 = gnn.SGConv(hidden_size, hidden_size, K=3)
        self.bn_1 = gnn.BatchNorm(hidden_size)

        self.gcn_2 = gnn.SGConv(hidden_size, hidden_size//2, K=3)
        self.bn_2 = gnn.BatchNorm(hidden_size//2)
        self.bn_3= gnn.BatchNorm(1)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(hidden_size//2, hidden_size // 4),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(hidden_size // 4, 32)
        )

    def select_tree(self,graph,logits_class):
        # printlog(f"Input graph x shape:{graph.x.shape}, edge index shape: {graph.edge_index.shape}, link labels shape: {graph.link_labels.shape}",print_signal)
        tree_class = torch.argmax(logits_class, dim=-1)
        # printlog(f"Predict class shape: {tree_class.shape}",print_signal)
        # 获取树节点
        tree_index = torch.nonzero(tree_class).squeeze(1)
        # printlog(f"Tree index shape: {tree_index.shape}", print_signal)


#         # printlog(f"Graph only tree node shape: {graph.x.shape}", print_signal)
        edge_index, link_labels = subgraph(tree_index, graph.edge_index,graph.link_labels, relabel_nodes=True)
        # printlog(f"Subgraph edge_index shape:{edge_index.shape}, link labels shape : {link_labels.shape}",print_signal)
        sub_graph = Data(x=graph.x[tree_index], edge_index=edge_index, link_labels=link_labels)

        return sub_graph
    def encode(self, x,edge_index):

        x_normalization = self.bn_0(x)
        x = self.gcn_1(x_normalization,edge_index)
        h = self.bn_1(F.relu(x))
        h = self.bn_2(F.relu(self.gcn_2(h,edge_index)))

        # printlog(f" After BN x shape:{x.shape},edge_index max:{torch.max(edge_index)}", print_signal)

        return h



    def decode(self, z, edge_index):

        z = self.classifier(z)
        v_a = z[edge_index[0]]
        v_b = z[edge_index[1]]
        # 计算余弦相似度
        cosine_sim = F.cosine_similarity(v_a, v_b).unsqueeze(1)
        cosine_sim = self.bn_3(cosine_sim)

        prob_link = torch.sigmoid(cosine_sim)
        # printlog(f"Link Net decode output prob_link shape:{prob_link.shape}", print_signal)
        return prob_link

    def decode_all(self, z, edge_index):
        prob_adj = z[edge_index] @ z[edge_index].t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

    def forward(self,x,edge_index):

        # edge_index = graph.edge_index
        x = self.encode(x,edge_index)
        return self.decode(x,edge_index)



# Internal graph convolution
class SubGcn(nn.Module):
    def __init__(self, c_in, hidden_size, nc):
        super().__init__()
        self.gcn = gnn.SGConv(c_in, hidden_size, K=3)
        # 1
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(hidden_size // 2, nc)
        )

    def forward(self, graph):
        h = F.relu(self.gcn(graph.x, graph.edge_index))
        h_avg = gnn.global_mean_pool(h, graph.batch)
        logits = self.classifier(h)
        return logits


# Internal graph convolution feature module
class SubGcnFeature(nn.Module):
    def __init__(self, c_in, hidden_size):
        super().__init__()
        self.gcn = gnn.SGConv(c_in, hidden_size, K=3)

    def forward(self, graph):
        h = F.relu(self.gcn(graph.x, graph.edge_index))
        h_avg = gnn.global_mean_pool(h, graph.batch)
        return h_avg


# External graph convolution
class GraphNet(nn.Module):
    def __init__(self, c_in, hidden_size, nc):
        super().__init__()
        self.bn_0 = gnn.BatchNorm(c_in)
        self.gcn_1 = gnn.GCNConv(c_in, hidden_size)
        self.bn_1 = gnn.BatchNorm(hidden_size)
        self.gcn_2 = gnn.GraphConv(hidden_size, hidden_size)
        self.bn_2 = gnn.BatchNorm(hidden_size)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(hidden_size // 2, nc)
        )
        self.linknet = Link_Net(hidden_size, nc)
    def forward(self, graph):


        x_normalization = self.bn_0(graph.x)
        x = self.gcn_1(x_normalization, graph.edge_index)
        h = self.bn_1(F.relu(x))
        h = self.bn_2(F.relu(self.gcn_2(h, graph.edge_index)))

        logits = self.classifier(h + x_normalization)
        link_prob = self.linknet(h + x_normalization,graph.edge_index)
        # logits = self.classifier(h)
        return logits,link_prob

class GraphNetFeature(nn.Module):
    def __init__(self, c_in, hidden_size):
        super().__init__()
        self.bn_0 = gnn.BatchNorm(c_in)
        self.gcn_1 = gnn.GCNConv(c_in, hidden_size)
        self.bn_1 = gnn.BatchNorm(hidden_size)
        self.gcn_2 = gnn.GCNConv(hidden_size, hidden_size)
        self.bn_2 = gnn.BatchNorm(hidden_size)

    def forward(self, graph):
        x_normalization = self.bn_0(graph.x)
        # x_normalization = graph.x
        h = self.bn_1(F.relu(self.gcn_1(x_normalization, graph.edge_index)))
        h = self.bn_2(F.relu(self.gcn_2(h, graph.edge_index)))
        return x_normalization + h


