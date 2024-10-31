from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, BatchNorm1d
from torch_geometric.nn import GCNConv
#from torch_geometric.nn import SGConv, SAGEConv, GCNConv, GATConv, TransformerConv
#from torch_geometric.nn.models import LabelPropagation

#from gcn_mf import GCNmfConv
#from pa_gnn import PaGNNConv

class GNN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim, num_layers=2, dropout=0, conv_type="GCN"):
        super(GNN, self).__init__()
        self.convs = ModuleList([get_conv(num_features, hidden_dim)])
        for i in range(num_layers - 2):
            self.convs.append(get_conv(hidden_dim, hidden_dim))
        self.convs.append(get_conv(hidden_dim, num_classes)) 
        self.num_layers = num_layers
        self.dropout = dropout
        
    def forward(self, x, edge_index=None, adjs=None, full_batch=True):
        return self.forward_full_batch(x, edge_index) if full_batch else self.forward_sampled(x, adjs)

    def forward_full_batch(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index).relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.convs[-1](x, edge_index)

        return torch.nn.functional.log_softmax(out, dim=1)

    def forward_sampled(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x.log_softmax(dim=1)

    def inference(self, x_all, inference_loader, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in inference_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        return x_all

def get_conv(input_dim, output_dim):
    return GCNConv(input_dim, output_dim)
 

