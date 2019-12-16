"""
This module implements the PyTorch modules that define the sparse
message-passing graph neural networks for segment classification.
In particular, this implementation utilizes the pytorch_geometric
and supporting libraries:
https://github.com/rusty1s/pytorch_geometric
"""

# Externals
import torch
import torch.nn as nn
from torch_scatter import scatter_add
# from torch_geometric.nn import global_add_pool

# Locals
from .utils import make_mlp

class EdgeNetwork(nn.Module):
    """
    A module which computes weights for edges of the graph.
    For each edge, it selects the associated nodes' features
    and applies some fully-connected network layers with a final
    sigmoid activation.
    """
    def __init__(self, input_dim, hidden_dim=8, hidden_activation=nn.Tanh,
                 layer_norm=True):
        super(EdgeNetwork, self).__init__()
        self.network = make_mlp(input_dim*2,
                                [hidden_dim, hidden_dim, hidden_dim, 1],
                                hidden_activation=hidden_activation,
                                output_activation=None,
                                layer_norm=layer_norm)

    def forward(self, x, edge_index):
        # Select the features of the associated nodes
        start, end = edge_index
        x1, x2 = x[start], x[end]
        edge_inputs = torch.cat([x[start], x[end]], dim=1)
        return self.network(edge_inputs).squeeze(-1)

class NodeNetwork(nn.Module):
    """
    A module which computes new node features on the graph.
    For each node, it aggregates the neighbor node features
    (separately on the input and output side), and combines
    them with the node's previous features in a fully-connected
    network to compute the new features.
    """
    def __init__(self, input_dim, output_dim, hidden_activation=nn.Tanh,
                 layer_norm=True):
        super(NodeNetwork, self).__init__()
        self.network = make_mlp(input_dim*3, [output_dim]*4,
                                hidden_activation=hidden_activation,
                                output_activation=hidden_activation,
                                layer_norm=layer_norm)

    def forward(self, x, e, edge_index):
        start, end = edge_index
        # Aggregate edge-weighted incoming/outgoing features
        mi = scatter_add(e[:, None] * x[start], end, dim=0, dim_size=x.shape[0])
        mo = scatter_add(e[:, None] * x[end], start, dim=0, dim_size=x.shape[0])
        node_inputs = torch.cat([mi, mo, x], dim=1)
        return self.network(node_inputs)

class OutputNetwork(nn.Module):
    """
    Module that aggregates node and edge features and outputs an n_tracks one_hot vector
    """
    def __init__(self, input_dim, max_tracks, hidden_dim=8, hidden_activation=nn.Tanh,
                 layer_norm=True):
        super(OutputNetwork, self).__init__()
        self.network = make_mlp(input_dim,
                                [hidden_dim, hidden_dim, hidden_dim, max_tracks + 1],
                                hidden_activation=hidden_activation,
                                output_activation=None,
                                layer_norm=False)

    def forward(self, x):
#         print("Features: ", x)
        sum_input = x.sum(dim=0)
#         print("Means: ", av_input)
        return self.network(sum_input)
    
    
class GNNTrackCounter(nn.Module):
    """
    Segment classification graph neural network model.
    Consists of an input network, an edge network, and a node network.
    """
    def __init__(self, input_dim=3, hidden_dim=8, n_graph_iters=3,
                 hidden_activation=nn.Tanh, layer_norm=True, max_tracks=40):
        super(GNNTrackCounter, self).__init__()
        self.n_graph_iters = n_graph_iters
        # Setup the input network
        self.input_network = make_mlp(input_dim, [hidden_dim],
                                      output_activation=hidden_activation,
                                      layer_norm=layer_norm)
        # Setup the edge network
        self.edge_network = EdgeNetwork(input_dim+hidden_dim, hidden_dim,
                                        hidden_activation, layer_norm=layer_norm)
        # Setup the node layers
        self.node_network = NodeNetwork(input_dim+hidden_dim, hidden_dim,
                                            hidden_activation, layer_norm=layer_norm)
        # An output summation
        self.output_network = OutputNetwork(input_dim+hidden_dim, max_tracks, hidden_dim, 
                                            hidden_activation, layer_norm=layer_norm)

    def forward(self, inputs):
        """Apply forward pass of the model"""
        # Apply input network to get hidden representation
        x = self.input_network(inputs.x)
#         print(x.shape)
        # Shortcut connect the inputs onto the hidden representation
        x = torch.cat([x, inputs.x], dim=-1)
#         print(x.shape)
        # Loop over iterations of edge and node networks
        for i in range(self.n_graph_iters):
            # Apply edge network
            e = torch.sigmoid(self.edge_network(x, inputs.edge_index))
            # Apply node network
            x = self.node_network(x, e, inputs.edge_index)
#             print(x.shape)
            # Shortcut connect the inputs onto the hidden representation
            x = torch.cat([x, inputs.x], dim=-1)
        # Apply output network
        o = self.output_network(x)
#         print(o)
#         print(torch.sigmoid(o))
        return o
#         return self.edge_network(x, inputs.edge_index)