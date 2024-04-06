import torch.nn as nn
import torch.nn.functional as F

from .subgraph import SubGraph
from .globalgraph import SelfAttentionLayer

from utils import config

class TrajPredMLP(nn.Module):
    """Predict one feature trajectory, in offset format"""

    def __init__(self, in_channels, out_channels, hidden_unit):
        super(TrajPredMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.ReLU(),
            nn.Linear(hidden_unit, out_channels)
        )

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(
                    layer.weight, 
                    nonlinearity='relu'
                )

    def forward(self, x):
        return self.mlp(x)
    
class VectornetGNN(nn.Module):
    """
    hierarchical GNN with trajectory prediction MLP
    """

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        num_subgraph_layers=3, 
        num_global_graph_layer=1, 
        subgraph_width=64, 
        global_graph_width=64, 
        traj_pred_mlp_width=64
    ):
        super(VectornetGNN, self).__init__()
        self.polyline_vec_shape = in_channels * (2 ** num_subgraph_layers)
        
        # create polyline subgraph
        self.subgraph = SubGraph(
            in_channels, 
            num_subgraph_layers, 
            subgraph_width
        )
        
        # create global graph
        self.self_atten_layer = SelfAttentionLayer(
            self.polyline_vec_shape, 
            global_graph_width
        )
        
        # create prediction MLP
        self.traj_pred_mlp = TrajPredMLP(
            global_graph_width, 
            out_channels, 
            traj_pred_mlp_width
        )

    def forward(self, data):
        """
        args: 
            data (Data): [x, y, cluster, edge_index, valid_len, batch]
        """              
        subgraph_out = self.subgraph(data)
        
        self_atten_input = subgraph_out.x.view(
            -1, 
            config.NUM_GRAPH, 
            subgraph_out.num_features
        )

        self_atten_out = self.self_atten_layer(self_atten_input)
        
        agent_feature = self_atten_out[:, 0:config.NUM_AGENTS+1, :]
        
        mlp_input = agent_feature.contiguous().view(
            -1, 
            agent_feature.shape[2]
        )
        
        prediction = self.traj_pred_mlp(mlp_input)    
        
        return prediction