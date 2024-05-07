import torch
import torch.nn.functional as F 
import torch.nn as nn
import numpy as np
    
class SelfAttentionLayer(nn.Module):
    def __init__(self, in_channels, global_graph_width=64):
        super().__init__()
        self.queryFC = nn.Linear(in_channels, global_graph_width)
        self.keyFC   = nn.Linear(in_channels, global_graph_width)
        self.valueFC = nn.Linear(in_channels, global_graph_width)
        
        nn.init.kaiming_normal_(self.queryFC.weight)
        nn.init.kaiming_normal_(self.keyFC.weight)
        nn.init.kaiming_normal_(self.valueFC.weight)

    def forward(self, x):
        # Assuming x is of shape (B, N, D), where B is the batch size,
        # N is the number of items in the sequence, and D is the features dimension.
        p_query = F.relu(self.queryFC(x))   # (B, N, D_k)
        p_key   = F.relu(self.keyFC(x))     # (B, N, D_k)
        p_value = F.relu(self.valueFC(x))   # (B, N, D_v)

        # Perform batch matrix multiplication between p_query and the transpose of p_key
        # The shapes are (B, N, D_k) and (B, D_k, N) -> (B, N, N)
        query_result = torch.bmm(p_query, p_key.transpose(1, 2))

        # Scale query_result to avoid gradients vanishing with high dimensions
        query_result = query_result / (p_key.size(-1) ** 0.5)

        # Apply softmax to the last dimension to get the attention weights
        attention = F.softmax(query_result, dim=-1)

        # Now we perform batch matrix multiplication between attention weights and p_value
        # The shapes are (B, N, N) and (B, N, D_v) -> (B, N, D_v)
        output = torch.bmm(attention, p_value)

        # Add the residual connection
        return output + p_query