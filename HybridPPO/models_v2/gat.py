"""
Graph Attention Network (GAT) for Job Shop Scheduling

Simplified implementation for self-labeling pretraining.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GraphAttentionNetwork(nn.Module):
    """
    Graph Attention Network (GAT) for processing job shop scheduling graphs.
    
    This simplified implementation is designed to be compatible with
    the self-labeling pretraining code while using the GATv2Conv backend.
    """
    
    def __init__(self, in_features=15, hidden_dim=64, out_features=128, num_heads=3, 
                 dropout=0.15):
        """
        Initialize the GAT model.
        
        Args:
            in_features: Dimension of input node features
            hidden_dim: Dimension of hidden layers
            out_features: Dimension of output embeddings
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(GraphAttentionNetwork, self).__init__()
        
        # For compatibility with self_labeling_pretrain_standalone.py
        # Initialize the GAT layers immediately with fixed structure
        self.embedding1 = GATv2Conv(
            in_channels=in_features,
            out_channels=hidden_dim,
            dropout=dropout,
            heads=num_heads,
            concat=True,
            add_self_loops=False,
            negative_slope=0.15
        )
        
        # Second GAT layer
        second_layer_input = hidden_dim * num_heads + in_features
        self.embedding2 = GATv2Conv(
            in_channels=second_layer_input,
            out_channels=out_features,
            dropout=dropout,
            heads=num_heads,
            concat=False,
            add_self_loops=False,
            negative_slope=0.15
        )
        
        # Output size needed by self_labeling_pretrain_standalone.py
        self.out_size = in_features + out_features
    
    def forward(self, node_features, edge_index):
        """
        Forward pass through the GAT network
        
        Args:
            node_features: Node features of shape [num_nodes, in_features]
            edge_index: Graph connectivity of shape [2, num_edges]
            
        Returns:
            Node embeddings of shape [num_nodes, out_features]
        """
        # First GAT layer
        h1 = self.embedding1(node_features, edge_index)
        h1 = F.relu(h1)
        
        # Concatenate with original features
        h = torch.cat([node_features, h1], dim=-1)
        
        # Second GAT layer
        h2 = self.embedding2(h, edge_index)
        h2 = F.relu(h2)
        
        # Concatenate original features with embeddings for final output
        result = torch.cat([node_features, h2], dim=-1)
        
        return result 