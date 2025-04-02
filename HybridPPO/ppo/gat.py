"""
Graph Attention Network (GAT) for Job Shop Scheduling

Standalone implementation that doesn't rely on pretrained weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GraphAttentionNetwork(nn.Module):
    """
    Graph Attention Network (GAT) for processing job shop scheduling graphs.
    
    This implementation is self-contained and does not rely on loading
    pretrained weights from external sources. It can adapt to different
    input feature sizes.
    """
    
    def __init__(self, in_features=15, hidden_dim=64, out_features=128, num_heads=3, 
                 dropout=0.15, concat=True, second_layer_concat=False, residual=False,
                 force_pretrained_dim=False):
        """
        Initialize the GAT model.
        
        Args:
            in_features: Dimension of input node features (default 15 for compatibility with pretrained models)
            hidden_dim: Dimension of hidden layers
            out_features: Dimension of output embeddings
            num_heads: Number of attention heads
            dropout: Dropout rate
            concat: Whether to concatenate attention heads in first layer
            second_layer_concat: Whether to concatenate attention heads in second layer
            residual: Whether to use residual connections
            force_pretrained_dim: Whether to force using pretrained dimensions regardless of input size
        """
        super(GraphAttentionNetwork, self).__init__()
        
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout = dropout
        self.concat = concat
        self.second_layer_concat = second_layer_concat
        self.residual = residual
        self.force_pretrained_dim = force_pretrained_dim
        
        # Feature projectors for different input sizes (initialized on demand)
        self.feature_projectors = {}
        
        # If forcing pretrained dimensions, initialize the layers immediately
        if force_pretrained_dim:
            self._initialize_fixed_layers()
        else:
            # Otherwise, we'll initialize the GATv2Conv layers in the forward method
            # to handle dynamic input feature sizes
            self.embedding1 = None
            self.embedding2 = None
            self.initialized = False
            
        # Calculate output size for reference (actual size might differ based on actual input)
        if second_layer_concat:
            self.out_size = in_features + out_features * num_heads
        else:
            self.out_size = in_features + out_features
    
    def _get_feature_projector(self, input_dim, target_dim=15, device=None):
        """
        Get or create a feature projector for the given input dimension.
        
        Args:
            input_dim: Input feature dimension
            target_dim: Target feature dimension (default 15 for pretrained model)
            device: Device to place the new projector on
            
        Returns:
            A feature projector module
        """
        if input_dim == target_dim:
            return nn.Identity()
            
        key = f"{input_dim}_to_{target_dim}"
        if key not in self.feature_projectors:
            # Create a new projector
            projector = nn.Sequential(
                nn.Linear(input_dim, target_dim),
                nn.ReLU()
            )
            if device is not None:
                projector = projector.to(device)
            self.feature_projectors[key] = projector
            # Simplified message
            print(f"Created projector: {input_dim} -> {target_dim}")
        
        return self.feature_projectors[key]
    
    def _initialize_fixed_layers(self):
        """
        Initialize the layers with fixed pretrained dimensions
        """
        # First GAT layer with fixed input size of 15 features (for pretrained compatibility)
        self.embedding1 = GATv2Conv(
            in_channels=15,  # Fixed at 15 for pretrained models
            out_channels=self.hidden_dim,
            dropout=self.dropout,
            heads=self.num_heads,
            concat=self.concat,
            add_self_loops=False,
            negative_slope=0.15
        )
        
        # Second GAT layer with fixed dimensions for pretrained models
        second_layer_input = self.hidden_dim * self.num_heads + 15 if self.concat else self.hidden_dim + 15
        
        # Use 207 input dimension for second layer as in pretrained model
        if self.concat and second_layer_input != 207:
            print(f"Warning: Expected second layer input 207, got {second_layer_input}. Using 207 for compatibility.")
            second_layer_input = 207
        
        # Custom GATv2Conv that exactly matches the pretrained model structure
        # The key is to preserve all of the original GATv2Conv behavior but have bias of size 128
        # instead of the default 384 (3 heads * 128 output dim)
        self.embedding2 = GATv2Conv(
            in_channels=second_layer_input,
            out_channels=self.out_features,
            dropout=self.dropout,
            heads=self.num_heads,
            concat=False,  # Important: set this to False to get a single output tensor of size 128
            add_self_loops=False,
            negative_slope=0.15
        )
        
        # Update output size based on the actual pretrained model structure
        self.out_size = 15 + 128  # Observed from the expected GNN output
        
        self.initialized = True
        print(f"Initialized fixed-dimension GAT for pretrained compatibility")
    
    def _initialize_layers(self, actual_in_features):
        """
        Initialize the GATv2Conv layers based on the actual input features.
        
        Args:
            actual_in_features: The actual number of input features detected
        """
        # First GAT layer
        self.embedding1 = GATv2Conv(
            in_channels=actual_in_features,
            out_channels=self.hidden_dim,
            dropout=self.dropout,
            heads=self.num_heads,
            concat=self.concat,
            add_self_loops=False,
            negative_slope=0.15
        )
        
        # Second GAT layer
        # Calculate input dimension based on whether heads are concatenated
        second_layer_input = self.hidden_dim * self.num_heads + actual_in_features if self.concat else self.hidden_dim + actual_in_features
        
        self.embedding2 = GATv2Conv(
            in_channels=second_layer_input,
            out_channels=self.out_features,
            dropout=self.dropout,
            heads=self.num_heads,
            concat=self.second_layer_concat,
            add_self_loops=False,
            negative_slope=0.15
        )
        
        # Update output size based on actual input
        if self.second_layer_concat:
            self.out_size = actual_in_features + self.out_features * self.num_heads
        else:
            self.out_size = actual_in_features + self.out_features
        
        self.initialized = True
        print(f"Initialized GAT with actual input size: {actual_in_features}")
        print(f"Output size: {self.out_size}")
    
    def forward(self, node_features, edge_index):
        """
        Forward pass through the GAT network
        
        Args:
            node_features: Node features of shape [num_nodes, actual_features]
            edge_index: Graph connectivity of shape [2, num_edges]
            
        Returns:
            Node embeddings of shape [num_nodes, out_features]
        """
        # Get the actual input feature dimension
        actual_features = node_features.size(1)
        
        # Handle the case where we're forcing pretrained dimensions
        if self.force_pretrained_dim:
            # Use a learnable feature projector instead of simple padding/truncation
            if actual_features != 15:
                if not hasattr(self, '_dim_warning_shown'):
                    # Simplified message
                    print(f"Using feature projection: {actual_features} -> 15")
                    self._dim_warning_shown = True
                
                # Get or create a feature projector for this dimension
                projector = self._get_feature_projector(
                    actual_features, 
                    15, 
                    device=node_features.device
                )
                
                # Project the features to 15 dimensions
                node_features = projector(node_features)
        else:
            # Initialize the network with the actual input size if not already
            if not self.initialized or self.embedding1 is None:
                self._initialize_layers(actual_features)
            # If the network is initialized but with a different input size, reinitialize
            elif self.embedding1.lin_l.weight.size(1) != actual_features:
                print(f"Input feature size changed from {self.embedding1.lin_l.weight.size(1)} to {actual_features}. Reinitializing network.")
                self._initialize_layers(actual_features)
        
        # First GAT layer
        h1 = self.embedding1(node_features, edge_index)
        if self.residual and node_features.size(-1) == h1.size(-1):
            h1 = h1 + node_features
        h1 = F.relu(h1)
        
        # If concat is enabled, concatenate with original features
        if self.concat:
            h = torch.cat([node_features, h1], dim=-1)
            
            # Handle specific case for pretrained models
            if self.force_pretrained_dim and h.size(1) != 207 and not hasattr(self, '_second_layer_warning_shown'):
                # Simplified message
                print(f"Adjusting second layer input dimension to match pretrained model")
                self._second_layer_warning_shown = True
                
                # Pad or truncate to match 207 features
                if h.size(1) < 207:
                    padding = torch.zeros(h.size(0), 207 - h.size(1), device=h.device)
                    h = torch.cat([h, padding], dim=1)
                else:
                    h = h[:, :207]
        else:
            h = h1
        
        # Second GAT layer
        h2 = self.embedding2(h, edge_index)
        if self.residual and h.size(-1) == h2.size(-1):
            h2 = h2 + h
        h2 = F.relu(h2)
        
        # If second layer concat is enabled, concatenate original features with the embeddings
        if self.second_layer_concat:
            result = torch.cat([node_features, h2], dim=-1)
        else:
            result = h2
            
        return result 