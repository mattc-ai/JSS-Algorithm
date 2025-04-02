"""
Multi-Head Attention for Job Shop Scheduling

Standalone implementation that doesn't rely on pretrained weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module for processing job shop scheduling data.
    
    This implementation is self-contained and does not rely on loading
    pretrained weights from external sources.
    """
    
    def __init__(self, encoder_size, context_size, hidden_size=64, 
                 num_heads=3, dropout=0.15):
        """
        Initialize the multi-head attention module.
        
        Args:
            encoder_size: Size of the encoded features
            context_size: Size of the context features
            hidden_size: Size of the hidden layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(MultiHeadAttention, self).__init__()
        
        self.encoder_size = encoder_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Memory network
        self.linear1 = nn.Linear(context_size, hidden_size * num_heads)
        self.linear2 = nn.Linear(hidden_size * num_heads, hidden_size)
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size * num_heads,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Classification network
        self.act = nn.LeakyReLU(0.15)
        self.linear3 = nn.Linear(encoder_size + hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)
    
    def forward(self, encoded_features, state_features, mask=None):
        """
        Forward pass through the multi-head attention module.
        
        Args:
            encoded_features: Encoded features of shape [batch_size, num_nodes, encoder_size]
            state_features: State features of shape [batch_size, num_jobs, context_size]
            mask: Optional mask for attention of shape [batch_size, num_jobs]
            
        Returns:
            attention_scores: shape [batch_size, num_jobs]
            classification_output: shape [batch_size, num_jobs, hidden_size]
        """
        # Process state through memory network
        x1 = self.linear1(state_features)
        
        # Apply self-attention
        x2 = x1 + self.self_attn(x1, x1, x1)[0]
        
        # Generate memory context
        x2 = F.relu(self.linear2(x2))
        
        # Concatenate with encoder features
        combined = torch.cat([encoded_features, x2], dim=-1)
        
        # Generate classification output
        xx = self.act(self.linear3(combined))
        attention_scores = self.linear4(xx).squeeze(-1)
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        return attention_scores, xx


class GlobalAttention(nn.Module):
    """
    Global attention module that captures dependencies between nodes.
    
    This module uses a multi-head self-attention mechanism to model
    the relationships between different nodes in the scheduling problem.
    """
    
    def __init__(self, hidden_dim, num_heads=4, dropout=0.15):
        """
        Initialize the global attention module.
        
        Args:
            hidden_dim: Dimension of hidden layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(GlobalAttention, self).__init__()
        
        # Ensure hidden_dim is divisible by num_heads
        self.adjusted_dim = (hidden_dim // num_heads) * num_heads
        if self.adjusted_dim != hidden_dim:
            print(f"Warning: hidden_dim {hidden_dim} is not divisible by num_heads {num_heads}. "
                  f"Adjusting to {self.adjusted_dim}.")
        
        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.adjusted_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # If we had to adjust the dimension, add a projection layer
        if self.adjusted_dim != hidden_dim:
            self.input_projection = nn.Linear(hidden_dim, self.adjusted_dim)
            self.output_projection = nn.Linear(self.adjusted_dim, hidden_dim)
        else:
            self.input_projection = nn.Identity()
            self.output_projection = nn.Identity()
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        """
        Forward pass through the global attention module.
        
        Args:
            x: Input tensor of shape [batch_size, num_nodes, hidden_dim]
            mask: Optional attention mask of shape [batch_size, num_nodes]
            
        Returns:
            Output tensor of shape [batch_size, num_nodes, hidden_dim]
        """
        # Apply input projection if needed
        x_proj = self.input_projection(x)
        
        # Create an attention mask from the input mask if provided
        attn_mask = None
        if mask is not None:
            # Convert boolean mask to attention mask
            attn_mask = ~mask.bool()  # Invert if mask has 1s for valid positions
        
        # Self-attention block
        attn_output, _ = self.self_attention(
            query=x_proj,
            key=x_proj,
            value=x_proj,
            key_padding_mask=attn_mask
        )
        
        # Apply output projection if needed
        attn_output = self.output_projection(attn_output)
        
        # Residual connection and layer normalization
        x = self.layer_norm1(x + attn_output)
        
        # Feed-forward block
        ff_output = self.feed_forward(x)
        
        # Residual connection and layer normalization
        x = self.layer_norm2(x + ff_output)
        
        return x 