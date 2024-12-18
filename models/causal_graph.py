import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalGraphEncoder(nn.Module):
    """
    Causal Graph Encoder:
    This module learns latent causal relationships among input tokens
    and encodes this information for subsequent processing.
    """
    def __init__(self, d_model, dropout=0.1):
        super(CausalGraphEncoder, self).__init__()
        self.node_proj = nn.Linear(d_model, d_model)  # Project each token into latent space
        self.edge_proj = nn.Linear(d_model, d_model)  # Learn relationships (edges) between tokens
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for CausalGraphEncoder.
        
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model), the input token representations.

        Returns:
            Tensor of shape (batch_size, seq_len, d_model), enriched with causal structure information.
        """
        # Step 1: Node-level representation
        # Each token representation is transformed into a "causal node" in latent space.
        node_features = self.activation(self.node_proj(x))  # (batch_size, seq_len, d_model)

        # Step 2: Compute pairwise relationships (edges)
        # Each token interacts with every other token to compute relationships.
        # Pairwise attention-like mechanism for edge information.
        edge_features = torch.matmul(node_features, node_features.transpose(-2, -1))  # (batch_size, seq_len, seq_len)
        edge_features = torch.softmax(edge_features, dim=-1)  # Normalize relationships (as attention)

        # Step 3: Enrich token features using causal relationships
        # Aggregate information from related tokens using edge weights.
        enriched_features = torch.matmul(edge_features, node_features)  # (batch_size, seq_len, d_model)

        # Step 4: Apply dropout and return enriched features
        enriched_features = self.dropout(enriched_features)
        return enriched_features
