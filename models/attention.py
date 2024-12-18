import torch
import torch.nn as nn
import torch.nn.functional as F

class InterventionalAttention(nn.Module):
    """
    Interventional Attention:
    Extends standard multi-head attention by simulating interventions (e.g., token removal)
    and combining their effects with standard attention.
    """
    def __init__(self, d_model, n_heads, dropout=0.1, lambda_intervention=0.5):
        super(InterventionalAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        self.lambda_intervention = lambda_intervention

        # Projection layers for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)  # Output projection

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model), the input token representations.

        Returns:
            Tensor of shape (batch_size, seq_len, d_model), the enriched token representations.
        """
        B, L, D = x.size()

        # Step 1: Linear projections for Q, K, V
        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)  # (B, n_heads, L, d_k)
        K = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)  # (B, n_heads, L, d_k)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)  # (B, n_heads, L, d_k)

        # Step 2: Compute standard attention scores and probabilities
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # (B, n_heads, L, L)
        attn_probs = F.softmax(attn_scores, dim=-1)  # (B, n_heads, L, L)

        # Step 3: Simulate intervention (e.g., remove the first token)
        # Create a mask to "remove" the first token (simple example)
        intervention_mask = torch.ones_like(attn_probs)
        intervention_mask[:, :, :, 0] = 0  # Set first token weights to zero
        attn_probs_intervened = attn_probs * intervention_mask  # Apply the mask

        # Step 4: Normalize intervened probabilities
        attn_probs_intervened = attn_probs_intervened / attn_probs_intervened.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        # Step 5: Combine standard and intervened attention
        combined_probs = attn_probs + self.lambda_intervention * (attn_probs_intervened - attn_probs)

        # Step 6: Apply combined attention to values (V)
        attn_output = torch.matmul(combined_probs, V)  # (B, n_heads, L, d_k)

        # Step 7: Reshape and project back to original dimensions
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)  # (B, L, d_model)
        output = self.fc_out(self.dropout(attn_output))  # Final output projection
        return output
