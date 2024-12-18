import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalGraphEncoder(nn.Module):
    """
    因果関係を学習するためのエンコーダ。
    ノード（トークン）間の潜在的な因果グラフを推定する。
    """
    def __init__(self, d_model, dropout=0.1):
        super(CausalGraphEncoder, self).__init__()
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        causal_matrix = torch.sigmoid(self.fc(x))  # 疑似的な因果スコアを生成
        causal_graph = torch.matmul(causal_matrix, x)  # 因果関係を考慮した特徴変換
        return self.dropout(causal_graph)

class InterventionalAttention(nn.Module):
    """
    通常のAttention機構に介入操作を加えるモジュール。
    特定のトークンが存在しない場合（介入）をシミュレートする。
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(InterventionalAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.size()
        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        # 通常のAttentionスコア
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)

        # 介入シミュレーション（トークン除去）
        intervened_attn_probs = attn_probs.clone()
        intervened_attn_probs[:, :, :, 0] = 0  # 先頭トークンを介入として無視

        # 通常と介入の統合
        combined_probs = attn_probs + 0.5 * (intervened_attn_probs - attn_probs)

        out = torch.matmul(combined_probs, V)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.fc_out(self.dropout(out))

class CausalCompositionalityLayer(nn.Module):
    """
    複数の因果関係を統合し、より深い因果連鎖を表現するレイヤー。
    """
    def __init__(self, d_model, dropout=0.1):
        super(CausalCompositionalityLayer, self).__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout(x)
        return self.layer_norm(x + residual)

class CausaFormer(nn.Module):
    """
    CausaFormer本体: 因果構造学習、介入的Attention、因果連鎖統合を含むTransformer拡張モデル。
    """
    def __init__(self, input_dim=512, num_layers=6, n_heads=8, dropout=0.1):
        super(CausaFormer, self).__init__()
        self.embedding = nn.Linear(input_dim, input_dim)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "causal_graph": CausalGraphEncoder(input_dim, dropout),
                "interventional_attention": InterventionalAttention(input_dim, n_heads, dropout),
                "causal_composition": CausalCompositionalityLayer(input_dim, dropout)
            })
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        x = self.embedding(x)
        for layer in self.layers:
            x = layer["causal_graph"](x)
            x = layer["interventional_attention"](x)
            x = layer["causal_composition"](x)
        return self.output_layer(x)
