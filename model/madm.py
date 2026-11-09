"""
MADM: Multivariate Association Discrepancy Mining
for Multivariate Time Series Anomaly Detection

Architecture:
  - Temporal Branch: inherited from Anomaly Transformer (Xu et al., ICLR 2022)
    uses anomaly attention with a learnable Gaussian prior to compute temporal association discrepancy
  - Spatial Branch: proposed contribution
    HypergraphGAT + Hypergraph Attention Recalibration (HAR) + spatial association discrepancy
  - Gated Fusion: learns adaptive fusion weights for the two branch features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt


# ============================================================
#  Utilities
# ============================================================
def scatter_softmax(src, index, dim_size=None):
    """Numerically stable per-source-node softmax."""
    if dim_size is None:
        dim_size = int(index.max().item()) + 1
    max_vals = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    max_vals.scatter_reduce_(0, index.unsqueeze(-1).expand_as(src), src,
                             reduce='amax', include_self=False)
    src_shifted = src - max_vals[index]
    exp_src = torch.exp(src_shifted)
    sum_vals = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    sum_vals.scatter_add_(0, index.unsqueeze(-1).expand_as(exp_src), exp_src)
    return exp_src / (sum_vals[index] + 1e-8)


# ============================================================
#  1. Embedding
# ============================================================
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        self.tokenConv = nn.Conv1d(c_in, d_model, kernel_size=3, padding=1,
                                   padding_mode='circular', bias=False)
        nn.init.kaiming_normal_(self.tokenConv.weight, mode='fan_in',
                                nonlinearity='leaky_relu')

    def forward(self, x):
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(np.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.0):
        super().__init__()
        self.value_embedding    = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(self.value_embedding(x) + self.position_embedding(x))


# ============================================================
#  2. Temporal Branch — inherited from Anomaly Transformer (ICLR 2022)
# ============================================================
class TriangularCausalMask:
    def __init__(self, B, L, device='cpu'):
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones([B, 1, L, L], dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self):
        return self._mask


class AnomalyAttention(nn.Module):
    """
    Self-attention with a learnable Gaussian prior.
    Outputs:
      series — actual attention distribution S_T
      prior  — Gaussian kernel prior P_T(sigma), sigma is learnable
    """
    def __init__(self, win_size, mask_flag=True, scale=None,
                 attention_dropout=0.0, output_attention=False):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        distances = torch.zeros(win_size, win_size)
        for i in range(win_size):
            for j in range(win_size):
                distances[i][j] = abs(i - j)
        self.register_buffer('distances', distances)

    def forward(self, queries, keys, values, sigma, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum('blhe,bshe->bhls', queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores

        sigma = sigma.transpose(1, 2)
        win_size = attn.shape[-1]
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, win_size)

        d = self.distances.unsqueeze(0).unsqueeze(0).repeat(
            sigma.shape[0], sigma.shape[1], 1, 1).to(sigma.device)
        prior = (1.0 / (torch.sqrt(2 * torch.tensor(np.pi, device=sigma.device))
                 * sigma) * torch.exp(-d ** 2 / 2 / (sigma ** 2)))

        series = self.dropout(torch.softmax(attn, dim=-1))
        V = torch.einsum('bhls,bshd->blhd', series, values)

        if self.output_attention:
            return V.contiguous(), series, prior, sigma
        return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super().__init__()
        d_k = d_model // n_heads
        self.inner_attention   = attention
        self.query_projection  = nn.Linear(d_model, d_k * n_heads)
        self.key_projection    = nn.Linear(d_model, d_k * n_heads)
        self.value_projection  = nn.Linear(d_model, d_k * n_heads)
        self.sigma_projection  = nn.Linear(d_model, n_heads)
        self.out_projection    = nn.Linear(d_k * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys    = self.key_projection(keys).view(B, L, H, -1)
        values  = self.value_projection(values).view(B, L, H, -1)
        sigma   = self.sigma_projection(queries.view(B, L, -1)).view(B, L, H)
        out, series, prior, sigma = self.inner_attention(
            queries, keys, values, sigma, attn_mask)
        return self.out_projection(out.view(B, L, -1)), series, prior, sigma


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        new_x, series, prior, sigma = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(F.gelu(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), series, prior, sigma


class TemporalBranch(nn.Module):
    """
    Temporal Branch: stacks L_e anomaly attention encoder layers.
    Returns:
      z_time      [B, L, D]     temporal features
      series_list list[S_T]     per-layer attention distributions
      prior_list  list[P_T]     per-layer Gaussian priors
      sigma_list  list[sigma]   per-layer prior bandwidths
    """
    def __init__(self, win_size, d_model=512, n_heads=8, e_layers=3,
                 d_ff=512, dropout=0.0, output_attention=True):
        super().__init__()
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                AttentionLayer(
                    AnomalyAttention(win_size, mask_flag=False,
                                     attention_dropout=dropout,
                                     output_attention=output_attention),
                    d_model, n_heads),
                d_model, d_ff, dropout=dropout)
            for _ in range(e_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        series_list, prior_list, sigma_list = [], [], []
        for layer in self.encoder_layers:
            x, series, prior, sigma = layer(x)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)
        return self.norm(x), series_list, prior_list, sigma_list


# ============================================================
#  3. Spatial Branch — proposed contribution
# ============================================================
class HypergraphGAT(nn.Module):
    """
    Multi-head hypergraph attention (pure PyTorch, no torch_geometric required).
    Each sensor attends only to its neighbor set, using per-source-node softmax
    to avoid degree bias.
    """
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.1,
                 negative_slope=0.2, bias=True, residual=True):
        super().__init__()
        assert out_channels % heads == 0
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.heads        = heads
        self.head_dim     = out_channels // heads
        self.residual     = residual
        self.dropout      = dropout
        self.neg_slope    = negative_slope

        self.Wq     = nn.Linear(in_channels, out_channels, bias=bias)
        self.Wk     = nn.Linear(in_channels, out_channels, bias=bias)
        self.Wv     = nn.Linear(in_channels, out_channels, bias=bias)
        self.W_edge = nn.Linear(1, heads, bias=False)
        self.fc_out = nn.Linear(out_channels, out_channels, bias=bias)
        self.norm   = nn.LayerNorm(out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        for m in [self.Wq, self.Wk, self.Wv, self.fc_out]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.W_edge.weight)

    def forward(self, x, hyperedge_index, edge_features, return_attention=False):
        """
        Args:
            x:                [N, in_channels]
            hyperedge_index:  [2, E]  (src, dst)
            edge_features:    [E, 1]
            return_attention: whether to return attention weights (used for discrepancy loss)
        """
        N = x.size(0)
        Q = self.Wq(x).view(N, self.heads, self.head_dim)
        K = self.Wk(x).view(N, self.heads, self.head_dim)
        V = self.Wv(x).view(N, self.heads, self.head_dim)

        edge_bias = self.W_edge(edge_features)           # [E, H]
        src, dst  = hyperedge_index
        q_i, k_j, v_j = Q[src], K[dst], V[dst]

        alpha = (q_i * k_j).sum(dim=-1) / sqrt(self.head_dim)  # [E, H]
        alpha = alpha + edge_bias
        alpha = F.leaky_relu(alpha, self.neg_slope)
        alpha = scatter_softmax(alpha, src)
        alpha_drop = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.zeros(N, self.heads, self.head_dim, device=x.device, dtype=x.dtype)
        idx = src.unsqueeze(-1).unsqueeze(-1).expand(-1, self.heads, self.head_dim)
        out.scatter_add_(0, idx, v_j * alpha_drop.unsqueeze(-1))
        out = self.fc_out(out.view(N, self.out_channels))
        if self.residual:
            out = self.norm(out + x)

        if return_attention:
            return out, alpha
        return out


class HAR(nn.Module):
    """
    Hypergraph Attention Recalibration (HAR).
    Generates per-sensor recalibration weights from global average pooling,
    global max pooling, and a degree vector embedding, so that structurally
    more important sensors (higher degree) receive larger feature weights.
    """
    def __init__(self, num_sensors, d_h=16, reduction=4):
        super().__init__()
        self.degree_embed = nn.Sequential(
            nn.Linear(num_sensors, num_sensors // reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        mlp_in = d_h * 2 + num_sensors // reduction
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, mlp_in // 2),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_in // 2, num_sensors),
            nn.Sigmoid()
        )

    def forward(self, x, degree_vec):
        """
        Args:
            x:          [B, C, d_h]
            degree_vec: [C]  hypergraph out-degree (L-inf normalized)
        Returns:
            out:   [B, C, d_h]
            alpha: [B, C]  sensor importance weights
        """
        B = x.size(0)
        gap = torch.mean(x, dim=1)                                    # [B, d_h]
        gmp = torch.max(x, dim=1)[0]                                  # [B, d_h]
        deg = self.degree_embed(degree_vec.unsqueeze(0).expand(B, -1))  # [B, C//r]
        alpha = self.mlp(torch.cat([gap, gmp, deg], dim=-1))          # [B, C]
        return x * alpha.unsqueeze(-1), alpha


class RelationalBranch(nn.Module):
    """
    Spatial Branch: mines spatial association discrepancy via hypergraph attention.

    Pipeline:
      1. Dimension reduction:  [B, L, D] -> [B, L, C, d_h]
      2. Hypergraph attention:  compute Spatial Learned Association S_S
      3. Spatial association discrepancy: symmetric KL(S_S || P_S), P_S is topology-induced prior
      4. HAR recalibration:    emphasize structurally important sensors
      5. Dimension restoration: -> [B, L, D]
    """
    def __init__(self, enc_in, d_model=512, d_h=16, n_heads=4,
                 hyperedge_index=None, edge_features=None, dropout=0.1):
        super().__init__()
        self.enc_in = enc_in
        self.d_h    = d_h

        self.sensor_down     = nn.Linear(d_model, enc_in * d_h)
        self.hypergraph_conv = HypergraphGAT(d_h, d_h, heads=n_heads, dropout=dropout)
        self.har             = HAR(enc_in, d_h)
        self.sensor_up       = nn.Linear(enc_in * d_h, d_model)

        self.register_buffer('hyperedge_index',
            hyperedge_index if hyperedge_index is not None
            else torch.empty(2, 0, dtype=torch.long))
        self.register_buffer('edge_features',
            edge_features if edge_features is not None
            else torch.empty(0, 1))

        self._build_relational_prior()

    @torch.no_grad()
    def _build_relational_prior(self):
        """P_S(j|i) = normalized edge weights; registered as a non-learnable buffer."""
        if self.hyperedge_index.numel() == 0:
            self.register_buffer('relational_prior', torch.empty(0))
            return
        src = self.hyperedge_index[0]
        ef  = self.edge_features.squeeze(-1).clamp(min=1e-8)
        ef_sum = torch.zeros(self.enc_in, device=ef.device, dtype=ef.dtype)
        ef_sum.scatter_add_(0, src, ef)
        prior = ef / ef_sum[src].clamp(min=1e-8)
        self.register_buffer('relational_prior', prior)  # [E]

    @torch.no_grad()
    def _compute_degree_vector(self):
        """Out-degree vector, L-inf normalized to [0, 1]."""
        if self.hyperedge_index.numel() == 0:
            return torch.ones(self.enc_in, device=self.hyperedge_index.device)
        src = self.hyperedge_index[0]
        degree = src.bincount(minlength=self.enc_in).float().clamp(min=1e-8)
        return F.normalize(degree, p=float('inf'), dim=0)

    def compute_relational_discrepancy(self, rel_attn, num_groups, device,
                                       detach_prior=True):
        """
        Compute spatial association discrepancy D_S(i) = KL(S_S||P_S) + KL(P_S||S_S).

        Args:
            rel_attn:     [G*E, H]  hypergraph attention weights
            num_groups:   G = B x L
            detach_prior: set True during Phase 1 training (stop-gradient on P_S)
        Returns:
            [G, C]  per-(timestep, sensor) discrepancy
        """
        if self.hyperedge_index.numel() == 0 or rel_attn is None:
            return torch.zeros(num_groups, self.enc_in, device=device)

        G   = num_groups
        E   = self.hyperedge_index.size(1)
        src = self.hyperedge_index[0]

        s_raw = rel_attn.mean(dim=-1).view(G, E)
        s_sum = torch.zeros(G, self.enc_in, device=device, dtype=s_raw.dtype)
        s_sum.scatter_add_(1, src.unsqueeze(0).expand(G, -1), s_raw)
        S = s_raw / s_sum.gather(1, src.unsqueeze(0).expand(G, -1)).clamp(min=1e-8)

        P = self.relational_prior.unsqueeze(0).expand(G, -1)
        if detach_prior:
            P = P.detach()

        log_S = torch.log(S + 1e-8)
        log_P = torch.log(P.clamp(min=1e-8))
        kl = S * (log_S - log_P) + P * (log_P - log_S)  # [G, E]

        disc = torch.zeros(G, self.enc_in, device=device, dtype=kl.dtype)
        disc.scatter_add_(1, src.unsqueeze(0).expand(G, -1), kl)
        return disc  # [G, C]

    def forward(self, x):
        B, L, D = x.shape
        G = B * L

        h = self.sensor_down(x).view(B, L, self.enc_in, self.d_h)
        degree_vec = self._compute_degree_vector()

        rel_attn = None
        if self.hyperedge_index.numel() > 0:
            h_flat = h.reshape(G * self.enc_in, self.d_h)
            E = self.hyperedge_index.size(1)
            offset = torch.arange(G, device=x.device) * self.enc_in
            src_exp = self.hyperedge_index[0].repeat(G) + offset.repeat_interleave(E)
            dst_exp = self.hyperedge_index[1].repeat(G) + offset.repeat_interleave(E)
            expanded_index = torch.stack([src_exp, dst_exp], dim=0)
            expanded_efeat = self.edge_features.repeat(G, 1)
            h_flat, rel_attn = self.hypergraph_conv(
                h_flat, expanded_index, expanded_efeat, return_attention=True)
        else:
            h_flat = h.reshape(G * self.enc_in, self.d_h)

        h = h_flat.view(B, L, self.enc_in, self.d_h)

        rel_disc = self.compute_relational_discrepancy(
            rel_attn, G, x.device, detach_prior=True)
        rel_disc = rel_disc.view(B, L, self.enc_in)

        h_flat, alpha = self.har(h.reshape(G, self.enc_in, self.d_h), degree_vec)
        out = self.sensor_up(h_flat.view(B, L, -1))

        return out, alpha, rel_disc, rel_attn


# ============================================================
#  4. Gated Fusion
# ============================================================
class GatedFusion(nn.Module):
    """
    Adaptive weighted fusion of the two branch outputs.
      g = sigma(W_g * [Z_time; Z_spatial])
      g' = beta + (1 - beta) * g     # beta=0: fully data-driven
      Z_fuse = g' * Z_time + (1-g') * Z_spatial
    """
    def __init__(self, d_model, dropout=0.1, beta=0.0):
        super().__init__()
        self.beta = beta
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, z_time, z_rel):
        g = self.gate(torch.cat([z_time, z_rel], dim=-1))
        g = self.beta + (1 - self.beta) * g
        z = g * z_time + (1 - g) * z_rel
        return self.norm(z + self.ffn(self.norm(z)))


# ============================================================
#  5. MADM Full Model
# ============================================================
class MADM(nn.Module):
    """
    MADM: Multivariate Association Discrepancy Mining

    Input:  X in R^{B x L x C}
    Output (output_attention=True):
      output      [B, L, C]       reconstruction
      series_list list[S_T]       temporal learned associations (per layer)
      prior_list  list[P_T]       temporal prior associations (per layer)
      sigma_list  list[sigma]     prior bandwidths (per layer)
      alpha       [B*L, C]        HAR sensor weights
      rel_disc    [B, L, C]       per-sensor spatial association discrepancy
      rel_attn    [G*E, H]        spatial attention weights (used in training loss)
    """
    def __init__(self, win_size, enc_in, c_out,
                 d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, d_h=16,
                 hyperedge_index=None, edge_features=None,
                 output_attention=True, beta=0.0):
        super().__init__()
        self.output_attention = output_attention

        self.embedding        = DataEmbedding(enc_in, d_model, dropout)
        self.temporal_branch  = TemporalBranch(win_size, d_model, n_heads,
                                               e_layers, d_ff, dropout,
                                               output_attention=output_attention)
        self.relational_branch = RelationalBranch(enc_in, d_model, d_h,
                                                  n_heads // 2,
                                                  hyperedge_index, edge_features,
                                                  dropout)
        self.fusion    = GatedFusion(d_model, dropout, beta=beta)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        enc = self.embedding(x)

        z_time, series_list, prior_list, sigma_list = self.temporal_branch(enc)
        z_rel, alpha, rel_disc, rel_attn = self.relational_branch(enc)

        fused  = self.fusion(z_time, z_rel)
        output = self.projection(fused)

        if self.output_attention:
            return output, series_list, prior_list, sigma_list, alpha, rel_disc, rel_attn
        return output
