"""
models.py
---------
PhysE-Inv: Physics-Encoded Inverse Modeling for Arctic Snow Depth Estimation.

Models included:
    - MultiHeadSelfAttention      : Multi-head self-attention block
    - LSTMContrastiveWithAttention: Main PhysE-Inv model (LSTM + Attention + Physics head)
    - BiLSTMContrastive           : Bidirectional LSTM baseline
    - ResNet1DContrastive         : 1D ResNet baseline
    - ODEFunc / NeuralODEContrastive: Neural ODE baseline

Contrastive learning utilities:
    - augment_data                : Gaussian noise augmentation
    - contrastive_loss            : NT-Xent contrastive loss

Reference:
    Sampath et al., "PhysE-Inv: Physics-Encoded Inverse Modeling for
    Arctic Snow Depth Estimation." Under review, 2025.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint


# ---------------------------------------------------------------------------
# Contrastive Learning Utilities
# ---------------------------------------------------------------------------

def augment_data(x: torch.Tensor, noise_std: float = 0.01) -> torch.Tensor:
    """
    Gaussian noise augmentation for contrastive learning.

    The noise level sigma=0.01 corresponds to the approximate observational
    uncertainty in ERA5 snow density fields at Arctic Ocean grid points.

    Args:
        x        : Input tensor.
        noise_std: Standard deviation of Gaussian noise.

    Returns:
        Augmented tensor x + epsilon, where epsilon ~ N(0, noise_std).
    """
    return x + torch.randn_like(x) * noise_std


def contrastive_loss(
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    scale: float = 0.05,
) -> torch.Tensor:
    """
    NT-Xent contrastive loss (normalized temperature-scaled cross entropy).

    Applied to latent parameter representations to regularize the inversion
    process under ill-posed conditions and sparse observations.

    Args:
        z_i  : Embeddings from original sequence, shape (B, D).
        z_j  : Embeddings from augmented sequence, shape (B, D).
        scale: Temperature scaling factor.

    Returns:
        Scalar contrastive loss.
    """
    batch_size = z_i.size(0)
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)

    combined_z          = torch.cat([z_i, z_j], dim=0)
    combined_similarity = torch.matmul(combined_z, combined_z.T) / scale

    labels          = torch.arange(batch_size, device=z_i.device)
    combined_labels = torch.cat([labels, labels], dim=0)

    mask               = ~torch.eye(combined_labels.shape[0], device=combined_labels.device).bool()
    combined_similarity = combined_similarity.masked_select(mask).view(combined_labels.shape[0], -1)

    return F.cross_entropy(combined_similarity, combined_labels)


# ---------------------------------------------------------------------------
# 1. Multi-Head Self-Attention Block
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention block with residual connection and layer norm.

    Args:
        hidden_dim: Embedding dimension.
        num_heads : Number of attention heads.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 2):
        super(MultiHeadSelfAttention, self).__init__()
        self.attn    = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.norm    = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x)
        return self.norm(x + self.dropout(attn_output))


# ---------------------------------------------------------------------------
# 2. PhysE-Inv — Main Model
#    LSTM Encoder → Multi-Head Attention → LSTM Decoder
#    + Physics Parameter Head (w, b, c)
#    + Contrastive regularization on latent representations
# ---------------------------------------------------------------------------

class LSTMContrastiveWithAttention(nn.Module):
    """
    PhysE-Inv: Physics-encoded inverse modeling framework.

    Architecture:
        - LSTM encoder captures temporal dynamics
        - Multi-head attention identifies critical temporal transitions
        - LSTM decoder reconstructs the sequence
        - fc_depth    : predicts snow depth proxy directly
        - fc_params   : estimates physically constrained parameters {w, b, c}
          where w in [-1,1], b > 0, c in [-10, 10]

    The three parameters are grounded in the hydrostatic balance equation:
        depth_proxy = w * density + b + c * auxiliary_term

    Args:
        input_dim   : Number of input features (default: 1, snow density).
        hidden_dim  : LSTM hidden state size.
        num_layers  : Number of stacked LSTM layers.
        output_dim  : Output dimension.
        dropout_rate: Dropout probability.
        num_heads   : Number of attention heads.
    """

    def __init__(
        self,
        input_dim: int    = 1,
        hidden_dim: int   = 64,
        num_layers: int   = 2,
        output_dim: int   = 1,
        dropout_rate: float = 0.4,
        num_heads: int    = 4,
    ):
        super(LSTMContrastiveWithAttention, self).__init__()
        self.encoder_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout_rate
        )
        self.attention    = MultiHeadSelfAttention(hidden_dim, num_heads=num_heads)
        self.decoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout_rate
        )
        self.fc_depth  = nn.Linear(hidden_dim, output_dim)
        self.fc_params = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),       # outputs raw [w, b, c]
        )

    def _apply_physics_constraints(self, params: torch.Tensor) -> torch.Tensor:
        """
        Apply physical bounds to estimated parameters.
            w in [-1, 1]   : density-depth coupling coefficient
            b > 0          : positive baseline density offset
            c in [-10, 10] : scaling factor within climatological range
        """
        w = torch.tanh(params[:, 0:1])                      # [-1, 1]
        b = F.softplus(params[:, 1:2])                       # > 0
        c = 10.0 * torch.tanh(params[:, 2:3])               # [-10, 10]
        return torch.cat([w, b, c], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        x_aug: torch.Tensor,
    ) -> tuple:
        """
        Forward pass.

        Args:
            x    : Original input sequence, shape (B, T, input_dim).
            x_aug: Augmented input sequence for contrastive loss.

        Returns:
            depth_pred      : Predicted snow depth proxy, shape (B, T, 1).
            estimated_depth : Physics-refined depth prediction, shape (B, T, 1).
            params          : Constrained physical parameters {w, b, c}, shape (B, 3).
            z_x             : Latent representation of original sequence.
            z_aug           : Latent representation of augmented sequence.
        """
        # Original sequence
        enc_x, (h_x, c_x) = self.encoder_lstm(x)
        attn_x             = self.attention(enc_x)
        dec_x, _           = self.decoder_lstm(attn_x, (h_x, c_x))

        # Augmented sequence (for contrastive loss)
        enc_aug, (h_aug, c_aug) = self.encoder_lstm(x_aug)
        attn_aug                = self.attention(enc_aug)
        dec_aug, _              = self.decoder_lstm(attn_aug, (h_aug, c_aug))

        # Predictions
        depth_pred = self.fc_depth(dec_x)

        # Physics parameter estimation from last timestep
        raw_params      = self.fc_params(dec_x[:, -1, :])
        params          = self._apply_physics_constraints(raw_params)
        w, b, c         = params[:, 0:1], params[:, 1:2], params[:, 2:3]

        # Physics-refined depth: hydrostatic balance proxy
        estimated_depth = w * depth_pred + b + c * x

        # Latent representations for contrastive loss
        z_x   = dec_x[:, -1, :]
        z_aug = dec_aug[:, -1, :]

        return depth_pred, estimated_depth, params, z_x, z_aug


# ---------------------------------------------------------------------------
# 3. BiLSTM Baseline
# ---------------------------------------------------------------------------

class BiLSTMContrastive(nn.Module):
    """
    Bidirectional LSTM baseline with physics parameter head.

    Args:
        input_dim   : Number of input features.
        hidden_dim  : LSTM hidden state size.
        num_layers  : Number of stacked LSTM layers.
        output_dim  : Output dimension.
        dropout_rate: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int    = 1,
        hidden_dim: int   = 64,
        num_layers: int   = 2,
        output_dim: int   = 1,
        dropout_rate: float = 0.4,
    ):
        super(BiLSTMContrastive, self).__init__()
        self.hidden_dim   = hidden_dim
        self.num_directions = 2

        self.encoder_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout_rate, bidirectional=True
        )
        self.decoder_lstm = nn.LSTM(
            hidden_dim * 2, hidden_dim, num_layers,
            batch_first=True, dropout=dropout_rate
        )
        self.fc_depth  = nn.Linear(hidden_dim, output_dim)
        self.fc_params = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x: torch.Tensor, x_aug: torch.Tensor) -> tuple:
        enc_x, _   = self.encoder_lstm(x)
        dec_x, _   = self.decoder_lstm(enc_x)
        enc_aug, _ = self.encoder_lstm(x_aug)
        dec_aug, _ = self.decoder_lstm(enc_aug)

        depth_pred = self.fc_depth(dec_x)
        params     = self.fc_params(dec_x[:, -1, :])
        z_x        = dec_x[:, -1, :]
        z_aug      = dec_aug[:, -1, :]

        return depth_pred, params, z_x, z_aug


# ---------------------------------------------------------------------------
# 4. 1D ResNet Baseline
# ---------------------------------------------------------------------------

class BasicBlock1D(nn.Module):
    """1D residual block for ResNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample=None,
    ):
        super(BasicBlock1D, self).__init__()
        self.conv1      = nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1        = nn.BatchNorm1d(out_channels)
        self.relu       = nn.ReLU(inplace=True)
        self.conv2      = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.bn2        = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out      = self.relu(self.bn1(self.conv1(x)))
        out      = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet1DContrastive(nn.Module):
    """
    1D ResNet baseline with physics parameter head.

    Args:
        input_dim : Number of input channels.
        base_dim  : Base number of convolutional filters.
        blocks    : List specifying number of residual blocks per stage.
        output_dim: Output dimension.
    """

    def __init__(
        self,
        input_dim: int  = 1,
        base_dim: int   = 32,
        blocks: list    = [1, 1],
        output_dim: int = 1,
    ):
        super(ResNet1DContrastive, self).__init__()
        self.in_channels = base_dim
        self.conv1   = nn.Conv1d(input_dim, base_dim, 7, stride=2, padding=3)
        self.bn1     = nn.BatchNorm1d(base_dim)
        self.relu    = nn.ReLU()
        self.maxpool = nn.MaxPool1d(3, stride=2, padding=1)

        self.layer1  = self._make_layer(base_dim,      blocks[0])
        self.layer2  = self._make_layer(base_dim * 2,  blocks[1], stride=2)

        self.avgpool   = nn.AdaptiveAvgPool1d(1)
        self.fc_depth  = nn.Linear(base_dim * 2, output_dim)
        self.fc_params = nn.Sequential(
            nn.Linear(base_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int = 1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )
        layers = [BasicBlock1D(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, x_aug: torch.Tensor) -> tuple:
        def encode(inp):
            inp = inp.transpose(1, 2)
            inp = self.maxpool(self.relu(self.bn1(self.conv1(inp))))
            inp = self.layer2(self.layer1(inp))
            return self.avgpool(inp).squeeze(-1)

        feat     = encode(x)
        feat_aug = encode(x_aug)

        depth_pred = self.fc_depth(feat).unsqueeze(-1)
        params     = self.fc_params(feat)

        return depth_pred, params, feat, feat_aug


# ---------------------------------------------------------------------------
# 5. Neural ODE Baseline
# ---------------------------------------------------------------------------

class ODEFunc(nn.Module):
    """ODE function for Neural ODE — defines the derivative dh/dt."""

    def __init__(self, hidden_dim: int):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class NeuralODEContrastive(nn.Module):
    """
    Neural ODE baseline with attention and physics parameter head.

    Args:
        input_dim  : Number of input features.
        hidden_dim : Hidden state size.
        output_dim : Output dimension.
        t_span     : Integration time points for ODE solver.
    """

    def __init__(
        self,
        input_dim: int  = 1,
        hidden_dim: int = 64,
        output_dim: int = 1,
        t_span: torch.Tensor = None,
    ):
        super(NeuralODEContrastive, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.ode_func   = ODEFunc(hidden_dim)
        self.attention  = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm       = nn.LayerNorm(hidden_dim)
        self.fc_depth   = nn.Linear(hidden_dim, output_dim)
        self.fc_params  = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )
        self.t_span = t_span if t_span is not None else torch.tensor([0.0, 1.0])

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        h0  = self.input_proj(x[:, 0, :])
        out = odeint(self.ode_func, h0, self.t_span)
        out = out.permute(1, 0, 2)
        attn_out, _ = self.attention(out, out, out)
        return self.norm(out + attn_out)

    def forward(self, x: torch.Tensor, x_aug: torch.Tensor) -> tuple:
        dec_x   = self._encode(x)
        dec_aug = self._encode(x_aug)

        depth_pred = self.fc_depth(dec_x)
        params     = self.fc_params(dec_x[:, -1, :])
        z_x        = dec_x[:, -1, :]
        z_aug      = dec_aug[:, -1, :]

        return depth_pred, params, z_x, z_aug
