# network17.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = ["MultiScaleCNN", "EnhancedTransformer", "CrossModalFusion", "MAFNet", "Simple1DCNN"]


# ---------------------------- Utility ----------------------------
def _init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv1d,)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.Linear,)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d,)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


# ------------------------- MultiScale CNN -------------------------
class MultiScaleCNN(nn.Module):
    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.conv3 = nn.Conv1d(1, c, 3, padding=1)
        self.conv5 = nn.Conv1d(1, c, 5, padding=2)
        self.conv7 = nn.Conv1d(1, c, 7, padding=3)

        self.depthwise = nn.Conv1d(3 * c, 3 * c, 3, padding=1, groups=3 * c)
        self.pointwise = nn.Conv1d(3 * c, 128, 1)
        self.res_conv = nn.Conv1d(1, 128, 1)

        # Channel Attention (Squeeze-Excitation style)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(128, 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 128, 1),
            nn.Sigmoid()
        )

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, L]
        return: [B, 128, L]
        """
        identity = self.res_conv(x)            # [B,128,L]
        x3 = F.gelu(self.conv3(x))
        x5 = F.gelu(self.conv5(x))
        x7 = F.gelu(self.conv7(x))
        x = torch.cat([x3, x5, x7], dim=1)     # [B, 3c, L]
        x = self.depthwise(x)                  # [B, 3c, L]
        x = self.pointwise(x)                  # [B,128,L]
        att = self.ca(x)                       # [B,128,1]
        x = x * att                            # broadcast
        return x + identity



class EnhancedTransformer(nn.Module):
    def __init__(self, d_model: int = 128, nhead: int = 8, seq_len: int = 17, num_layers: int = 4) -> None:
        super().__init__()
        self.seq_len = int(seq_len)

        self.emb = nn.Conv1d(1, d_model, 1)
        self.pos_enc = nn.Parameter(torch.randn(1, d_model, self.seq_len))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            activation=F.gelu,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        local_heads = max(1, nhead // 2)
        self.local_attn = nn.MultiheadAttention(d_model, local_heads, dropout=0.1, batch_first=True)

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, L]
        return: [B, 256]
        """
        B, C, L = x.shape
        x = self.emb(x) + self.pos_enc          # [B, d_model, L]
        x = rearrange(x, "b c l -> b l c")      # [B, L, d_model]

        global_feat = self.transformer(x)       # [B, L, d_model]
        local_feat, _ = self.local_attn(x, x, x)# [B, L, d_model]

        x = torch.cat([global_feat, local_feat], dim=-1)  # [B, L, 2*d_model]
        x = rearrange(x, "b l c -> b c l")                # [B, 2*d_model, L]
        return F.adaptive_max_pool1d(x, 1).squeeze(-1)    # [B, 2*d_model]


# ------------------------- Fusion Module --------------------------
class CrossModalFusion(nn.Module):
    def __init__(self, cnn_dim: int, trans_dim: int) -> None:
        super().__init__()
        self.cnn_proj = nn.Linear(cnn_dim, 256)
        self.trans_proj = nn.Linear(trans_dim, 256)
        self.cross_attn = nn.MultiheadAttention(256, 8, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(512, 256),
            nn.Sigmoid()
        )
        self.apply(_init_weights)

    def forward(self, cnn_feat: torch.Tensor, trans_feat: torch.Tensor) -> torch.Tensor:
        """
        cnn_feat: [B, Cc]  (flattened CNN features)
        trans_feat: [B, Ct]
        return: [B, 256]
        """
        q = self.cnn_proj(cnn_feat).unsqueeze(1)    # [B,1,256]
        k = self.trans_proj(trans_feat).unsqueeze(1)# [B,1,256]
        v = k
        cross, _ = self.cross_attn(q, k, v)         # [B,1,256]
        cross = cross.squeeze(1)                    # [B,256]
        cnn_p = q.squeeze(1)                        # [B,256]

        combined = torch.cat([cnn_p, cross], dim=-1)# [B,512]
        gate = self.gate(combined)                  # [B,256] in (0,1)
        fused = gate * cnn_p + (1.0 - gate) * cross
        return fused



class MAFNet(nn.Module):
    def __init__(self, num_classes: int = 4, seq_len: int = 17) -> None:
        super().__init__()
        self.seq_len = int(seq_len)

        self.cnn_trunk = nn.Sequential(
            MultiScaleCNN(),          # [B,128,L]
            nn.MaxPool1d(2),          # [B,128,floor(L/2)]
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, 3, padding=1),
            nn.GELU(),

            nn.AdaptiveMaxPool1d(self.seq_len // 2)
        )


        self.transformer = EnhancedTransformer(d_model=128, nhead=8, seq_len=self.seq_len)

        cnn_flat_dim = 256 * (self.seq_len // 2)
        self.fusion = CrossModalFusion(cnn_dim=cnn_flat_dim, trans_dim=256)

        self.classifier = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, seq_len]
        return: logits [B, num_classes]
        """
        B, C, L = x.shape
        cnn_feat = self.cnn_trunk(x)                  # [B,256,floor(L/2)]
        cnn_flat = cnn_feat.flatten(start_dim=1)      # [B, 256*floor(L/2)]

        trans_feat = self.transformer(x)              # [B,256]
        fused = self.fusion(cnn_flat, trans_feat)     # [B,256]
        return self.classifier(fused)                 # [B,num_classes]


# --------------------------- CNN ---------------------------
class Simple1DCNN(nn.Module):
    def __init__(self, num_classes: int = 2, seq_len: int = 17) -> None:
        super().__init__()
        self.seq_len = int(seq_len)

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
