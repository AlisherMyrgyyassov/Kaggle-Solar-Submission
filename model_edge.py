"""EdgeAttNet-style graft: edge-guided MHSA at the ConvNeXt-UNet bottleneck.

Reconstructed from EdgeAttNet (arXiv:2509.02964) Sec. V-C/D, since the public
repo does not actually ship the model class (model/edgeattnet_model.py is a
duplicate of main.py). Architecture-only reuse; no pretrained weights.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_v2 import ConvNeXtUNet


class EdgePrior(nn.Module):
    """E = sigmoid(Conv3x3(x)) -> downsample -> Conv1x1 to bottleneck channels."""

    def __init__(self, in_channels=3, bottleneck_ch=768):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 3, padding=1)
        self.proj = nn.Conv2d(1, bottleneck_ch, 1)

    def forward(self, x, size):
        e = torch.sigmoid(self.conv(x))
        e = F.interpolate(e, size=size, mode="bilinear", align_corners=False)
        return self.proj(e)


class EG_MHSA(nn.Module):
    """Multi-head self-attention with the edge prior injected into Q and K.

    Paper Sec V-D: Q = x + E', K = x + E', V = x (E' is the projected edge map).
    No positional encodings. Pre-LayerNorm, residual + dropout.
    """

    def __init__(self, dim, n_heads=4, dropout=0.1):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, e=None):
        B, C, H, W = x.shape
        N = H * W
        xf = x.flatten(2).transpose(1, 2)
        h = self.norm(xf)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        if e is not None:
            ef = e.flatten(2).transpose(1, 2)
            q = q + ef
            k = k + ef
        q = q.reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.drop(out)
        out = (xf + out).transpose(1, 2).reshape(B, C, H, W)
        return out


class ConvNeXtEdgeAttUNet(ConvNeXtUNet):
    """Pretrained ConvNeXt encoder + n_blocks EG-MHSA at the bottleneck + smp decoder.

    use_edge=False gives the plain-MHSA control (Q=K=V=x, no edge prior).
    """

    def __init__(self, encoder_name="convnext_tiny", in_channels=3, classes=1,
                 pretrained=True, n_blocks=2, use_edge=True, n_heads=4, dropout=0.1):
        super().__init__(encoder_name=encoder_name, in_channels=in_channels,
                         classes=classes, pretrained=pretrained)
        bneck = self.ENCODER_CHANNELS[encoder_name][-1]
        self.use_edge = use_edge
        self.edge_prior = EdgePrior(in_channels, bneck) if use_edge else None
        self.blocks = nn.ModuleList([EG_MHSA(bneck, n_heads, dropout) for _ in range(n_blocks)])

    def forward(self, x):
        skips = self.encoder(x)
        b = skips[-1]
        e = self.edge_prior(x, b.shape[-2:]) if self.use_edge else None
        for blk in self.blocks:
            b = blk(b, e)
        features = [x] + list(skips[:-1]) + [b]
        out = self.decoder(features)
        return self.head(out)


def get_edgeatt_unet(config, encoder_name="convnext_tiny", pretrained=True,
                     use_edge=True, n_blocks=2, n_heads=4, dropout=0.1):
    model = ConvNeXtEdgeAttUNet(
        encoder_name=encoder_name,
        in_channels=config.IN_CHANNELS,
        classes=config.NUM_CLASSES,
        pretrained=pretrained,
        n_blocks=n_blocks,
        use_edge=use_edge,
        n_heads=n_heads,
        dropout=dropout,
    )
    return model.to(config.DEVICE)
