"""
S-Adapter (Statistical Adapter) - sesuai paper
==============================================
Cai, R., dkk. "S-Adapter: Generalizing Vision Transformer for Face Anti-Spoofing
with Statistical Tokens." IEEE Transactions on Information Forensics and Security, 2024.
arXiv:2309.04038.

Alur (Bagian III-A paper), untuk token map berukuran C x H x W:

  token X (N x C, tanpa CLS)
    -> bentuk ulang menjadi peta 2D            X^M  : C x H x W            (III-A1)
    -> proyeksi turun 1x1 ke k kanal           (k = 8, paper: "N^P x 8")
    -> konvolusi 3x3 + central difference      Z* = (1-theta) Z + theta Z^g  (Pers. 3-5)
         Z^g_n = sum_p w(p) (Z_p - Z_n),  theta = 0.7 (HTER terendah, IV-E5)
    -> histogram statistik lunak (soft binning)                             (Pers. 6-8)
         Hist_c = (1/JK) sum_{3x3} exp( -gamma_c^2 (Z*_c - mu_c)^2 )
         jendela J = K = 3, stride 1, padding 1; mu & gamma dipelajari
    -> proyeksi naik ke C (paper: "projected ... via a linear layer")
    -> dijumlahkan ke token masukan (residual)                              (IV-E4)

Interpretasi implementasi (detail yang tidak dinyatakan eksplisit di teks paper):
  * Konvolusi central difference ditulis dalam bentuk ekuivalennya:
        Z* = conv3x3(z) - theta * (sum bobot kernel) * z
    karena (1-theta)Z + theta*sum_p w(p)(Z_p - Z_n) = Z - theta*Z_n*sum_p w(p).
  * mu dan gamma dari "dua konvolusi 1x1 piksel-per-piksel" diimplementasikan sebagai
    parameter per kanal (1x1 depthwise = satu bobot/bias per kanal).
  * Proyeksi turun 1x1 ke 8 kanal sebelum konvolusi 3x3 dipilih karena menghasilkan
    tambahan parameter ~0,33 M pada ViT-Base dengan 24 adapter, sama dengan Tabel VII
    paper (ViT 85,65 M -> ViT-S-Adapter 85,98 M).
  * Proyeksi naik diinisialisasi nol sehingga di awal training adapter belum
    mengubah keluaran backbone pretrained (praktik umum adapter).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StatisticalAdapter(nn.Module):
    """S-Adapter. Bekerja pada token (B, N, C) dengan grid H x W, atau peta (B, C, H, W)."""

    def __init__(self, dim: int, hidden_dim: int = 8, theta: float = 0.7):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.theta = theta

        self.down_proj = nn.Linear(dim, hidden_dim)                                   # C -> k
        self.conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.hist_center = nn.Parameter(torch.zeros(hidden_dim))                      # mu
        self.hist_width = nn.Parameter(torch.ones(hidden_dim))                        # gamma
        self.hist_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)             # 1/JK, J=K=3
        self.up_proj = nn.Linear(hidden_dim, dim)                                     # k -> C

        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def _statistik(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, k, H, W) -> histogram statistik lunak (B, k, H, W)."""
        z_conv = self.conv(z)
        if self.theta > 0:
            kernel_sum = self.conv.weight.sum(dim=(2, 3), keepdim=True)              # (k, k, 1, 1)
            z_conv = z_conv - self.theta * F.conv2d(z, kernel_sum)                  # central difference
        mu = self.hist_center.view(1, -1, 1, 1)
        gamma = self.hist_width.view(1, -1, 1, 1)
        return self.hist_pool(torch.exp(-(gamma * (z_conv - mu)) ** 2))

    def forward_tokens(self, tokens: torch.Tensor, grid_hw) -> torch.Tensor:
        """
        tokens : (B, N, C) token patch (TANPA CLS), N = H * W
        return : (B, N, C) keluaran adapter Y (BELUM dijumlahkan; residual dilakukan pemanggil)
        """
        B, N, _ = tokens.shape
        H, W = grid_hw
        z = self.down_proj(tokens)                                   # (B, N, k)
        z = z.transpose(1, 2).reshape(B, self.hidden_dim, H, W)      # (B, k, H, W)
        hist = self._statistik(z)                                    # (B, k, H, W)
        return self.up_proj(hist.flatten(2).transpose(1, 2))         # (B, N, C)

    def forward_map(self, fmap: torch.Tensor) -> torch.Tensor:
        """
        fmap   : (B, C, H, W) peta fitur (misalnya ConvNeXt); setiap posisi = satu token
        return : (B, C, H, W) keluaran adapter Y (BELUM dijumlahkan)
        """
        B, C, H, W = fmap.shape
        y = self.forward_tokens(fmap.flatten(2).transpose(1, 2), (H, W))
        return y.transpose(1, 2).reshape(B, C, H, W)


class ViTInputRenorm(nn.Module):
    """
    Mengubah normalisasi input dari ImageNet (mean 0.485/0.456/0.406, std 0.229/0.224/0.225,
    dipakai ConvNeXt & pipeline data) ke normalisasi pretrained ViT timm (mean 0.5, std 0.5).
    Tidak punya parameter yang dilatih.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("in_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("in_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.in_std + self.in_mean - 0.5) / 0.5


def vit_forward_dengan_s_adapter(vit: nn.Module, adapters_msa, adapters_mlp, x: torch.Tensor) -> torch.Tensor:
    """
    Forward ViT timm dengan 2 S-Adapter per blok (setelah MHSA dan setelah MLP), Pers. 2 paper:
        x = x + MHSA(LN(x));  x = x + A_msa(x)
        x = x + MLP(LN(x));   x = x + A_mlp(x)
    S-Adapter hanya bekerja pada token patch; token CLS tidak diubah adapter.
    Return: token setelah LayerNorm akhir, (B, 1 + N, d)
    """
    grid = vit.patch_embed.grid_size
    x = vit.patch_embed(x)
    x = vit._pos_embed(x)
    x = vit.patch_drop(x)
    x = vit.norm_pre(x)

    def tambah(x, adapter):
        y = adapter.forward_tokens(x[:, 1:], grid)
        return torch.cat([x[:, :1], x[:, 1:] + y], dim=1)

    for blk, a_msa, a_mlp in zip(vit.blocks, adapters_msa, adapters_mlp):
        x = x + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(x))))
        x = tambah(x, a_msa)
        x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))
        x = tambah(x, a_mlp)
    return vit.norm(x)
