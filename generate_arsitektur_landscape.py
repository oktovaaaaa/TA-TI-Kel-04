"""
Diagram arsitektur LANDSCAPE (gaya paper: alur horizontal, blok vertikal, minim teks).

Output -> assets/Arsitektur_Detail/
  04_ConvNeXt_Branch_Landscape.png
  05_ViT_Branch_Landscape.png
  06_Fusion_Module_Landscape.png

Konfigurasi H5 (Champion): ConvNeXt-Small + ViT-Small, d = 384, h = 6, r = 64.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Polygon, Rectangle

OUT = Path(__file__).parent / "assets" / "Arsitektur_Detail"
OUT.mkdir(parents=True, exist_ok=True)

W, H, DPI = 20.0, 10.0, 250
FLAT = False  # True -> blok 2D datar (tanpa efek 3D)
TXT, MUTED = "#1b1b1b", "#555555"

C_CONV = "#9fdc8c"     # blok ConvNeXt
C_DOWN = "#ffe08a"     # downsample / reshape
C_NORM = "#d6d6d6"     # normalisasi
C_ACT = "#ffc58f"      # aktivasi
C_LIN = "#a9cdf2"      # linear / conv 1x1
C_ADP = "#f4a3a3"      # S-Adapter
C_ATT = "#fde9a8"      # attention
C_MISC = "#e6e0f5"     # pooling / dll
C_TOK_C, C_TOK_V, C_CLS, C_POS = "#9fdc8c", "#a9cdf2", "#f28b82", "#cdb4f0"
GREEN_D, BLUE_D, RED_D, PURPLE_D, ORANGE_D = "#2e7d32", "#1f4e8c", "#c62828", "#5b2a86", "#e07b1f"


# ───────────────────────── helpers ─────────────────────────
def canvas():
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    return fig, ax


def shade(c, f):
    r, g, b = to_rgb(c)
    return (r * f, g * f, b * f)


def t(ax, x, y, s, fs=10, w="normal", c=TXT, ha="center", va="center", rot=0, st="normal", z=8):
    ax.text(x, y, s, fontsize=fs, fontweight=w, color=c, ha=ha, va=va, rotation=rot,
            style=st, zorder=z)


def slab(ax, x, y, w, h, c, label, d=0.1, fs=9, ec="#333333"):
    """Blok 3D vertikal dengan label diputar 90° (atau 2D datar bila FLAT)."""
    if FLAT:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=c, ec=shade(c, 0.5), lw=1.4, zorder=4))
        t(ax, x + w / 2, y + h / 2, label, fs=fs, rot=90)
        return x + w
    ax.add_patch(Rectangle((x, y), w, h, fc=c, ec=ec, lw=0.9, zorder=4))
    ax.add_patch(Polygon([(x, y), (x + d, y - d), (x + w + d, y - d), (x + w, y)],
                         fc=shade(c, 0.88), ec=ec, lw=0.9, zorder=4))
    ax.add_patch(Polygon([(x + w, y), (x + w + d, y - d), (x + w + d, y + h - d), (x + w, y + h)],
                         fc=shade(c, 0.72), ec=ec, lw=0.9, zorder=4))
    t(ax, x + w / 2, y + h / 2, label, fs=fs, rot=90)
    return x + w + d


def arr(ax, x1, y1, x2, y2, c=TXT, lw=1.4, ms=12, ls="-", z=6):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=lw, mutation_scale=ms,
                                linestyle=ls, shrinkA=0, shrinkB=0), zorder=z)


def ln(ax, xs, ys, c=TXT, lw=1.3, ls="-", z=5):
    ax.plot(xs, ys, color=c, lw=lw, ls=ls, zorder=z, solid_capstyle="round")


def plus(ax, x, y, r=0.17):
    ax.add_patch(Circle((x, y), r, fc="white", ec=TXT, lw=1.3, zorder=7))
    ln(ax, [x - r * .6, x + r * .6], [y, y], z=8)
    ln(ax, [x, x], [y - r * .6, y + r * .6], z=8)


def skip(ax, x1, x2, y_mid, y_top, r=0.17):
    """Residual: cabang di x1 -> naik -> ke x2 -> turun ke ⊕."""
    ax.add_patch(Circle((x1, y_mid), 0.04, fc=TXT, zorder=7))
    ln(ax, [x1, x1, x2], [y_mid, y_top, y_top])
    arr(ax, x2, y_top, x2, y_mid - r, ms=10)


def frame(ax, x, y, w, h, ec, title=None, fc="none", ls=(0, (5, 3)), lw=1.6, tfs=10.5,
          tpos="bottom"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.15",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=1))
    if title:
        ty = y + h - 0.22 if tpos == "bottom" else y + 0.25
        t(ax, x + w / 2, ty, title, fs=tfs, w="bold", c=ec)


def panel(ax, x, y, w, h, fc, ec, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.2",
                                fc=fc, ec=ec, lw=1.3, zorder=0))
    t(ax, x + 0.25, y + 0.3, title, fs=13, w="bold", c=ec, ha="left")


def vstack(ax, x, yc, n, c, s=0.2, g=0.05, first=None, cross_first=False):
    """Tumpukan token vertikal berpusat di yc. Return x kanan."""
    total = (n + 1) * (s + g) + 0.25
    y = yc - total / 2
    for i in range(n):
        ax.add_patch(Rectangle((x, y), s, s, fc=first if (first and i == 0) else c,
                               ec="#333", lw=0.8, zorder=5))
        if cross_first and i == 0:
            ln(ax, [x - .05, x + s + .05], [y - .05, y + s + .05], c=RED_D, lw=2, z=7)
            ln(ax, [x - .05, x + s + .05], [y + s + .05, y - .05], c=RED_D, lw=2, z=7)
        y += s + g
    t(ax, x + s / 2, y + 0.08, "⋮", fs=10)
    y += 0.25
    ax.add_patch(Rectangle((x, y), s, s, fc=c, ec="#333", lw=0.8, zorder=5))
    return x + s


def cube(ax, x, y, s, n, c, ec, d=0.22):
    if FLAT:
        ax.add_patch(Rectangle((x, y), s, s, fc=c, ec=ec, lw=1.2, zorder=5))
        for i in range(1, n):
            ln(ax, [x + s * i / n] * 2, [y, y + s], c=ec, lw=.5, z=6)
            ln(ax, [x, x + s], [y + s * i / n] * 2, c=ec, lw=.5, z=6)
        return x + s
    ax.add_patch(Polygon([(x, y), (x + d, y - d), (x + s + d, y - d), (x + s, y)],
                         fc=shade(c, .88), ec=ec, lw=.9, zorder=4))
    ax.add_patch(Polygon([(x + s, y), (x + s + d, y - d), (x + s + d, y + s - d), (x + s, y + s)],
                         fc=shade(c, .72), ec=ec, lw=.9, zorder=4))
    ax.add_patch(Rectangle((x, y), s, s, fc=c, ec=ec, lw=1, zorder=5))
    for i in range(1, n):
        ln(ax, [x + s * i / n] * 2, [y, y + s], c=ec, lw=.5, z=6)
        ln(ax, [x, x + s], [y + s * i / n] * 2, c=ec, lw=.5, z=6)
    return x + s + d


def face(ax, x, y, s, grid_n=0, ec=TXT):
    sq = FancyBboxPatch((x, y), s, s, boxstyle="round,pad=0,rounding_size=0.06",
                        fc="#f5f5f5", ec=ec, lw=1.2, zorder=4)
    ax.add_patch(sq)
    ax.add_patch(Circle((x + s / 2, y + s * .38), s * .19, fc="#222", zorder=5))
    body = Ellipse((x + s / 2, y + s * 1.0), s * .72, s * .62, fc="#222", zorder=5)
    ax.add_patch(body)
    body.set_clip_path(sq)
    if grid_n:
        for i in range(1, grid_n):
            ln(ax, [x + s * i / grid_n] * 2, [y, y + s], c="#4a90d9", lw=.5, z=6)
            ln(ax, [x, x + s], [y + s * i / grid_n] * 2, c="#4a90d9", lw=.5, z=6)
    return x + s


def to_box(ax, x, yc, h, c, label):
    ax.add_patch(FancyBboxPatch((x, yc - h / 2), 0.55, h, boxstyle="round,pad=0,rounding_size=0.1",
                                fc=shade(c, 1.0), ec=PURPLE_D, lw=1.4, zorder=4))
    t(ax, x + 0.275, yc, label, fs=10, w="bold", rot=90, c=PURPLE_D)


def legend(ax, x, y, items, fs=9.5, step=0.42):
    for i, (c, lab, kind) in enumerate(items):
        yy = y + i * step
        if kind == "box":
            ax.add_patch(Rectangle((x, yy - .13), .35, .26, fc=c, ec="#333", lw=.8, zorder=5))
        elif kind == "line":
            ln(ax, [x, x + .35], [yy, yy], c=c, lw=2, ls=(0, (4, 2)))
        elif kind == "plus":
            plus(ax, x + .17, yy, r=.13)
        t(ax, x + .5, yy, lab, fs=fs, ha="left")


def save(fig, name):
    if FLAT:
        name = (name.replace("07_", "10_").replace("08_", "11_").replace("09_", "12_")
                .replace("_General.png", "_General_2D.png"))
    p = OUT / name
    fig.savefig(p, dpi=DPI, facecolor="white")
    plt.close(fig)
    print("Saved:", p)


def chain(ax, x, y, h, blocks, w=0.5, gap=0.42, yc=None, dims=None, dim_y=None):
    """Gambar rangkaian slab horizontal + panah. blocks: list (color,label) atau ('+',None)."""
    yc = y + h / 2 if yc is None else yc
    pos = []
    for i, (c, lab) in enumerate(blocks):
        if c == "+":
            plus(ax, x + .17, yc)
            pos.append((x + .17, "plus"))
            xe = x + .34
        else:
            xe = slab(ax, x, y, w, h, c, lab)
            pos.append((x + w / 2, "slab"))
        if dims and dims[i]:
            t(ax, pos[-1][0], dim_y, dims[i], fs=8, c=MUTED, st="italic")
        if i < len(blocks) - 1:
            arr(ax, xe + .02, yc, xe + gap - .02, yc, ms=10)
        x = xe + gap
    return x - gap, pos


# ═══════════════════════ 1. ConvNeXt ═══════════════════════
def convnext(general=False):
    L = lambda a, b: b if general else a
    fig, ax = canvas()
    t(ax, W / 2, 0.42, L("ConvNeXt Branch (Local Feature Extraction) — ConvNeXt-Small + S-Adapter", "ConvNeXt Branch (Local Feature Extraction) + S-Adapter"),
      fs=17, w="bold", c=GREEN_D)
    panel(ax, 0.2, 0.8, 19.6, 4.45, "#f1f8ef", GREEN_D, "")
    Y, Hs = 1.75, 2.4
    YC = Y + Hs / 2

    x = 0.45
    face(ax, x, YC - .6, 1.2)
    t(ax, x + .6, YC + .95, "Input\n224×224×3", fs=9.5, w="bold")
    arr(ax, 1.7, YC, 2.0, YC)
    xe = slab(ax, 2.05, Y, .5, Hs, C_CONV, "Stem: Conv 4×4, s=4 + LN")
    t(ax, 2.35, Y - .35, L("56×56×96", "56×56×C₁"), fs=8.5, c=MUTED, st="italic")
    arr(ax, xe + .02, YC, 2.95, YC)

    stages = [("Stage 1", "×3", "56×56×96", False, False),
              ("Stage 2", "×3", "28×28×192", True, False),
              ("Stage 3", "×27", "14×14×384", True, False),
              ("Stage 4", "×3", "7×7×768", True, True)]
    if general:
        stages = [("Stage 1", "×N₁", "56×56×C₁", False, False),
                  ("Stage 2", "×N₂", "28×28×C₂", True, False),
                  ("Stage 3", "×N₃", "14×14×C₃", True, False),
                  ("Stage 4", "×N₄", "7×7×C₄", True, False)]
    sx = 2.95
    blk_pos = {}
    for name, rep, dim, has_down, train in stages:
        w = 1.45 if has_down else 1.05
        frame(ax, sx, 1.2, w, 3.85, ORANGE_D if train else GREEN_D, f"{name}  {rep}",
              lw=2.2 if train else 1.5, tfs=9.5)
        t(ax, sx + w / 2, 1.42, dim, fs=8.5, c=MUTED, st="italic")
        bx = sx + .18
        if has_down:
            e = slab(ax, bx, Y, .45, Hs, C_DOWN, "Downsample: LN + Conv 2×2, s=2", fs=8.5)
            arr(ax, e + .01, YC, e + .14, YC, ms=8)
            blk_pos[name + "_down"] = (bx + .22, Y + Hs)
            bx = e + .15
        slab(ax, bx, Y, .5, Hs, C_CONV, "ConvNeXt Block")
        blk_pos[name] = (bx + .25, Y + Hs)
        sx += w
        arr(ax, sx + .02, YC, sx + .28, YC)
        sx += .3

    # feature map
    xe = cube(ax, sx, YC - .5, 1.0, 7, "#c9efbd", GREEN_D)
    t(ax, sx + .5, YC + .95, L("Feature Map\n7×7×768", "Feature Map\n7×7×C₄"), fs=9.5, w="bold")
    arr(ax, xe + .05, YC, xe + .32, YC)
    fx = xe + .35
    xe = vstack(ax, fx, YC, 6, C_TOK_C)
    t(ax, fx + .1, YC + 1.3, L("Flatten\n49×768", "Flatten\n49×C₄"), fs=9, w="bold")
    arr(ax, xe + .03, YC, xe + .3, YC)

    # S-Adapter
    ax0 = xe + .3
    blocks = [(C_DOWN, "Reshape 7×7"), (C_LIN, "Conv 1×1 (768→64) + BN + GELU"),
              (C_ADP, "DWConv 3×3 + BN + GELU"), (C_LIN, "Conv 1×1 (64→768) + BN"),
              (C_DOWN, "Flatten"), ("+", None),
              (C_LIN, "Linear (768→384)"), (C_NORM, "LayerNorm")]
    dims = ["7×7×768", "7×7×64", "7×7×64", "7×7×768", "49×768", "", "49×384", ""]
    if general:
        blocks = [(C_DOWN, "Reshape 7×7"), (C_LIN, "Conv 1×1 (C₄→r) + BN + GELU"),
                  (C_ADP, "DWConv 3×3 + BN + GELU"), (C_LIN, "Conv 1×1 (r→C₄) + BN"),
                  (C_DOWN, "Flatten"), ("+", None),
                  (C_LIN, "Linear (C₄→d)"), (C_NORM, "LayerNorm")]
        dims = ["7×7×C₄", "7×7×r", "7×7×r", "7×7×C₄", "49×C₄", "", "49×d", ""]
    xe, pos = chain(ax, ax0 + .2, Y + .1, Hs - .2, blocks, w=.45, gap=.3, yc=YC, dims=dims,
                    dim_y=Y + Hs + .25)
    frame(ax, ax0, 1.2, xe - ax0 + .25, 3.85, RED_D, "S-Adapter (ConvNeXt Tokens)  — TSR", lw=2.2,
          tfs=9.5)
    skip(ax, ax0 + .1, pos[5][0], YC, 1.45)
    t(ax, pos[5][0] + .32, YC - .3, "×α", fs=9, w="bold", c=RED_D)
    ax_end = xe + .25
    arr(ax, ax_end + .02, YC, ax_end + .28, YC)
    tx = ax_end + .3
    xe = vstack(ax, tx, YC, 6, C_TOK_C)
    t(ax, tx + .1, YC + 1.38, L("$T_c$\n49×384", "$T_c$\n49×d"), fs=10, w="bold")
    arr(ax, xe + .03, YC, xe + .3, YC)
    to_box(ax, xe + .32, YC, 2.4, "#f1e4fb", "Fusion Module")

    # ───── row 2: ConvNeXt block detail ─────
    bw_ = 12.6 if general else 11.5
    cg = .7 if general else .55
    panel(ax, 0.2, 5.55, bw_, 4.25, "#fafdf9", GREEN_D, "ConvNeXt Block")
    Y2, H2 = 6.55, 2.3
    YC2 = Y2 + H2 / 2
    t(ax, .75, YC2, "$x$\nH×W×C", fs=10)
    arr(ax, 1.15, YC2, 1.55, YC2)
    blocks = [(C_CONV, "Depthwise Conv 7×7"), (C_NORM, "LayerNorm"), (C_LIN, "Conv 1×1 (C→4C)"),
              (C_ACT, "GELU"), (C_LIN, "Conv 1×1 (4C→C)"), (C_DOWN, "Layer Scale γ"),
              (C_NORM, "Drop Path"), ("+", None)]
    dims = ["H×W×C", "", "H×W×4C", "", "H×W×C", "", "", ""]
    xe, pos = chain(ax, 1.6, Y2, H2, blocks, w=.5, gap=cg, dims=dims, dim_y=Y2 + H2 + .3)
    skip(ax, 1.35, pos[-1][0], YC2, 6.2)
    arr(ax, xe + .02, YC2, xe + .5, YC2)
    t(ax, xe + .95, YC2, "$y$\nH×W×C", fs=10)
    # zoom line dari Stage 1 block
    zx, zy = blk_pos["Stage 1"]
    ln(ax, [zx, 0.2 + bw_ / 2], [5.05, 5.55], c=GREEN_D, lw=1.1, ls=(0, (3, 3)))

    # downsample detail
    dx = 2.5 if general else 0
    if general:
        panel(ax, 13.05, 5.55, 6.75, 4.25, "#fffdf3", "#b08a00", "Downsampling")
    else:
        panel(ax, 11.95, 5.55, 3.75, 4.25, "#fffdf3", "#b08a00", "Downsampling")
    t(ax, 12.5 + dx, YC2, "H×W×C", fs=9)
    arr(ax, 12.85 + dx, YC2, 13.05 + dx, YC2, ms=10)
    xe, _ = chain(ax, 13.1 + dx, Y2, H2, [(C_NORM, "LayerNorm"), (C_DOWN, "Conv 2×2, s=2")], w=.5,
                  gap=.4)
    arr(ax, xe + .02, YC2, xe + .22, YC2, ms=10)
    t(ax, xe + .27, YC2 - .3, "H/2×W/2", fs=9, ha="left")
    t(ax, xe + .27, YC2 + .05, "×2C", fs=9, ha="left")

    if general:
        save(fig, "07_ConvNeXt_Branch_General.png")
        return
    # legend
    panel(ax, 15.95, 5.55, 3.85, 4.25, "#ffffff", "#444444", "Keterangan")
    legend(ax, 16.25, 6.3, [
        (C_CONV, "Konvolusi (ConvNeXt)", "box"),
        (C_LIN, "Conv 1×1 / Linear", "box"),
        (C_DOWN, "Downsample / Reshape", "box"),
        (C_NORM, "Normalisasi", "box"),
        (C_ACT, "Aktivasi", "box"),
        (C_ADP, "Depthwise (S-Adapter)", "box"),
        (GREEN_D, "Frozen", "line"),
        (ORANGE_D, "Trainable (fine-tune)", "line"),
        (None, "Residual add", "plus")], step=.39)
    save(fig, "04_ConvNeXt_Branch_Landscape.png")


# ═══════════════════════ 2. ViT ═══════════════════════
def vit(general=False):
    L = lambda a, b: b if general else a
    fig, ax = canvas()
    t(ax, W / 2, 0.42, L("Vision Transformer Branch (Global Feature Extraction) — ViT-Small/16 + S-Adapter", "Vision Transformer Branch (Global Feature Extraction) + S-Adapter"),
      fs=17, w="bold", c=BLUE_D)
    panel(ax, 0.2, 0.8, 19.6, 4.45, "#eef5fd", BLUE_D, "")
    Y, Hs = 1.9, 2.3
    YC = Y + Hs / 2

    face(ax, .45, YC - .6, 1.2)
    t(ax, 1.05, YC + .95, "Input\n224×224×3", fs=9.5, w="bold")
    arr(ax, 1.72, YC, 2.05, YC)
    face(ax, 2.1, YC - .6, 1.2, grid_n=14)
    t(ax, 2.7, YC + .95, "Patch 16×16\n14×14 = 196", fs=9.5, w="bold")
    arr(ax, 3.37, YC, 3.65, YC)
    xe = vstack(ax, 3.7, YC, 6, "#eeeeee")
    t(ax, 3.8, YC + 1.38, L("196×768", "196×(16·16·3)"), fs=9, w="bold")
    arr(ax, xe + .03, YC, xe + .3, YC)
    xe = slab(ax, xe + .32, Y, .5, Hs, C_LIN, "Linear Projection (Conv 16×16, s=16)", fs=8.5)
    t(ax, xe - .3, Y - .35, L("196×384", "196×d"), fs=8.5, c=MUTED, st="italic")
    arr(ax, xe + .02, YC, xe + .3, YC)
    cx = xe + .32
    xe = vstack(ax, cx, YC, 6, C_TOK_V, first=C_CLS)
    t(ax, cx + .1, YC + 1.38, "[CLS] +\npatch", fs=9, w="bold")
    arr(ax, xe + .03, YC, xe + .3, YC)
    px = xe + .5
    plus(ax, px, YC)
    # positional embedding
    ax.add_patch(FancyBboxPatch((px - .55, 1.05), 1.1, .42, boxstyle="round,pad=0,rounding_size=.08",
                                fc=C_POS, ec="#333", lw=.9, zorder=5))
    t(ax, px, 1.26, "Pos. Embed", fs=8.5)
    arr(ax, px, 1.47, px, YC - .17, ms=10)
    arr(ax, px + .17, YC, px + .7, YC)
    t(ax, px, YC + .45, L("197×384", "197×d"), fs=8, c=MUTED, st="italic", rot=0)

    # encoder
    ex0 = px + .75
    blocks = [(C_NORM, "LayerNorm"), (C_ATT, L("Multi-Head Self-Attention (h = 6)", "Multi-Head Self-Attention (h heads)")), ("+", None),
              (C_NORM, "LayerNorm"), (C_LIN, L("MLP (384→1536→384)", "MLP (d→4d→d)")), ("+", None),
              (C_ADP, L("S-Adapter (384→64→384)", "S-Adapter (d→r→d)")), ("+", None)]
    xe, pos = chain(ax, ex0 + .4, Y, Hs, blocks, w=.55, gap=.58)
    frame(ax, ex0, 1.2, xe - ex0 + .3, 3.85, BLUE_D, L("Transformer Encoder Block + S-Adapter   × 12", "Transformer Encoder Block + S-Adapter   × L"),
          lw=2.0, tfs=10)
    skip(ax, ex0 + .2, pos[2][0], YC, 1.5)
    skip(ax, pos[2][0] + .25, pos[5][0], YC, 1.5)
    skip(ax, pos[5][0] + .25, pos[7][0], YC, 1.5)
    ee = xe + .3
    arr(ax, ee + .02, YC, ee + .3, YC)
    xe = slab(ax, ee + .32, Y, .5, Hs, C_NORM, "LayerNorm")
    arr(ax, xe + .02, YC, xe + .3, YC)
    rx = xe + .32
    xe = vstack(ax, rx, YC, 6, C_TOK_V, first=C_CLS, cross_first=True)
    t(ax, rx + .1, YC + 1.38, "Remove\n[CLS]", fs=9, w="bold")
    arr(ax, xe + .03, YC, xe + .6, YC)
    tx = xe + .62
    xe = vstack(ax, tx, YC, 6, C_TOK_V)
    t(ax, tx + .1, YC + 1.38, L("$T_v$\n196×384", "$T_v$\n196×d"), fs=10, w="bold")
    arr(ax, xe + .03, YC, xe + .3, YC)
    to_box(ax, xe + .32, YC, 2.4, "#f1e4fb", "Fusion Module")
    mhsa_x, mlp_x, adp_x = pos[1][0], pos[4][0], pos[6][0]

    # ───── row 2 ─────
    Y2, H2 = 6.65, 2.1
    YC2 = Y2 + H2 / 2
    # MHSA
    panel(ax, 0.2, 5.55, 9.3, 4.25, "#fffcf0", "#b08a00", L("Multi-Head Self-Attention (h = 6, $d_h$ = 64)", "Multi-Head Self-Attention (h heads, $d_h$ = d / h)"))
    t(ax, .6, YC2, "$X$", fs=11)
    for i, (lab, yy) in enumerate([("Linear Q", Y2 + .05), ("Linear K", YC2 - .3), ("Linear V", Y2 + H2 - .65)]):
        ax.add_patch(FancyBboxPatch((1.15, yy), 1.0, .55, boxstyle="round,pad=0,rounding_size=.06",
                                    fc=C_LIN, ec="#333", lw=.9, zorder=5))
        t(ax, 1.65, yy + .275, lab, fs=8.5)
        arr(ax, .8, YC2, 1.13, yy + .275, ms=8, lw=1)
    qy, ky, vy = Y2 + .33, YC2 - .02, Y2 + H2 - .37
    x1 = 2.5
    e, p2 = chain(ax, x1, Y2, H2, [(C_ATT, "MatMul $QK^T$"), (C_NORM, L("Scale ÷√64", r"Scale ÷ $\sqrt{d_h}$")),
                                   (C_ACT, "Softmax"), (C_ATT, "MatMul ·V"), (C_DOWN, "Concat heads"),
                                   (C_LIN, "Linear $W_o$")], w=.5, gap=.45,
                  dims=["", "", L("197×197", "197×197 (×h)"), "", L("197×384", "197×d"), ""], dim_y=Y2 + H2 + .3)
    arr(ax, 2.17, qy, x1 - .02, qy, ms=8, lw=1)
    arr(ax, 2.17, ky, x1 - .02, ky, ms=8, lw=1)
    vx = p2[3][0]
    ln(ax, [2.17, 2.3, 2.3], [vy, vy, Y2 + H2 + .5], lw=1)
    ln(ax, [2.3, vx], [Y2 + H2 + .5, Y2 + H2 + .5], lw=1)
    arr(ax, vx, Y2 + H2 + .5, vx, Y2 + H2 + .02, ms=8, lw=1)
    t(ax, (2.3 + vx) / 2, Y2 + H2 + .65, "V", fs=8.5, c=MUTED)
    arr(ax, e + .02, YC2, e + .3, YC2, ms=10)
    t(ax, e + .55, YC2, "out", fs=9)
    # MLP
    panel(ax, 9.7, 5.55, 4.4, 4.25, "#f4f9ff", BLUE_D, "MLP (Feed Forward)")
    e, _ = chain(ax, 10.15, Y2, H2, [(C_LIN, L("Linear 384→1536", "Linear d→4d")), (C_ACT, "GELU"),
                                     (C_LIN, L("Linear 1536→384", "Linear 4d→d"))], w=.55, gap=.55,
                 dims=["", "", ""], dim_y=0)
    t(ax, 12.0, Y2 + H2 + .4, L("197×384 → 197×1536 → 197×384", "197×d → 197×4d → 197×d"), fs=8.5, c=MUTED, st="italic")
    # S-Adapter
    panel(ax, 14.3, 5.55, 5.5, 4.25, "#fff6f6", RED_D, "S-Adapter")
    t(ax, 14.75, YC2, "$x$", fs=11)
    arr(ax, 14.9, YC2, 15.2, YC2, ms=10)
    e, p3 = chain(ax, 15.25, Y2, H2, [(C_LIN, L("Linear 384→64", "Linear d→r")), (C_ACT, "GELU"),
                                      (C_LIN, L("Linear 64→384", "Linear r→d")), (C_DOWN, "Scale × s"), ("+", None)],
                  w=.45, gap=.35, dims=["", "", "", "", ""], dim_y=0)
    skip(ax, 15.05, p3[-1][0], YC2, 6.3)
    arr(ax, e + .02, YC2, e + .3, YC2, ms=10)
    t(ax, e + .5, YC2, "$x'$", fs=11)
    t(ax, 17.05, Y2 + H2 + .4, L("bottleneck r = 64,  s₀ = 0.1", "bottleneck r,  scale s (learnable)"), fs=8.5, c=MUTED, st="italic")
    # zoom lines
    for sx_, tx_ in [(mhsa_x, 4.85), (mlp_x, 11.9), (adp_x, 17.05)]:
        ln(ax, [sx_, tx_], [5.05, 5.55], c=BLUE_D, lw=1.1, ls=(0, (3, 3)))
    save(fig, L("05_ViT_Branch_Landscape.png", "08_ViT_Branch_General.png"))


# ═══════════════════════ 3. Fusion ═══════════════════════
def fusion(general=False):
    L = lambda a, b: b if general else a
    fig, ax = canvas()
    t(ax, W / 2, 0.42, "Fusion & Classification Module — Bidirectional Cross-Attention + Token Refinement",
      fs=17, w="bold", c=PURPLE_D)
    panel(ax, 0.2, 0.8, 19.6, 4.9, "#f7f0fd", PURPLE_D, "")
    YA, YB, HL = 2.25, 4.2, 1.25          # pusat lane A / B, tinggi slab
    # inputs
    xe = vstack(ax, .5, YA, 4, C_TOK_C, s=.17, g=.04)
    t(ax, .6, 1.05, L("$T_c$  49×384", "$T_c$  49×d"), fs=9.5, w="bold", ha="left")
    vstack(ax, .5, YB, 4, C_TOK_V, s=.17, g=.04)
    t(ax, .6, 5.35, L("$T_v$  196×384", "$T_v$  196×d"), fs=9.5, w="bold", ha="left")
    for yc, lab, other in [(YA, "Cross-Attention L→G\nQ = $T_c$,  K = V = $T_v$", YB),
                           (YB, "Cross-Attention G→L\nQ = $T_v$,  K = V = $T_c$", YA)]:
        arr(ax, .75, yc, 1.25, yc)
        slab(ax, 1.3, yc - HL / 2, .45, HL, C_NORM, "LN", fs=8.5)
        arr(ax, 1.87, yc, 2.7, yc)
        ax.add_patch(FancyBboxPatch((2.75, yc - .55), 1.8, 1.1, boxstyle="round,pad=0,rounding_size=.1",
                                    fc=C_ATT, ec="#333", lw=1, zorder=5))
        t(ax, 3.65, yc, lab, fs=8.5)
        arr(ax, 4.57, yc, 4.82, yc)
        plus(ax, 4.99, yc)
        arr(ax, 5.16, yc, 5.35, yc)
        slab(ax, 5.4, yc - HL / 2, .45, HL, C_NORM, "LN", fs=8.5)
        arr(ax, 5.97, yc, 6.2, yc)
        # K,V dari lane lain
        sgn = 1 if other > yc else -1
        ax.add_patch(Circle((2.25, other), .04, fc=TXT, zorder=7))
        arr(ax, 2.25, other, 3.2, yc + sgn * .55, ms=10, c=PURPLE_D)
        # residual
        sk = yc - .95 if yc == YA else yc + .95
        ax.add_patch(Circle((1.0, yc), .04, fc=TXT, zorder=7))
        ln(ax, [1.0, 1.0, 4.99], [yc, sk, sk])
        arr(ax, 4.99, sk, 4.99, yc - .17 if sk < yc else yc + .17, ms=10)
    t(ax, 2.2, 3.22, "K, V", fs=8.5, c=PURPLE_D, w="bold")
    vstack(ax, 6.25, YA, 4, C_TOK_C, s=.17, g=.04)
    vstack(ax, 6.25, YB, 4, C_TOK_V, s=.17, g=.04)
    t(ax, 6.33, 1.05, "$T'_c$", fs=10, w="bold")
    t(ax, 6.33, 5.35, "$T'_v$", fs=10, w="bold")
    # concat
    arr(ax, 6.47, YA, 6.75, YA)
    arr(ax, 6.47, YB, 6.75, YB)
    slab(ax, 6.77, YA - .55, .5, YB - YA + 1.1, C_DOWN, "Concat  $[T'_c\\,;\\,T'_v]$")
    YC = (YA + YB) / 2
    arr(ax, 7.39, YC, 7.65, YC)
    xe = vstack(ax, 7.7, YC, 7, "#cfe3c8", s=.18, g=.04)
    t(ax, 7.79, YC + 1.35, L("$T_r$\n245×384", "$T_r$\n245×d"), fs=9.5, w="bold")
    arr(ax, xe + .03, YC, xe + .3, YC)
    # refinement
    r0 = xe + .32
    Y, Hs = YC - 1.05, 2.1
    blocks = [(C_NORM, "LayerNorm"), (C_ATT, L("MHSA (h = 6)", "MHSA (h heads)")), ("+", None),
              (C_NORM, "LayerNorm"), (C_LIN, L("MLP 384→1536→384", "MLP d→4d→d")), ("+", None)]
    xe, pos = chain(ax, r0 + .35, Y, Hs, blocks, w=.5, gap=.32, yc=YC)
    frame(ax, r0, 1.25, xe - r0 + .25, 4.25, PURPLE_D, "Refinement Transformer Block", lw=1.8, tfs=9.5)
    skip(ax, r0 + .18, pos[2][0], YC, 1.65)
    skip(ax, pos[2][0] + .25, pos[5][0], YC, 1.65)
    ee = xe + .25
    arr(ax, ee + .02, YC, ee + .28, YC)
    xe = slab(ax, ee + .3, Y, .5, Hs, C_MISC, "Global Avg Pooling")
    t(ax, xe - .3, Y + Hs + .3, L("384", "d"), fs=8.5, c=MUTED, st="italic")
    arr(ax, xe + .02, YC, xe + .3, YC)
    c0 = xe + .3
    blocks = [(C_LIN, L("Linear 384→256", "Linear d→256")), (C_ACT, "ReLU"), (C_NORM, "Dropout 0.3"),
              (C_LIN, "Linear 256→2"), (C_MISC, "Softmax")]
    xe, _ = chain(ax, c0 + .2, Y, Hs, blocks, w=.42, gap=.26, yc=YC)
    frame(ax, c0, 1.25, xe - c0 + .2, 4.25, BLUE_D, "MLP Classifier", lw=1.8, tfs=9.5)
    oe = xe + .2
    ax.add_patch(Circle((oe + .55, YC - .55), .25, fc="#7ed957", ec="#333", lw=.9, zorder=5))
    t(ax, oe + .55, YC - .1, "Real", fs=9.5, w="bold")
    ax.add_patch(Circle((oe + .55, YC + .5), .25, fc="#ff3b30", ec="#333", lw=.9, zorder=5))
    t(ax, oe + .55, YC + .95, "Spoof", fs=9.5, w="bold")
    arr(ax, oe + .02, YC, oe + .28, YC - .45, ms=9)
    arr(ax, oe + .02, YC, oe + .28, YC + .4, ms=9)

    # ───── row 2: detail cross-attention ─────
    panel(ax, 0.2, 5.95, L(13.6, 19.6), 3.85, "#fffcf0", "#b08a00",
          L("Detail Cross-Attention (contoh Local→Global, h = 6, $d_h$ = 64)",
            "Cross-Attention Detail (Local→Global example, h heads, $d_h$ = d / h)"))
    Y2, H2 = 6.95, 1.95
    YC2 = Y2 + H2 / 2
    t(ax, .75, Y2 + .3, "$T_c$", fs=11)
    t(ax, .75, Y2 + H2 - .3, "$T_v$", fs=11)
    for lab, yy, src in [("Linear Q", Y2 + .05, Y2 + .3), ("Linear K", YC2 - .27, Y2 + H2 - .3),
                         ("Linear V", Y2 + H2 - .6, Y2 + H2 - .3)]:
        ax.add_patch(FancyBboxPatch((1.25, yy), 1.0, .5, boxstyle="round,pad=0,rounding_size=.06",
                                    fc=C_LIN, ec="#333", lw=.9, zorder=5))
        t(ax, 1.75, yy + .25, lab, fs=8.5)
        arr(ax, .98, src, 1.23, yy + .25, ms=8, lw=1)
    x1 = 2.65
    e, p2 = chain(ax, x1, Y2, H2, [(C_ATT, "MatMul $QK^T$"), (C_NORM, L("Scale ÷√64", r"Scale ÷ $\sqrt{d_h}$")),
                                   (C_ACT, "Softmax"), (C_ATT, "MatMul ·V"), (C_DOWN, "Concat heads"),
                                   (C_LIN, "Linear $W_o$"), (C_NORM, "Dropout"), ("+", None),
                                   (C_NORM, "LayerNorm")], w=.45, gap=L(.38, 1.05),
                  dims=["", "", L("49×196", "49×196 (×h)"), "", "", "", "", "", L("", "49×d")], dim_y=Y2 + H2 + .3)
    arr(ax, 2.27, Y2 + .3, x1 - .02, Y2 + .3, ms=8, lw=1)
    arr(ax, 2.27, YC2, x1 - .02, YC2, ms=8, lw=1)
    vx = p2[3][0]
    ln(ax, [2.27, 2.45, 2.45, vx], [Y2 + H2 - .35, Y2 + H2 - .35, Y2 + H2 + .55, Y2 + H2 + .55], lw=1)
    arr(ax, vx, Y2 + H2 + .55, vx, Y2 + H2 + .02, ms=8, lw=1)
    t(ax, (2.45 + vx) / 2, Y2 + H2 + .7, "V", fs=8.5, c=MUTED)
    px_ = p2[7][0]
    ax.add_patch(Circle((1.08, Y2 + .3), .035, fc=TXT, zorder=7))
    ln(ax, [1.08, 1.08, px_], [Y2 + .3, 6.55, 6.55], lw=1)
    arr(ax, px_, 6.55, px_, YC2 - .17, ms=8, lw=1)
    arr(ax, e + .02, YC2, e + .3, YC2, ms=10)
    t(ax, e + .55, YC2, "$T'_c$", fs=11)

    if general:
        save(fig, "09_Fusion_Module_General.png")
        return
    panel(ax, 14.0, 5.95, 5.8, 3.85, "#ffffff", "#444444", "Keterangan")
    legend(ax, 14.3, 6.65, [
        (C_ATT, "Attention", "box"), (C_LIN, "Linear", "box"), (C_NORM, "Norm / Dropout", "box"),
        (C_ACT, "Aktivasi / Softmax", "box"), (C_DOWN, "Concat", "box"),
        (C_TOK_C, "Token lokal (ConvNeXt)", "box")], step=.42)
    legend(ax, 17.1, 6.65, [(C_TOK_V, "Token global (ViT)", "box"), (C_MISC, "Pooling", "box"),
                            (None, "Residual add", "plus")], step=.42)
    save(fig, "06_Fusion_Module_Landscape.png")


if __name__ == "__main__":
    for g in (False, True):
        convnext(g)
        vit(g)
        fusion(g)
    FLAT = True
    convnext(True)
    vit(True)
    fusion(True)
