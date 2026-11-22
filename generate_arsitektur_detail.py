"""
Generate diagram arsitektur DETAIL per-branch (turunan dari assets/Arsitektur TA.png).

Output -> assets/Arsitektur_Detail/
  01_ConvNeXt_Branch_Detail.png
  02_ViT_Branch_Detail.png
  03_Fusion_Module_Detail.png

Seluruh angka dimensi & parameter diambil dari konfigurasi model H5 (Champion):
ConvNeXt-Small + ViT-Small, d = 384, adapter bottleneck r = 64, fusion heads = 6
(models/s_adapter_convnext_hybrid.py). Jumlah parameter sudah diverifikasi
dengan menginstansiasi model (pretrained=False) dan menghitung numel().
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle

OUT_DIR = Path(__file__).parent / "assets" / "Arsitektur_Detail"
OUT_DIR.mkdir(parents=True, exist_ok=True)

W, H = 13.0, 18.0
DPI = 250

# Palet (selaras dengan diagram utama)
GREEN_BG, GREEN, GREEN_D = "#e4f2e1", "#9fdc8c", "#2e7d32"
BLUE_BG, BLUE, BLUE_D = "#e3effc", "#bcd9f5", "#1f4e8c"
PURPLE_BG, PURPLE, PURPLE_D = "#f1e4fb", "#d9b3f2", "#5b2a86"
RED_BG, RED, RED_D = "#fde2e2", "#f59a9a", "#c62828"
YELLOW = "#fff2cc"
GREY = "#d9d9d9"
NAVY = "#1a237e"
TXT = "#1b1b1b"
MUTED = "#555555"


# ───────────────────────── helpers ─────────────────────────
def new_canvas(bg="white"):
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # y ke bawah
    ax.axis("off")
    fig.patch.set_facecolor(bg)
    return fig, ax


def box(ax, x, y, w, h, fc="white", ec=TXT, lw=1.3, ls="-", r=0.12, z=2, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, ls=ls, zorder=z, alpha=alpha)
    ax.add_patch(p)
    return p


def txt(ax, x, y, s, size=10, weight="normal", color=TXT, ha="left", va="center",
        style="normal", family=None, z=5):
    kw = dict(fontsize=size, fontweight=weight, color=color, ha=ha, va=va,
              style=style, zorder=z)
    if family:
        kw["family"] = family
    ax.text(x, y, s, **kw)


def arrow(ax, x1, y1, x2, y2, color=TXT, lw=1.6, style="-|>", ms=14, z=4, ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                mutation_scale=ms, linestyle=ls,
                                shrinkA=0, shrinkB=0), zorder=z)


def line(ax, xs, ys, color=TXT, lw=1.4, ls="-", z=3):
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=z, solid_capstyle="round")


def tokens(ax, x, y, n, color, w=0.13, h=0.42, gap=0.04, first=None, dots=True, ec="#333"):
    """Strip token; first=warna khusus token pertama (mis. CLS)."""
    cx = x
    for i in range(n):
        c = first if (first and i == 0) else color
        ax.add_patch(FancyBboxPatch((cx, y), w, h, boxstyle="round,pad=0,rounding_size=0.03",
                                    fc=c, ec=ec, lw=0.8, zorder=5))
        cx += w + gap
    if dots:
        txt(ax, cx + 0.03, y + h - 0.05, "…", size=9, ha="left", va="bottom")
        cx += 0.22
        ax.add_patch(FancyBboxPatch((cx, y), w, h, boxstyle="round,pad=0,rounding_size=0.03",
                                    fc=color, ec=ec, lw=0.8, zorder=5))
        cx += w
    return cx


def grid(ax, x, y, s, n, fc, ec):
    ax.add_patch(Rectangle((x, y), s, s, fc=fc, ec=ec, lw=1.0, zorder=5))
    for i in range(1, n):
        line(ax, [x + s * i / n] * 2, [y, y + s], color=ec, lw=0.5, z=6)
        line(ax, [x, x + s], [y + s * i / n] * 2, color=ec, lw=0.5, z=6)


def badge(ax, x, y, kind):
    if kind == "frozen":
        s, fc, ec, c = "FROZEN", "#e8eef7", "#5c7cab", "#2c4a7a"
    else:
        s, fc, ec, c = "TRAINABLE", "#ffe3cc", "#e07b1f", "#a34d00"
    w = 0.12 + 0.085 * len(s)
    box(ax, x - w, y, w, 0.26, fc=fc, ec=ec, lw=1.0, r=0.08, z=6)
    txt(ax, x - w / 2, y + 0.135, s, size=7.2, weight="bold", color=c, ha="center", z=7)


def shape(ax, x, y, s, color=NAVY, size=10.5):
    txt(ax, x, y, s, size=size, style="italic", color=color, ha="right", weight="bold")


def plus(ax, x, y, r=0.15, ec=TXT):
    ax.add_patch(Circle((x, y), r, fc="white", ec=ec, lw=1.3, zorder=6))
    line(ax, [x - r * 0.6, x + r * 0.6], [y, y], color=ec, lw=1.3, z=7)
    line(ax, [x, x], [y - r * 0.6, y + r * 0.6], color=ec, lw=1.3, z=7)


def panel(ax, x, y, w, h, title, fc, ec, tcolor):
    box(ax, x, y, w, h, fc=fc, ec=ec, lw=1.4, r=0.15, z=1)
    box(ax, x, y, w, 0.42, fc=ec, ec=ec, lw=1.4, r=0.15, z=1.5)
    ax.add_patch(Rectangle((x, y + 0.25), w, 0.17, fc=ec, ec="none", zorder=1.6))
    txt(ax, x + 0.15, y + 0.21, title, size=10.5, weight="bold", color="white", z=7)


def table(ax, x, y, rows, col_x, size=8.6, row_h=0.27, header=True, bold_rows=(), color=TXT,
          hl_rows=(), hl_color="#fff3b0", width=None):
    for i, row in enumerate(rows):
        yy = y + i * row_h
        if i in hl_rows and width:
            ax.add_patch(Rectangle((x - 0.05, yy - row_h / 2), width, row_h, fc=hl_color,
                                   ec="none", zorder=3))
        for j, cell in enumerate(row):
            wgt = "bold" if (header and i == 0) or i in bold_rows else "normal"
            ha = "left"
            cx = x + col_x[j]
            if isinstance(cell, tuple):
                cell, ha = cell
            txt(ax, cx, yy, cell, size=size, weight=wgt, color=color, ha=ha)
        if header and i == 0 and width:
            line(ax, [x - 0.05, x - 0.05 + width], [yy + row_h / 2] * 2, color=MUTED, lw=0.8)


def header(ax, title, subtitle, color):
    txt(ax, W / 2, 0.48, title, size=19, weight="bold", color=color, ha="center")
    txt(ax, W / 2, 0.92, subtitle, size=10.5, color=MUTED, ha="center")


def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Saved:", path)


# ═══════════════════════ 1. CONVNEXT BRANCH ═══════════════════════
def draw_convnext():
    fig, ax = new_canvas()
    header(ax, "ConvNeXt Branch — Detail Alur & Perhitungan (Local Feature Extraction)",
           "Konfigurasi H5 (Champion): ConvNeXt-Small  |  d = 384 (mengikuti ViT-Small)  |  "
           "bottleneck r = 64  |  B = batch size", GREEN_D)

    # background kolom utama
    box(ax, 0.3, 1.25, 7.0, 16.5, fc=GREEN_BG, ec=GREEN_D, lw=1.2, r=0.25, z=0)
    X, BW, CX = 0.55, 6.5, 0.55 + 6.5 / 2
    R = X + BW - 0.15  # tepi kanan utk shape

    def step(y, h, title, lines, shp, fc="white", ec=GREEN_D, kind=None, params=None,
             lsize=8.8):
        box(ax, X, y, BW, h, fc=fc, ec=ec, lw=1.4)
        txt(ax, X + 0.15, y + 0.25, title, size=10.5, weight="bold")
        for i, l in enumerate(lines):
            txt(ax, X + 0.15, y + 0.56 + i * 0.27, l, size=lsize, color=MUTED)
        if shp:
            shape(ax, R, y + h - 0.25, shp)
        if kind:
            badge(ax, R, y + 0.1, kind)
        if params:
            txt(ax, R, y + 0.56, params, size=8, color=MUTED, ha="right", style="italic")

    def down(y1, y2):
        arrow(ax, CX, y1, CX, y2, color=GREEN_D)

    # 1. Input
    step(1.5, 0.95, "Input Image", ["RGB → Resize 224×224 → Normalize (mean/std ImageNet)"],
         "(B, 3, 224, 224)")
    down(2.45, 2.7)
    # 2. Stem
    step(2.7, 1.2, "STEM — Conv2d(3→96, k=4, s=4) + LayerNorm2d",
         [r"Patchify non-overlap 4×4:  $H_{out}=\frac{224-4}{4}+1=56$",
          "48 bobot/filter × 96 filter + bias + LN"],
         "(B, 96, 56, 56)", fc=GREEN, kind="frozen", params="4,896 param")
    down(3.9, 4.15)
    # 3. Stage 1
    step(4.15, 1.0, "Stage 1 — 3 × ConvNeXt Block (C₁ = 96)",
         ["Tanpa downsample, resolusi tetap 56×56"],
         "(B, 96, 56, 56)", fc=GREEN, kind="frozen", params="237,888 param")
    down(5.15, 5.4)
    # 4. Stage 2
    step(5.4, 1.2, "Stage 2 — Downsample + 3 × ConvNeXt Block (C₂ = 192)",
         [r"LN2d + Conv2d(96→192, k=2, s=2):  $\frac{56-2}{2}+1=28$"],
         "(B, 192, 28, 28)", fc=GREEN, kind="frozen", params="992,256 param")
    down(6.6, 6.85)
    # 5. Stage 3
    step(6.85, 1.2, "Stage 3 — Downsample + 27 × ConvNeXt Block (C₃ = 384)",
         [r"LN2d + Conv2d(192→384, k=2, s=2):  $\frac{28-2}{2}+1=14$"],
         "(B, 384, 14, 14)", fc=GREEN, kind="frozen", params="32,747,520 param")
    down(8.05, 8.3)
    # 6. Stage 4
    step(8.3, 1.2, "Stage 4 — Downsample + 3 × ConvNeXt Block (C₄ = 768)",
         [r"LN2d + Conv2d(384→768, k=2, s=2):  $\frac{14-2}{2}+1=7$",
          "Di-unfreeze → fine-tuning domain anti-spoofing"],
         "(B, 768, 7, 7)", fc="#7fcf6b", kind="trainable", params="15,470,592 param")
    down(9.5, 9.75)
    # 7. Feature map
    step(9.75, 1.15, "Feature Map dari Stage 4", ["Peta fitur lokal 7×7, kedalaman C₄ = 768",
                                                   "(norm_pre = Identity)"],
         "(B, 768, 7, 7)")
    grid(ax, 4.55, 9.92, 0.8, 7, "#c9efbd", GREEN_D)
    txt(ax, 4.95, 10.83, "7 × 7", size=7.5, ha="center", color=MUTED)
    down(10.9, 11.15)
    # 8. Tokenization
    step(11.15, 1.05, "Tokenization (Spatial Flatten)",
         ["flatten(2) → permute(0, 2, 1)", r"$N = 7 \times 7 = 49$ token, tiap token ∈ ℝ$^{768}$"],
         "(B, 49, 768)")
    tokens(ax, 3.9, 11.32, 6, GREEN, h=0.36)
    down(12.2, 12.45)

    # 9. S-Adapter group
    gy, gh = 12.45, 4.1
    box(ax, X, gy, BW, gh, fc=RED_BG, ec=RED_D, lw=1.8, ls=(0, (5, 3)))
    txt(ax, X + 0.15, gy + 0.27, "S-Adapter (Adapted for ConvNeXt Tokens)", size=11,
        weight="bold", color=RED_D)
    badge(ax, R, gy + 0.12, "trainable")
    txt(ax, R - 1.55, gy + 0.25, "396,737 param", size=8, color=MUTED, ha="right", style="italic")
    sx, sw = X + 0.2, BW - 0.4
    sr = sx + sw - 0.12
    # ① restore
    box(ax, sx, gy + 0.55, sw, 0.5, fc=YELLOW, ec="#b08a00", lw=1.0)
    txt(ax, sx + 0.12, gy + 0.8, "① Restore 2D:  permute(0,2,1) → reshape(B, 768, 7, 7)", size=8.8)
    shape(ax, sr, gy + 0.8, "(B, 768, 7, 7)", size=9)
    arrow(ax, CX, gy + 1.05, CX, gy + 1.2, color=RED_D, ms=10)
    # ② TSR
    ty = gy + 1.2
    box(ax, sx, ty, sw, 1.5, fc=YELLOW, ec="#b08a00", lw=1.0)
    txt(ax, sx + 0.12, ty + 0.2, "② Token Spatial Refinement (TSR) — local statistical processing",
        size=8.8, weight="bold")
    tsr = [("Conv1×1 (768→64) + BN + GELU", "(B, 64, 7, 7)"),
           ("DWConv3×3 (groups=64, pad=1) + BN + GELU", "(B, 64, 7, 7)"),
           ("Conv1×1 (64→768) + BN", "(B, 768, 7, 7)")]
    for i, (a, b) in enumerate(tsr):
        yy = ty + 0.52 + i * 0.32
        box(ax, sx + 0.12, yy - 0.13, 3.95, 0.26, fc="white", ec="#b08a00", lw=0.8, r=0.06)
        txt(ax, sx + 0.2, yy, a, size=8.2)
        shape(ax, sr, yy, b, size=8.6)
        if i < 2:
            arrow(ax, sx + 2.1, yy + 0.13, sx + 2.1, yy + 0.19, color="#b08a00", ms=7, lw=1)
    arrow(ax, CX, ty + 1.5, CX, ty + 1.65, color=RED_D, ms=10)
    # ③ residual
    ry = ty + 1.65
    box(ax, sx, ry, sw, 0.55, fc=YELLOW, ec="#b08a00", lw=1.0)
    txt(ax, sx + 0.12, ry + 0.28,
        r"③ Flatten + Residual:  $X = T + \alpha\cdot\mathrm{TSR}(T)$,  $\alpha_0=0.1$ (learnable)",
        size=8.8)
    shape(ax, sr, ry + 0.28, "(B, 49, 768)", size=9)
    arrow(ax, CX, ry + 0.55, CX, ry + 0.7, color=RED_D, ms=10)
    # ④ proj
    py = ry + 0.7
    box(ax, sx, py, sw, 0.5, fc=BLUE, ec=BLUE_D, lw=1.0)
    txt(ax, sx + 0.12, py + 0.25, "④ Projection: Linear(768→384) + LayerNorm(384)", size=8.8)
    shape(ax, sr, py + 0.25, "(B, 49, 384)", size=9)

    down(gy + gh, 16.8)
    # 10. Output Tc
    box(ax, X, 16.8, BW, 0.8, fc="white", ec=GREEN_D, lw=2.0)
    txt(ax, X + 0.15, 17.07, r"Local Tokens  $T_c$", size=12, weight="bold")
    txt(ax, X + 0.15, 17.36, "→ masuk ke Fusion Module", size=8.5, color=MUTED)
    tokens(ax, 2.85, 17.0, 7, GREEN, h=0.4)
    shape(ax, R, 17.2, "(B, 49, 384)", size=11.5)

    # ───────────── panel kanan ─────────────
    PX, PW = 7.6, 5.1
    # A. ConvNeXt block detail
    panel(ax, PX, 1.25, PW, 6.95, "A. Detail Satu ConvNeXt Block", "#f6fbf5", GREEN_D, "white")
    items = [("Input  x", "(B, C, H, W)", "white"),
             ("DWConv 7×7 (groups=C, pad=3)", "spasial", GREEN),
             ("LayerNorm (channels-last)", "", GREY),
             ("Linear C → 4C (pointwise)", "ekspansi", BLUE),
             ("GELU", "", "#ffe0b2"),
             ("Linear 4C → C (pointwise)", "proyeksi", BLUE),
             ("Layer Scale  γ  (init 1e-6)", "", YELLOW)]
    bx, bw = PX + 0.25, 3.85
    y0 = 1.9
    for i, (a, b, c) in enumerate(items):
        yy = y0 + i * 0.6
        box(ax, bx, yy, bw, 0.4, fc=c, ec=TXT, lw=0.9, r=0.07)
        txt(ax, bx + 0.12, yy + 0.2, a, size=8.6)
        if b:
            txt(ax, bx + bw - 0.1, yy + 0.2, b, size=7.6, color=MUTED, ha="right", style="italic")
        arrow(ax, bx + bw / 2, yy + 0.4, bx + bw / 2, yy + 0.6, ms=8, lw=1.1)
    py_ = y0 + len(items) * 0.6 + 0.15
    plus(ax, bx + bw / 2, py_)
    # residual line
    rx = bx + bw + 0.4
    line(ax, [bx + bw, rx, rx], [y0 + 0.2, y0 + 0.2, py_], lw=1.2)
    arrow(ax, rx, py_, bx + bw / 2 + 0.15, py_, ms=9, lw=1.2)
    txt(ax, rx + 0.08, (y0 + py_) / 2, "skip\nconnection", size=7.5, color=MUTED, ha="left")
    arrow(ax, bx + bw / 2, py_ + 0.15, bx + bw / 2, py_ + 0.4, ms=8, lw=1.1)
    txt(ax, bx + bw / 2, py_ + 0.55, "Output  y  (B, C, H, W)", size=8.6, ha="center", weight="bold")
    fy = py_ + 0.95
    txt(ax, PX + 0.2, fy,
        r"$y = x + \gamma \odot W_2\,\mathrm{GELU}(W_1\,\mathrm{LN}(\mathrm{DWConv}_{7\times7}(x)))$",
        size=9.2)
    txt(ax, PX + 0.2, fy + 0.35, r"Param/blok $= 8C^2 + 58C$", size=8.8)
    txt(ax, PX + 0.2, fy + 0.62, r"Contoh C=96: $8(96^2)+58(96)=79{,}296$ × 3 blok = 237,888",
        size=8.2, color=MUTED)

    # B. Varian
    panel(ax, PX, 8.4, PW, 2.65, "B. Varian ConvNeXt (Studi Ablasi)", "#f6fbf5", GREEN_D, "white")
    rows = [["Varian", "Depth", "Channels C₁–C₄", "C₄"],
            ["Tiny", "[3,3,9,3]", "[96,192,384,768]", "768"],
            ["Small ★", "[3,3,27,3]", "[96,192,384,768]", "768"],
            ["Base", "[3,3,27,3]", "[128,256,512,1024]", "1024"]]
    table(ax, PX + 0.2, 9.1, rows, [0, 0.9, 1.95, 4.2], size=8.4, row_h=0.32,
          hl_rows=(2,), width=PW - 0.3)
    txt(ax, PX + 0.2, 10.45, r"Rumus umum:  $H_{out} = \lfloor (H_{in} + 2p - k)/s \rfloor + 1$",
        size=8.6)
    txt(ax, PX + 0.2, 10.78, "★ dipakai H5. Output selalu 7×7 = 49 token → Linear ke d.",
        size=8, color=MUTED)

    # C. Param S-Adapter
    panel(ax, PX, 11.25, PW, 4.0, "C. Rincian Parameter S-Adapter ConvNeXt", "#fff7f7", RED_D,
          "white")
    rows = [["Layer", "Perhitungan", ("Param", "right")],
            ["Conv1×1 768→64", "768×64", ("49,152", "right")],
            ["BatchNorm(64)", "2×64", ("128", "right")],
            ["DWConv3×3", "64×3×3", ("576", "right")],
            ["BatchNorm(64)", "2×64", ("128", "right")],
            ["Conv1×1 64→768", "64×768", ("49,152", "right")],
            ["BatchNorm(768)", "2×768", ("1,536", "right")],
            ["Skala α", "1", ("1", "right")],
            ["Linear 768→384", "768×384+384", ("295,296", "right")],
            ["LayerNorm(384)", "2×384", ("768", "right")],
            ["Total", "", ("396,737", "right")]]
    table(ax, PX + 0.2, 11.95, rows, [0, 2.0, 4.65], size=8.3, row_h=0.285, bold_rows=(10,),
          hl_rows=(10,), hl_color="#ffd6d6", width=PW - 0.3)
    txt(ax, PX + 0.2, 15.05, "Conv tanpa bias (bias=False) karena diikuti BatchNorm.", size=7.8,
        color=MUTED)

    # D. Ringkasan
    panel(ax, PX, 15.45, PW, 2.3, "D. Ringkasan Parameter Branch", "#f6fbf5", GREEN_D, "white")
    rows = [["Total parameter", ("49,849,889", "right")],
            ["Frozen (Stem + Stage 1–3)", ("33,982,560", "right")],
            ["Trainable (Stage 4 + S-Adapter)", ("15,867,329", "right")],
            ["Rasio trainable", ("31.83 %", "right")]]
    table(ax, PX + 0.2, 16.15, rows, [0, 4.65], size=8.8, row_h=0.32, header=False,
          bold_rows=(2,))
    fig.text(0.5, 0.004, "Sumber: models/s_adapter_convnext_hybrid.py — ConvNeXtTokenBranch & "
             "ConvNeXtTokenSAdapter", ha="center", fontsize=8, color=MUTED)
    save(fig, "01_ConvNeXt_Branch_Detail.png")


# ═══════════════════════ 2. ViT BRANCH ═══════════════════════
def draw_vit():
    fig, ax = new_canvas()
    header(ax, "Vision Transformer Branch — Detail Alur & Perhitungan (Global Feature Extraction)",
           "Konfigurasi H5 (Champion): ViT-Small/16  |  d = 384  |  h = 6 head  |  L = 12 blok  |  "
           "S-Adapter r = 64  |  B = batch size", BLUE_D)
    box(ax, 0.3, 1.25, 7.0, 16.5, fc=BLUE_BG, ec=BLUE_D, lw=1.2, r=0.25, z=0)
    X, BW, CX = 0.55, 6.5, 0.55 + 6.5 / 2
    R = X + BW - 0.15

    def step(y, h, title, lines, shp, fc="white", ec=BLUE_D, kind=None, params=None):
        box(ax, X, y, BW, h, fc=fc, ec=ec, lw=1.4)
        txt(ax, X + 0.15, y + 0.25, title, size=10.5, weight="bold")
        for i, l in enumerate(lines):
            txt(ax, X + 0.15, y + 0.58 + i * 0.29, l, size=8.8, color=MUTED)
        if shp:
            shape(ax, R, y + h - 0.25, shp)
        if kind:
            badge(ax, R, y + 0.1, kind)
        if params:
            txt(ax, R, y + 0.56, params, size=8, color=MUTED, ha="right", style="italic")

    def down(y1, y2, c=BLUE_D):
        arrow(ax, CX, y1, CX, y2, color=c)

    step(1.5, 0.95, "Input Image", ["RGB → Resize 224×224 → Normalize (mean/std ImageNet)"],
         "(B, 3, 224, 224)")
    down(2.45, 2.7)
    # Patch embedding
    step(2.7, 1.75, "Patch Embedding — Conv2d(3→384, k=16, s=16)",
         [r"$N = (224/16)^2 = 14 \times 14 = 196$ patch",
          r"Tiap patch $16\times16\times3 = 768$ nilai → $x_p E$,  $E \in \mathbb{R}^{768\times384}$",
          "Conv (B,384,14,14) → flatten(2) → transpose(1,2)"],
         "(B, 196, 384)", fc=BLUE, kind="frozen", params="295,296 param")
    grid(ax, 4.85, 3.12, 0.72, 14, "white", BLUE_D)
    down(4.45, 4.7)
    # CLS + pos
    step(4.7, 1.55, "Prepend [CLS] + Positional Embedding",
         [r"$z_0 = [x_{cls};\, x_p^1E;\, \dots;\, x_p^{196}E] + E_{pos}$",
          r"$x_{cls} \in \mathbb{R}^{1\times384}$,  $E_{pos} \in \mathbb{R}^{197\times384}$ (learnable)"],
         "(B, 197, 384)", fc="white", kind="frozen", params="76,032 param")
    tokens(ax, 0.75, 5.75, 7, BLUE, first=RED, h=0.3)
    txt(ax, 2.7, 5.9, "← [CLS] + 196 token", size=7.8, color=MUTED)
    down(6.25, 6.5)

    # Encoder group
    gy, gh = 6.5, 6.9
    box(ax, X, gy, BW, gh, fc="#f4f9ff", ec=BLUE_D, lw=1.8, ls=(0, (5, 3)))
    txt(ax, X + 0.15, gy + 0.27, r"Transformer Encoder Block $\ell$ + S-Adapter  ($\ell = 1 \dots 12$)",
        size=11, weight="bold", color=BLUE_D)
    bx, bw = X + 0.75, 4.3
    bcx = bx + bw / 2
    items = [("Input  $z_{\\ell-1}$", "(B, 197, 384)", "white", None),
             ("LayerNorm₁", "", GREY, None),
             ("Multi-Head Self-Attention (h=6, d_h=64)", "", "#fde9a8", None),
             ("⊕", "", None, "res1"),
             ("LayerNorm₂", "", GREY, None),
             ("MLP: Linear 384→1536 → GELU → Linear 1536→384", "", BLUE, None),
             ("⊕", "", None, "res2"),
             ("S-Adapter: Linear 384→64 → GELU → Linear 64→384, ×s", "", RED, None),
             ("⊕", "", None, "res3"),
             ("Output  $z_\\ell$", "(B, 197, 384)", "white", None)]
    y = gy + 0.6
    ys = {}
    for i, (a, b, c, tag) in enumerate(items):
        if tag:
            plus(ax, bcx, y + 0.15)
            ys[tag] = y + 0.15
            nxt = y + 0.3
        else:
            box(ax, bx, y, bw, 0.42, fc=c, ec=TXT, lw=0.9, r=0.07)
            txt(ax, bx + 0.12, y + 0.21, a, size=8.3)
            if b:
                txt(ax, bx + bw - 0.1, y + 0.21, b, size=8, color=NAVY, ha="right",
                    style="italic", weight="bold")
            ys[i] = y
            nxt = y + 0.42
        if i < len(items) - 1:
            arrow(ax, bcx, nxt, bcx, nxt + 0.18, ms=8, lw=1.1)
        y = nxt + 0.18
    # residual skip lines (kanan)
    def skip(y_from, y_to, xoff, label):
        rx = bx + bw + xoff
        line(ax, [bcx, rx], [y_from, y_from], lw=1.1, ls="-")
        line(ax, [rx, rx], [y_from, y_to], lw=1.1)
        arrow(ax, rx, y_to, bcx + 0.15, y_to, ms=8, lw=1.1)
        txt(ax, rx + 0.07, (y_from + y_to) / 2, label, size=7.6, color=MUTED)
    skip(ys[0] + 0.5, ys["res1"], 0.25, "skip")
    skip(ys["res1"] + 0.17, ys["res2"], 0.25, "skip")
    skip(ys["res2"] + 0.17, ys["res3"], 0.25, "skip")
    # loop ×12
    lx = X + 0.18
    line(ax, [bx, lx, lx, bx], [ys[9] + 0.21, ys[9] + 0.21, ys[0] + 0.21, ys[0] + 0.21],
         color=BLUE_D, lw=1.3, ls=(0, (3, 2)))
    arrow(ax, lx, ys[0] + 0.21, bx, ys[0] + 0.21, color=BLUE_D, ms=9, lw=1.3)
    txt(ax, lx + 0.03, (ys[0] + ys[9]) / 2, "×12", size=10, weight="bold", color=BLUE_D,
        ha="left", va="center")
    badge(ax, R, gy + 0.12, "frozen")
    txt(ax, R, gy + 0.55, "blok ViT: frozen", size=7.6, color=MUTED, ha="right")
    txt(ax, R, gy + 0.78, "S-Adapter: trainable", size=7.6, color=RED_D, ha="right")
    down(gy + gh, 13.65)

    step(13.65, 0.85, "Final LayerNorm (vit.norm)", ["Normalisasi seluruh 197 token  (768 param)"],
         "(B, 197, 384)", fc=GREY, kind="frozen")
    down(14.5, 14.75)
    step(14.75, 1.15, r"Remove [CLS] Token:  $T_v = z_{12}[:, 1{:}, :]$",
         ["Token CLS (indeks 0) dibuang;", "196 token patch dipertahankan"],
         "(B, 196, 384)")
    ex = tokens(ax, 4.6, 15.0, 1, RED, h=0.3, dots=False)
    line(ax, [4.57, 4.76], [14.98, 15.32], color=RED_D, lw=2, z=7)
    line(ax, [4.57, 4.76], [15.32, 14.98], color=RED_D, lw=2, z=7)
    tokens(ax, ex + 0.12, 15.0, 4, BLUE, h=0.3)
    down(15.9, 16.8)
    box(ax, X, 16.8, BW, 0.8, fc="white", ec=BLUE_D, lw=2.0)
    txt(ax, X + 0.15, 17.07, r"Global Tokens  $T_v$", size=12, weight="bold")
    txt(ax, X + 0.15, 17.36, "→ masuk ke Fusion Module", size=8.5, color=MUTED)
    tokens(ax, 2.95, 17.0, 7, BLUE, h=0.4)
    shape(ax, R, 17.2, "(B, 196, 384)", size=11.5)

    # ───────────── panel kanan ─────────────
    PX, PW = 7.6, 5.1
    panel(ax, PX, 1.25, PW, 5.6, "A. Perhitungan Multi-Head Self-Attention", "#f7fbff", BLUE_D,
          "white")
    L = [r"$X = \mathrm{LN}(z) \in \mathbb{R}^{B\times197\times384}$",
         r"$[Q,K,V] = X W_{qkv}$,  $W_{qkv} \in \mathbb{R}^{384\times1152}$",
         r"reshape → $Q,K,V \in \mathbb{R}^{B\times6\times197\times64}$",
         r"$d_h = d / h = 384 / 6 = 64$",
         r"$A = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{64}}\right) \in \mathbb{R}^{B\times6\times197\times197}$",
         r"$\mathrm{head}_i = A_i V_i \in \mathbb{R}^{B\times197\times64}$",
         r"$\mathrm{MHSA}(X) = [\mathrm{head}_1;\dots;\mathrm{head}_6]\,W_o$",
         r"$W_o \in \mathbb{R}^{384\times384}$  →  output (B, 197, 384)"]
    for i, l in enumerate(L):
        txt(ax, PX + 0.2, 1.95 + i * 0.5, l, size=9.6)
    txt(ax, PX + 0.2, 6.35, "Setiap token (termasuk CLS) melihat seluruh 197 token\n"
        "→ menangkap konteks global wajah.", size=8, color=MUTED)

    panel(ax, PX, 7.05, PW, 3.05, "B. Perhitungan S-Adapter (per blok)", "#fff7f7", RED_D, "white")
    txt(ax, PX + 0.2, 7.75,
        r"$\mathrm{SA}(x) = x + s\cdot W_{up}\,\mathrm{GELU}(W_{down}\,x)$", size=10)
    rows = [["Layer", "Perhitungan", ("Param", "right")],
            ["W_down 384→64", "384×64+64", ("24,640", "right")],
            ["W_up 64→384", "64×384+384", ("24,960", "right")],
            ["Skala s (init 0.1)", "1", ("1", "right")],
            ["Total / blok", "", ("49,601", "right")],
            ["× 12 blok", "", ("595,212", "right")]]
    table(ax, PX + 0.2, 8.2, rows, [0, 2.1, 4.65], size=8.4, row_h=0.3, bold_rows=(4, 5),
          hl_rows=(5,), hl_color="#ffd6d6", width=PW - 0.3)

    panel(ax, PX, 10.3, PW, 2.95, "C. Parameter 1 Transformer Block (frozen)", "#f7fbff", BLUE_D,
          "white")
    rows = [["LayerNorm₁ + LayerNorm₂", "2 × 768", ("1,536", "right")],
            ["W_qkv (384→1152)", "384×1152+1152", ("443,520", "right")],
            ["W_o (384→384)", "384×384+384", ("147,840", "right")],
            ["MLP fc1 (384→1536)", "384×1536+1536", ("591,360", "right")],
            ["MLP fc2 (1536→384)", "1536×384+384", ("590,208", "right")],
            ["Total / blok  (× 12 = 21,293,568)", "", ("1,774,464", "right")]]
    table(ax, PX + 0.2, 10.95, rows, [0, 2.25, 4.65], size=8.2, row_h=0.36, header=False,
          bold_rows=(5,))

    panel(ax, PX, 13.45, PW, 2.0, "D. Varian ViT (Studi Ablasi)", "#f7fbff", BLUE_D, "white")
    rows = [["Varian", "d", "h", "d_h", "Token", ("Adapter/blok", "right")],
            ["Tiny", "192", "3", "64", "197", ("24,833", "right")],
            ["Small ★", "384", "6", "64", "197", ("49,601", "right")],
            ["Base", "768", "12", "64", "197", ("99,137", "right")]]
    table(ax, PX + 0.2, 14.1, rows, [0, 1.05, 1.65, 2.15, 2.8, 4.65], size=8.4, row_h=0.3,
          hl_rows=(2,), width=PW - 0.3)

    panel(ax, PX, 15.65, PW, 2.1, "E. Ringkasan Parameter Branch", "#f7fbff", BLUE_D, "white")
    rows = [["Total parameter", ("22,260,876", "right")],
            ["Frozen (patch, CLS, pos, 12 blok, norm)", ("21,665,664", "right")],
            ["Trainable (12 × S-Adapter)", ("595,212", "right")],
            ["Rasio trainable", ("2.67 %", "right")]]
    table(ax, PX + 0.2, 16.3, rows, [0, 4.65], size=8.8, row_h=0.32, header=False,
          bold_rows=(2,))
    fig.text(0.5, 0.004, "Sumber: models/s_adapter_convnext_hybrid.py — ViTWithSAdapter & SAdapter  "
             "(urutan sesuai kode: x = block(x); x = adapter(x))", ha="center", fontsize=8,
             color=MUTED)
    save(fig, "02_ViT_Branch_Detail.png")


# ═══════════════════════ 3. FUSION MODULE ═══════════════════════
def draw_fusion():
    fig, ax = new_canvas()
    header(ax, "Fusion & Classification Module — Detail Alur & Perhitungan",
           "Konfigurasi H5 (Champion): d = 384  |  h = 6 head (d_h = 64)  |  dropout attention 0.1  |  "
           "B = batch size", PURPLE_D)
    box(ax, 0.3, 1.25, 7.9, 16.5, fc=PURPLE_BG, ec=PURPLE_D, lw=1.2, r=0.25, z=0)
    X, BW = 0.55, 7.4
    CX = X + BW / 2
    R = X + BW - 0.15

    # inputs
    box(ax, X, 1.5, 3.55, 1.05, fc="white", ec=GREEN_D, lw=1.8)
    txt(ax, X + 0.15, 1.75, r"Local Tokens $T_c$", size=11, weight="bold")
    txt(ax, X + 0.15, 2.05, "dari ConvNeXt Branch", size=8, color=MUTED)
    shape(ax, X + 3.42, 1.75, "(B, 49, 384)", size=9.5)
    tokens(ax, X + 1.75, 2.0, 6, GREEN, h=0.4)
    vx = X + 3.85
    box(ax, vx, 1.5, 3.55, 1.05, fc="white", ec=BLUE_D, lw=1.8)
    txt(ax, vx + 0.15, 1.75, r"Global Tokens $T_v$", size=11, weight="bold")
    txt(ax, vx + 0.15, 2.05, "dari ViT Branch", size=8, color=MUTED)
    shape(ax, vx + 3.42, 1.75, "(B, 196, 384)", size=9.5)
    tokens(ax, vx + 1.75, 2.0, 6, BLUE, h=0.4)

    # cross attention group
    gy, gh = 2.95, 4.7
    box(ax, X, gy, BW, gh, fc="white", ec=PURPLE_D, lw=1.6)
    txt(ax, CX, gy + 0.27, "Bidirectional Cross-Attention (Local ⇄ Global)", size=11.5,
        weight="bold", color=NAVY, ha="center")
    badge(ax, R, gy + 0.1, "trainable")

    def ca_col(x, title, q, kv, attn, out_eq, out_shape, qc, kc):
        w = 3.45
        box(ax, x, gy + 0.6, w, 3.9, fc="#f7f0fd", ec=PURPLE_D, lw=1.0, r=0.1)
        txt(ax, x + w / 2, gy + 0.85, title, size=9.6, weight="bold", ha="center", color=NAVY)
        box(ax, x + 0.15, gy + 1.15, 1.45, 0.45, fc=qc, ec=TXT, lw=0.8, r=0.06)
        txt(ax, x + 0.875, gy + 1.375, q, size=8.6, ha="center")
        box(ax, x + 1.85, gy + 1.15, 1.45, 0.45, fc=kc, ec=TXT, lw=0.8, r=0.06)
        txt(ax, x + 2.575, gy + 1.375, kv, size=8.6, ha="center")
        arrow(ax, x + 0.875, gy + 1.6, x + 1.5, gy + 1.85, ms=8, lw=1)
        arrow(ax, x + 2.575, gy + 1.6, x + 1.95, gy + 1.85, ms=8, lw=1)
        box(ax, x + 0.15, gy + 1.85, w - 0.3, 0.75, fc="#fde9a8", ec=TXT, lw=0.8, r=0.06)
        txt(ax, x + w / 2, gy + 2.07, "MultiheadAttention (h = 6)", size=8.6, ha="center",
            weight="bold")
        txt(ax, x + w / 2, gy + 2.38, attn, size=8.4, ha="center")
        arrow(ax, x + w / 2, gy + 2.6, x + w / 2, gy + 2.8, ms=8, lw=1)
        box(ax, x + 0.15, gy + 2.8, w - 0.3, 0.75, fc=GREY, ec=TXT, lw=0.8, r=0.06)
        txt(ax, x + w / 2, gy + 3.02, "Residual + Dropout + LayerNorm", size=8.4, ha="center")
        txt(ax, x + w / 2, gy + 3.32, out_eq, size=8.8, ha="center")
        shape(ax, x + w / 2 + 0.75, gy + 3.85, out_shape, size=9.5)
        tokens(ax, x + 0.2, gy + 3.68, 4, qc, h=0.33, w=0.11)

    ca_col(X + 0.15, "Local-to-Global Attention", r"$Q = \mathrm{LN}(T_c)$",
           r"$K\!=\!V\! = \mathrm{LN}(T_v)$",
           r"$A \in \mathbb{R}^{B\times6\times49\times196}$",
           r"$T'_c = \mathrm{LN}(T_c + \mathrm{Drop}(A V))$", "(B, 49, 384)", GREEN, BLUE)
    ca_col(X + 3.8, "Global-to-Local Attention", r"$Q = \mathrm{LN}(T_v)$",
           r"$K\!=\!V\! = \mathrm{LN}(T_c)$",
           r"$A \in \mathbb{R}^{B\times6\times196\times49}$",
           r"$T'_v = \mathrm{LN}(T_v + \mathrm{Drop}(A V))$", "(B, 196, 384)", BLUE, GREEN)
    # inputs to columns
    arrow(ax, X + 1.77, 2.55, X + 1.77, gy, color=GREEN_D)
    arrow(ax, vx + 1.77, 2.55, vx + 1.77, gy, color=BLUE_D)
    arrow(ax, CX, gy + gh, CX, gy + gh + 0.3, color=PURPLE_D)

    def step(y, h, title, lines, shp, fc="white", kind=None, params=None, lsize=8.8):
        box(ax, X, y, BW, h, fc=fc, ec=PURPLE_D, lw=1.4)
        txt(ax, X + 0.15, y + 0.25, title, size=10.5, weight="bold", color=NAVY)
        for i, l in enumerate(lines):
            txt(ax, X + 0.15, y + 0.58 + i * 0.3, l, size=lsize, color=TXT)
        if shp:
            shape(ax, R, y + h - 0.25, shp)
        if kind:
            badge(ax, R, y + 0.1, kind)
        if params:
            txt(ax, R, y + 0.56, params, size=8, color=MUTED, ha="right", style="italic")

    y = gy + gh + 0.3
    step(y, 1.25, "Concatenate Token",
         [r"$T_r = [T'_c\,;\,T'_v]$  (torch.cat, dim = 1)",
          r"$N = 49 + 196 = 245$ token"], "(B, 245, 384)")
    ex = tokens(ax, X + 3.9, y + 0.55, 4, GREEN, h=0.35, dots=False)
    tokens(ax, ex + 0.05, y + 0.55, 4, BLUE, h=0.35)
    arrow(ax, CX, y + 1.25, CX, y + 1.5, color=PURPLE_D)
    y += 1.5
    step(y, 1.9, "Feature Refinement — Transformer Block (pre-norm)",
         [r"$x \leftarrow x + \mathrm{MHSA}(\mathrm{LN}_1(x))$,   $A \in \mathbb{R}^{B\times6\times245\times245}$",
          r"$x \leftarrow x + \mathrm{MLP}(\mathrm{LN}_2(x))$",
          "MLP: Linear 384→1536 → GELU → Dropout(0.1) → Linear 1536→384 → Dropout"],
         "(B, 245, 384)", fc="#f7f0fd", kind="trainable", params="1,774,464 param")
    arrow(ax, CX, y + 1.9, CX, y + 2.15, color=PURPLE_D)
    y += 2.15
    step(y, 1.2, "Global Average Pooling (mean over tokens)",
         [r"$p = \frac{1}{245}\sum_{i=1}^{245} T_{r,i}$   (Tr_refined.mean(dim=1))"],
         "(B, 384)")
    box(ax, X + 4.6, y + 0.45, 0.45, 0.45, fc=PURPLE, ec=TXT, lw=0.9, r=0.05)
    arrow(ax, CX, y + 1.2, CX, y + 1.45, color=PURPLE_D)
    y += 1.45
    step(y, 1.55, "MLP Classifier",
         ["Linear(384→256) → ReLU → Dropout(0.3) → Linear(256→2)",
          r"$\mathrm{logits} = W_2\,\mathrm{Drop}(\mathrm{ReLU}(W_1 p + b_1)) + b_2$"],
         "(B, 2)", fc=BLUE, kind="trainable", params="99,074 param")
    arrow(ax, CX, y + 1.55, CX, y + 1.8, color=PURPLE_D)
    y += 1.8
    step(y, 1.65, "Softmax → Output (Real / Spoof)",
         [r"$P = \mathrm{softmax}(\mathrm{logits})$,   indeks 0 = Spoof, indeks 1 = Live (Real)",
          "Keputusan: Real jika P(live) ≥ P(spoof)"], "(B, 2)")
    ax.add_patch(Circle((X + 5.2, y + 0.98), 0.2, fc="#7ed957", ec=TXT, lw=0.8, zorder=6))
    txt(ax, X + 5.2, y + 1.37, "Real", size=8.5, ha="center", weight="bold")
    ax.add_patch(Circle((X + 6.1, y + 0.98), 0.2, fc="#ff3b30", ec=TXT, lw=0.8, zorder=6))
    txt(ax, X + 6.1, y + 1.37, "Spoof", size=8.5, ha="center", weight="bold")

    # panel kanan
    PX, PW = 8.5, 4.2
    panel(ax, PX, 1.25, PW, 4.1, "A. Rumus Cross-Attention", "#fbf6ff", PURPLE_D, "white")
    L = [r"$\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right)V$",
         r"$d_h = 384/6 = 64$",
         r"L→G: tiap 49 token lokal",
         r"     memilih info dari 196 token global",
         r"G→L: tiap 196 token global",
         r"     memilih info dari 49 token lokal",
         r"Q, K, V berbagi LN (norm_tc1, norm_tv1)"]
    for i, l in enumerate(L):
        txt(ax, PX + 0.2, 1.95 + i * 0.45, l, size=9.2 if i == 0 else 8.6)

    panel(ax, PX, 5.55, PW, 4.6, "B. Rincian Parameter Fusion", "#fbf6ff", PURPLE_D, "white")
    rows = [["Komponen", ("Param", "right")],
            ["MHA L→G (4·384²+4·384)", ("591,360", "right")],
            ["MHA G→L", ("591,360", "right")],
            ["4 × LayerNorm(384)", ("3,072", "right")],
            ["Cross-Attention total", ("1,185,792", "right")],
            ["Refinement block", ("1,774,464", "right")],
            ["Linear 384→256", ("98,560", "right")],
            ["Linear 256→2", ("514", "right")],
            ["Classifier total", ("99,074", "right")],
            ["Fusion+Head total", ("3,059,330", "right")]]
    table(ax, PX + 0.2, 6.25, rows, [0, 3.75], size=8.4, row_h=0.37, bold_rows=(4, 8, 9),
          hl_rows=(9,), hl_color="#ead6f7", width=PW - 0.3)

    panel(ax, PX, 10.35, PW, 3.6, "C. Ringkasan Model H5", "#fbf6ff", PURPLE_D, "white")
    rows = [["ConvNeXt Branch", ("49,849,889", "right")],
            ["ViT Branch", ("22,260,876", "right")],
            ["Fusion + Classifier", ("3,059,330", "right")],
            ["Total", ("75,170,095", "right")],
            ["Trainable", ("19,521,871", "right")],
            ["Rasio trainable", ("25.97 %", "right")]]
    table(ax, PX + 0.2, 11.05, rows, [0, 3.75], size=8.8, row_h=0.42, header=False,
          bold_rows=(3, 4))

    panel(ax, PX, 14.15, PW, 3.6, "D. Kompleksitas Attention", "#fbf6ff", PURPLE_D, "white")
    rows = [["Operasi", ("Matriks A / head", "right")],
            ["ViT self-attn", ("197 × 197", "right")],
            ["Local→Global", ("49 × 196", "right")],
            ["Global→Local", ("196 × 49", "right")],
            ["Refinement", ("245 × 245", "right")]]
    table(ax, PX + 0.2, 14.85, rows, [0, 3.75], size=8.6, row_h=0.4, width=PW - 0.3)
    txt(ax, PX + 0.2, 17.0, "Biaya ∝ N_q × N_k × d per lapisan", size=8, color=MUTED)

    fig.text(0.5, 0.004, "Sumber: models/s_adapter_convnext_hybrid.py — BidirectionalTokenCrossAttention, "
             "TokenRefinementBlock, TokenAntiSpoofingClassifier", ha="center", fontsize=8, color=MUTED)
    save(fig, "03_Fusion_Module_Detail.png")


if __name__ == "__main__":
    draw_convnext()
    draw_vit()
    draw_fusion()
