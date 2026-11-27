"""
Komponen bersama untuk eksperimen split resmi:
dataset, transformasi, metrik lengkap, pembuat model, optimizer, dan grafik.

Konvensi label: 1 = live, 0 = spoof. Skor model = probabilitas live.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from sklearn.metrics import roc_auc_score, roc_curve

# Agar folder `models/` di root repo bisa di-import
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.hybrid_convnext_vit_s_adapter import FocalLoss  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224

# Augmentasi & preprocessing SAMA dengan 15 model sebelumnya
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.1),
])

eval_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ============================================================
# DATASET
# ============================================================
class SplitDataset(Dataset):
    """
    Membaca citra berdasarkan baris di split.csv (lazy loading: citra dibuka
    hanya saat dibutuhkan per batch), lalu crop wajah memakai *_BB.txt resmi.
    Koordinat BB dinormalisasi pada grid 224 (sesuai README CelebA-Spoof).
    """

    def __init__(self, df: pd.DataFrame, data_dir: str, transform=None, use_bbox: bool = True):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.use_bbox = use_bbox
        self.n_gagal = 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.data_dir / row.path
        try:
            img = read_image(str(img_path), mode=ImageReadMode.RGB)
            if self.use_bbox:
                bb_path = img_path.with_name(img_path.stem + "_BB.txt")
                if bb_path.exists():
                    parts = bb_path.read_text().strip().split()
                    if len(parts) >= 4:
                        bx, by, bw, bh = [float(p) for p in parts[:4]]
                        real_h, real_w = img.shape[1], img.shape[2]
                        x1 = max(0, int(bx * (real_w / 224.0)))
                        y1 = max(0, int(by * (real_h / 224.0)))
                        x2 = min(real_w, x1 + int(bw * (real_w / 224.0)))
                        y2 = min(real_h, y1 + int(bh * (real_h / 224.0)))
                        if (x2 - x1) > 10 and (y2 - y1) > 10:
                            img = img[:, y1:y2, x1:x2]
        except Exception:
            self.n_gagal += 1
            img = torch.zeros(3, IMG_SIZE, IMG_SIZE, dtype=torch.uint8)

        if self.transform:
            img = self.transform(img.permute(1, 2, 0).numpy())
        return img, int(row.label), idx


def make_weighted_sampler(df: pd.DataFrame, seed: int):
    counts = df.label.value_counts().to_dict()
    weights = torch.tensor([len(df) / counts[l] for l in df.label], dtype=torch.double)
    g = torch.Generator()
    g.manual_seed(seed)
    return WeightedRandomSampler(weights, len(weights), replacement=True, generator=g)


# ============================================================
# METRIK LENGKAP
# ============================================================
def hitung_metrik(y_true, y_score, threshold: float = 0.5) -> dict:
    """
    Seluruh metrik dihitung dari prediksi asli (bukan rekonstruksi).
    y_true : 1 = live, 0 = spoof ; y_score : probabilitas live.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    y_pred = (y_score >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())  # live -> live
    tn = int(((y_pred == 0) & (y_true == 0)).sum())  # spoof -> spoof
    fp = int(((y_pred == 1) & (y_true == 0)).sum())  # spoof -> live (serangan lolos)
    fn = int(((y_pred == 0) & (y_true == 1)).sum())  # live -> spoof (pengguna asli ditolak)

    def bagi(a, b):
        return float(a / b) if b > 0 else 0.0

    n_spoof, n_live = tn + fp, tp + fn
    apcer = bagi(fp, n_spoof)
    bpcer = bagi(fn, n_live)

    prec_live, rec_live = bagi(tp, tp + fp), bagi(tp, tp + fn)
    prec_spoof, rec_spoof = bagi(tn, tn + fn), bagi(tn, tn + fp)
    f1_live = bagi(2 * prec_live * rec_live, prec_live + rec_live)
    f1_spoof = bagi(2 * prec_spoof * rec_spoof, prec_spoof + rec_spoof)

    m = {
        "n": int(len(y_true)), "n_live": n_live, "n_spoof": n_spoof,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": bagi(tp + tn, len(y_true)),
        "balanced_accuracy": (rec_live + rec_spoof) / 2,
        "precision_live": prec_live, "recall_live": rec_live, "f1_live": f1_live,
        "precision_spoof": prec_spoof, "recall_spoof": rec_spoof, "f1_spoof": f1_spoof,
        "precision_macro": (prec_live + prec_spoof) / 2,
        "recall_macro": (rec_live + rec_spoof) / 2,
        "f1_macro": (f1_live + f1_spoof) / 2,
        "apcer": apcer, "bpcer": bpcer, "acer": (apcer + bpcer) / 2, "hter": (apcer + bpcer) / 2,
        "auc": float("nan"), "eer": float("nan"), "eer_threshold": float("nan"),
    }
    if len(np.unique(y_true)) == 2:
        m["auc"] = float(roc_auc_score(y_true, y_score))
        fpr, tpr, thr = roc_curve(y_true, y_score)
        fnr = 1 - tpr
        i = int(np.nanargmin(np.abs(fnr - fpr)))
        m["eer"] = float((fpr[i] + fnr[i]) / 2)
        m["eer_threshold"] = float(thr[i])
    return m


# ============================================================
# MODEL & OPTIMIZER
# ============================================================
from models.ablasi import DAFTAR_MODEL, ALIAS, kode_resmi, buat_model as _buat_model_ablasi  # noqa: E402

# Semua nama yang boleh dipakai di --model: kode resmi (h1..h9, c1..c3, v4..v6) + nama lama (alias)
KONFIG_MODEL = {k: f"{m.KODE} - {m.NAMA}" for k, m in DAFTAR_MODEL.items()}
KONFIG_MODEL.update({alias: KONFIG_MODEL[kode] for alias, kode in ALIAS.items()})


def buat_model(jenis: str, mode: str = "freeze", pretrained: bool = True) -> nn.Module:
    """
    jenis : h1..h9 (hybrid), c1..c3 (baseline ConvNeXt), v4..v6 (baseline ViT), atau nama lama (lihat models/ablasi/__init__.py)
    mode  : 'freeze' -> mode penelitian (PEFT). Definisi tiap model ada di models/ablasi/
            'fullft' -> (opsional, tidak dipakai di 15 model) seluruh parameter dilatih
    """
    model = _buat_model_ablasi(jenis, pretrained=pretrained)
    if mode == "fullft":
        for p in model.parameters():
            p.requires_grad = True
    return model


AWALAN_BACKBONE = (
    "vit_branch.vit.",            # backbone ViT (hybrid)
    "convnext_branch.stem", "convnext_branch.stages", "convnext_branch.norm",  # backbone ConvNeXt (hybrid)
    "stem", "stages", "norm_pre",  # backbone ConvNeXt (baseline)
    "vit.",                        # backbone ViT (baseline)
)


def kelompok_parameter(model: nn.Module) -> dict:
    """
    3 kelompok learning rate:
    - backbone : bobot pretrained (selain ConvNeXt stage 4) -> lr_backbone (hanya terisi saat fullft)
    - adapter  : S-Adapter + ConvNeXt stage 4                -> lr_adapter (sama dengan model lama)
    - head     : cross-attention, refinement, classifier      -> lr_head    (sama dengan model lama)
    """
    grup = {"backbone": [], "adapter": [], "head": []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "adapter" in n or "stages.3" in n:
            grup["adapter"].append((n, p))
        elif n.startswith(AWALAN_BACKBONE):
            grup["backbone"].append((n, p))
        else:
            grup["head"].append((n, p))
    return grup


def buat_optimizer(model, lr_backbone, lr_adapter, lr_head):
    grup = kelompok_parameter(model)
    param_groups = []
    if grup["backbone"]:
        param_groups.append({"name": "backbone", "params": [p for _, p in grup["backbone"]],
                             "lr": lr_backbone, "weight_decay": 1e-4})
    param_groups.append({"name": "adapter", "params": [p for _, p in grup["adapter"]],
                         "lr": lr_adapter, "weight_decay": 1e-4})
    param_groups.append({"name": "head", "params": [p for _, p in grup["head"]],
                         "lr": lr_head, "weight_decay": 1e-2})
    jumlah = {k: sum(p.numel() for _, p in v) for k, v in grup.items()}
    return torch.optim.AdamW(param_groups), jumlah


def statistik_parameter(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "trainable_percentage": round(100 * trainable / total, 2),
    }


# ============================================================
# GRAFIK
# ============================================================
def plot_confusion_matrix(m: dict, judul: str, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])  # baris: aktual spoof, live
    baris_total = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm / baris_total

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Proporsi per kelas aktual")
    label = ["Spoof (0)", "Live (1)"]
    ax.set_xticks([0, 1], [f"Prediksi {l}" for l in label])
    ax.set_yticks([0, 1], [f"Aktual {l}" for l in label])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_norm[i, j] * 100:.2f}%\n({cm[i, j]:,} citra)",
                    ha="center", va="center", fontsize=12,
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_title(f"{judul}\nAkurasi {m['accuracy'] * 100:.2f}% | HTER {m['hter'] * 100:.2f}% | "
                 f"n = {m['n']:,}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_roc(y_true, y_score, m: dict, judul: str, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {m['auc']:.4f}, EER = {m['eer'] * 100:.2f}%)")
    ax.plot([0, 1], [0, 1], ls=":", color="gray", label="Tebakan acak (AUC = 0,5)")
    ax.scatter([m["apcer"]], [1 - m["bpcer"]], color="red", zorder=5,
               label=f"Threshold 0,5 (APCER {m['apcer'] * 100:.2f}%, BPCER {m['bpcer'] * 100:.2f}%)")
    ax.set_xlabel("False Positive Rate (spoof lolos sebagai live)")
    ax.set_ylabel("True Positive Rate (live dikenali sebagai live)")
    ax.set_title(f"{judul}\nKurva ROC dari {len(y_true):,} prediksi asli", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    # Inset zoom area FPR rendah (hanya berguna bila model sudah sangat baik)
    if not (m["auc"] >= 0.95):
        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return
    ins = ax.inset_axes([0.35, 0.3, 0.4, 0.35])
    ins.plot(fpr, tpr, lw=2)
    ins.set_xlim(0, 0.05)
    ins.set_ylim(0.9, 1.0)
    ins.set_title("Zoom FPR 0–5%", fontsize=8)
    ins.tick_params(labelsize=7)
    ins.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_history(hist: pd.DataFrame, judul: str, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    e = hist["epoch"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    panel = [
        ("Loss", ["train_loss", "val_loss", "test_loss"]),
        ("Accuracy", ["train_accuracy", "val_accuracy", "test_accuracy"]),
        ("AUC", ["train_auc", "val_auc", "test_auc"]),
        ("HTER", ["train_hter", "val_hter", "test_hter"]),
        ("APCER & BPCER (val)", ["val_apcer", "val_bpcer"]),
        ("Waktu per epoch (detik)", ["train_time_sec", "val_time_sec", "test_time_sec"]),
    ]
    for ax, (t, cols) in zip(axes.flat, panel):
        for c in cols:
            if c in hist and hist[c].notna().any():
                ax.plot(e, hist[c], marker="o", ms=3, label=c)
        best = hist.loc[hist["is_best"] == 1, "epoch"]
        for b in best.tail(1):
            ax.axvline(b, ls="--", color="gray", alpha=0.6, label=f"best epoch {b}")
        ax.set_title(t)
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(judul, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
