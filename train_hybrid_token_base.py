"""
Training Script: Hybrid Token S-Adapter (ConvNeXt-Base + ViT-Base) for Face Anti-Spoofing
========================================================================================
Sesuai Gambar Arsitektur: Arsitektur TA.png
- Branch 1: ConvNeXt-Base (Local Feature Extraction)
    * STEM: (B, 128, 56, 56)
    * Stage 1-4: Stage 4 Feature Map = (B, 1024, 7, 7)
    * Flatten: 7x7 = 49 tokens (B, 49, 1024)
    * S-Adapter (Adapted for ConvNeXt Tokens - TSR): Proyeksi 1024 -> d (768)
    * Output: Local Tokens Tc of shape (B, 49, 768)
- Branch 2: ViT-Base (Global Feature Extraction)
    * Patch Embedding 16x16: 14x14 = 196 patch tokens (B, 196, 768)
    * CLS + Positional Embedding: (B, 197, 768)
    * L=12 Transformer Encoder Blocks dengan S-Adapter
    * Remove CLS Token: Global Tokens Tv of shape (B, 196, 768)
- Fusion Module:
    * Bidirectional Cross-Attention:
        - Local-to-Global: Q=Tc, K=Tv, V=Tv -> (B, 49, 768)
        - Global-to-Local: Q=Tv, K=Tc, V=Tc -> (B, 196, 768)
    * Concatenate: Tr = [T'c ; T'v] -> (B, 245, 768)
    * Transformer Refinement Block (Self-Attention + MLP) -> (B, 245, 768)
    * Global Average Pooling -> (B, 768)
    * MLP Classifier: Linear(768 -> 256) -> ReLU -> Dropout(0.3) -> Linear(256 -> 2)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import math, random, time, warnings, argparse
warnings.filterwarnings('ignore')
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.io import read_image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from models.s_adapter_convnext_hybrid import HybridTokenSAdapterConvNeXt, FocalLoss, get_optimizer

# ============================================================
# 1. SEED & DEVICE SETUP
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# 2. MANIFEST CELEBA-SPOOF
# ============================================================
def build_manifest(data_root="./CelebA_Spoof"):
    data_root = Path(data_root)
    records = []
    for split in ['train', 'test']:
        split_dir = data_root / split
        if not split_dir.exists():
            split_dir = data_root / 'Data' / split
        if not split_dir.exists():
            continue
        for subj in split_dir.iterdir():
            if not subj.is_dir(): continue
            for t in subj.iterdir():
                if not t.is_dir(): continue
                # Label 1 = Live (Real), Label 0 = Spoof
                label = 1 if t.name.lower() in ['1', 'live'] else 0
                for img in list(t.glob('*.jpg')) + list(t.glob('*.png')):
                    records.append({
                        'path': str(img),
                        'label': label,
                        'split': split,
                        'subject': subj.name
                    })
    return pd.DataFrame(records)

# ============================================================
# 3. BOUNDING BOX SCALING & TRANSFORMS
# ============================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE = 224

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.GaussianBlur(3, sigma=(0.1, 1.0)),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.1),
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

class CelebASpoofDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, use_bbox: bool = True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.use_bbox = use_bbox

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(row.path)
        try:
            img = read_image(str(img_path))
            if img.shape[0] == 4:
                img = img[:3]
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)

            # Crop wajah sesuai anotasi _BB.txt (grid 224x224 RetinaFace)
            if self.use_bbox:
                bb_path = Path(str(img_path).replace('.jpg', '_BB.txt').replace('.png', '_BB.txt'))
                if bb_path.exists():
                    with open(bb_path, 'r') as f:
                        parts = f.read().strip().split()
                        if len(parts) >= 4:
                            bbox = [float(p) for p in parts]
                            real_h, real_w = img.shape[1], img.shape[2]
                            x1 = max(0, int(bbox[0] * (real_w / 224.0)))
                            y1 = max(0, int(bbox[1] * (real_h / 224.0)))
                            w1 = int(bbox[2] * (real_w / 224.0))
                            h1 = int(bbox[3] * (real_h / 224.0))
                            x2 = min(real_w, x1 + w1)
                            y2 = min(real_h, y1 + h1)

                            if (x2 - x1) > 10 and (y2 - y1) > 10:
                                img = img[:, y1:y2, x1:x2]
        except Exception:
            img = torch.zeros(3, IMG_SIZE, IMG_SIZE, dtype=torch.uint8)

        if self.transform:
            img = self.transform(img.numpy().transpose(1, 2, 0))

        return img, int(row.label)

def make_weighted_sampler(df: pd.DataFrame):
    counts = df.label.value_counts().to_dict()
    total = len(df)
    weights = [total / counts[lbl] for lbl in df.label]
    return WeightedRandomSampler(weights, len(weights), replacement=True)

# ============================================================
# 4. METRICS: HTER, APCER, BPCER, AUC
# ============================================================
def calculate_metrics(y_true, y_scores, threshold=0.5):
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = (y_scores >= threshold).astype(int)

    spoof_mask = (y_true == 0)
    apcer = float(np.mean(y_pred[spoof_mask] == 1)) if spoof_mask.sum() > 0 else 0.0

    live_mask = (y_true == 1)
    bpcer = float(np.mean(y_pred[live_mask] == 0)) if live_mask.sum() > 0 else 0.0

    hter = (apcer + bpcer) / 2.0
    acc = float(np.mean(y_pred == y_true))

    try:
        auc = roc_auc_score(y_true, y_scores)
    except:
        auc = 0.5

    return {
        "accuracy": acc,
        "auc": auc,
        "apcer": apcer,
        "bpcer": bpcer,
        "hter": hter,
    }

# ============================================================
# 5. TRAINING & EVALUATION
# ============================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="Training Batch (Base)", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        out = model(images)
        logits = out["logits"]
        loss = criterion(logits, labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.3f}"})

    return total_loss / total, correct / total

@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_targets, all_scores = [], []

    for images, labels in tqdm(loader, desc="Eval Batch", leave=False):
        images, labels = images.to(device), labels.to(device)
        out = model(images)
        logits = out["logits"]
        probs = out["probs"]
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        all_targets.extend(labels.cpu().numpy().tolist())
        all_scores.extend(probs[:, 1].cpu().numpy().tolist())

    avg_loss = total_loss / len(all_targets)
    metrics = calculate_metrics(all_targets, all_scores)
    metrics["loss"] = avg_loss
    return metrics, all_targets, all_scores

# ============================================================
# 6. MAIN PIPELINE KHUSUS BASE
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Training Hybrid Token S-Adapter (ConvNeXt-Base + ViT-Base)")
    parser.add_argument("--data_dir", type=str, default="./CelebA_Spoof")
    parser.add_argument("--samples", type=int, default=-1, help="-1 untuk seluruh dataset CelebA-Spoof")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size 16 optimal untuk model Base di GPU VRAM 18GB")
    parser.add_argument("--epochs", type=int, default=20, help="Total epoch training")
    parser.add_argument("--lr_adapter", type=float, default=1e-4)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--vit_name", type=str, default="vit_base_patch16_224", help="Backbone ViT BASE (embed_dim d=768)")
    parser.add_argument("--convnext_name", type=str, default="convnext_base", help="Backbone ConvNeXt BASE (C4=1024)")
    parser.add_argument("--adapter_bottleneck", type=int, default=64, help="Bottleneck dimensi S-Adapter")
    parser.add_argument("--num_heads", type=int, default=12, help="Attention heads (768 / 12 = 64 head dim)")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_base", help="Folder khusus menyimpan model Base")
    parser.add_argument("--workers", type=int, default=4 if os.name != 'nt' else 0)
    args = parser.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "="*75)
    print("MEMBANGUN MANIFEST DATASET CELEBA-SPOOF (VERSI BASE)")
    print("="*75)
    df = build_manifest(args.data_dir)
    print(f"Total file terdeteksi: {len(df):,}")

    if args.samples > 0 and len(df) > args.samples:
        n_half = args.samples // 2
        df_live = df[df.label == 1].sample(n=min(n_half, (df.label == 1).sum()), random_state=SEED)
        df_spoof = df[df.label == 0].sample(n=min(n_half, (df.label == 0).sum()), random_state=SEED)
        df = pd.concat([df_live, df_spoof]).reset_index(drop=True)

    # Subject-Independent Splitting
    subjs = list(df.subject.unique())
    if len(subjs) >= 3:
        train_subjs, test_subjs = train_test_split(subjs, test_size=0.2, random_state=SEED)
        train_subjs, val_subjs  = train_test_split(train_subjs, test_size=0.2, random_state=SEED)
        df_train = df[df.subject.isin(train_subjs)].reset_index(drop=True)
        df_val   = df[df.subject.isin(val_subjs)].reset_index(drop=True)
        df_test  = df[df.subject.isin(test_subjs)].reset_index(drop=True)
    else:
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=SEED)
        df_train, df_val  = train_test_split(df_train, test_size=0.2, random_state=SEED)

    print(f"Data Split: Train={len(df_train):,} | Val={len(df_val):,} | Test={len(df_test):,}")

    train_ds = CelebASpoofDataset(df_train, train_transform, use_bbox=True)
    val_ds   = CelebASpoofDataset(df_val, val_transform, use_bbox=True)
    test_ds  = CelebASpoofDataset(df_test, val_transform, use_bbox=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=make_weighted_sampler(df_train),
        num_workers=args.workers, pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader  = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    print("\n" + "="*75)
    print("INISIALISASI MODEL HYBRID TOKEN S-ADAPTER + CONVNEXT (VERSI BASE)")
    print("="*75)
    print(f"Backbone 1 : ConvNeXt-Base (Stage 4 C4=1024 -> Spatial Flatten 49 -> S-Adapter TSR -> Tc 49x768)")
    print(f"Backbone 2 : ViT-Base (Patch 16x16 -> 196 tokens -> S-Adapter -> Tv 196x768)")
    print(f"Fusion     : Bidirectional Cross-Attention (12 Heads) -> Tr 245x768 -> Refinement Block")
    print(f"Classifier : Global Pooling (768) -> Linear(768 -> 256) -> ReLU -> Dropout -> Linear(256 -> 2)")

    model = HybridTokenSAdapterConvNeXt(
        vit_name=args.vit_name,
        convnext_name=args.convnext_name,
        adapter_bottleneck=args.adapter_bottleneck,
        num_heads=args.num_heads,
        num_classes=2,
        pretrained=True,
        freeze_backbone=True
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameter      : {total_params:,}")
    print(f"Trainable Parameter  : {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    criterion = FocalLoss(gamma=2.0, alpha=0.25)
    optimizer = get_optimizer(model, lr_adapter=args.lr_adapter, lr_head=args.lr_head)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_auc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_auc": [], "val_hter": []}

    print("\n" + "="*75)
    print("MEMULAI PROSES TRAINING (VERSI BASE)")
    print("="*75)
    best_ckpt_path = Path(args.save_dir) / "best_token_hybrid_base_model.pth"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_metrics, _, _ = evaluate_epoch(model, val_loader, criterion, DEVICE)
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_auc"].append(val_metrics["auc"])
        history["val_hter"].append(val_metrics["hter"])

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']*100:.2f}% "
            f"AUC: {val_metrics['auc']:.4f} HTER: {val_metrics['hter']*100:.2f}%"
        )

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": best_val_auc,
                "val_acc": val_metrics["accuracy"],
                "val_hter": val_metrics["hter"],
                "arch": "HybridTokenSAdapterConvNeXt_BASE"
            }, best_ckpt_path)
            print(f"  -> Checkpoint BASE terbaik disimpan di: {best_ckpt_path}")

    # Evaluasi Final pada Test Set
    print("\n" + "="*75)
    print("EVALUASI FINAL MODEL BASE PADA TEST SET")
    print("="*75)
    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Checkpoint terbaik (Epoch {ckpt['epoch']}) dimuat.")

    test_metrics, _, _ = evaluate_epoch(model, test_loader, criterion, DEVICE)
    print("\n--- HASIL EVALUASI TEST SET (HYBRID TOKEN S-ADAPTER + CONVNEXT BASE) ---")
    print(f"Accuracy : {test_metrics['accuracy']*100:.2f}%")
    print(f"AUC      : {test_metrics['auc']:.4f}")
    print(f"APCER    : {test_metrics['apcer']*100:.2f}%")
    print(f"BPCER    : {test_metrics['bpcer']*100:.2f}%")
    print(f"HTER     : {test_metrics['hter']*100:.2f}%")

    # Simpan Grafik Training Khusus Base
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(history["train_loss"], label="Train Loss", color="blue")
    axes[0].plot(history["val_loss"], label="Val Loss", color="red")
    axes[0].set_title("Base Model: Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train Acc", color="blue")
    axes[1].plot(history["val_acc"], label="Val Acc", color="green")
    axes[1].set_title("Base Model: Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(history["val_auc"], label="Val AUC", color="purple")
    axes[2].plot(history["val_hter"], label="Val HTER", color="orange")
    axes[2].set_title("Base Model: AUC & HTER")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    plt.tight_layout()
    chart_path = "./training_history_token_hybrid_base.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"[SUCCESS] Grafik training BASE disimpan di: {chart_path}")

if __name__ == "__main__":
    main()

