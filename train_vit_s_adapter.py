"""
Training Script: Single-Branch Vision Transformer (ViT) + S-Adapter (Face Anti-Spoofing)
========================================================================================
Skrip pelatihan studi ablasi khusus Vision Transformer (Global Feature) + S-Adapter (PEFT)
tanpa cabang ConvNeXt dan tanpa Cross-Attention.

Konfigurasi & Metrik Identik 100% dengan Model Hybrid:
- Dataset Split: Subject-Independent (70% Train, 15% Val, 15% Test) dengan SEED=42
- Weighted Random Sampler untuk menangani ketidakseimbangan kelas (Live vs Spoof)
- Loss: Focal Loss (alpha=0.25, gamma=2.0)
- Optimizer: AdamW dengan weight decay (1e-4 untuk adapter, 1e-2 untuk head)
- Scheduler: CosineAnnealingLR (T_max=25, eta_min=1e-6)
- Early Stopping: Composite Score (AUC + Accuracy - HTER - Loss), patience=5, min_delta=0.0005
- Evaluasi Test Set Unseen: Accuracy, AUC, HTER, APCER, BPCER, Confusion Matrix

Penggunaan:
  python train_vit_s_adapter.py --variant tiny
  python train_vit_s_adapter.py --variant small
  python train_vit_s_adapter.py --variant base
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import time
import argparse
import random
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.io import read_image

from models.vit_s_adapter import ViTSAdapter

# ============================================================
# 1. REPRODUCIBILITY & DEVICE SETUP
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
            if not subj.is_dir():
                continue
            for t in subj.iterdir():
                if not t.is_dir():
                    continue
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
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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
# 4. METRICS UTILITY: HTER, APCER, BPCER, AUC
# ============================================================
def calculate_metrics(y_true, y_scores, threshold=0.5):
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = (y_scores >= threshold).astype(int)

    spoof_mask = (y_true == 0)
    if spoof_mask.sum() > 0:
        apcer = float(np.mean(y_pred[spoof_mask] == 1))
    else:
        apcer = 0.0

    live_mask = (y_true == 1)
    if live_mask.sum() > 0:
        bpcer = float(np.mean(y_pred[live_mask] == 0))
    else:
        bpcer = 0.0

    hter = 0.5 * (apcer + bpcer)

    try:
        auc = float(roc_auc_score(y_true, y_scores))
    except Exception:
        auc = 0.5

    acc = float(np.mean(y_pred == y_true))

    return {
        "accuracy": acc,
        "auc": auc,
        "apcer": apcer,
        "bpcer": bpcer,
        "hter": hter,
    }

# ============================================================
# 5. FOCAL LOSS & EARLY STOPPING (KOMPOSIT IDENTIK)
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        focal_loss = alpha_t * ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0005):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -float('inf')
        self.best_checkpoint_score = -float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 1

    def check(self, val_metrics: dict, epoch: int):
        composite_score = (
            val_metrics.get("auc", 0.0)
            + val_metrics.get("accuracy", 0.0)
            - val_metrics.get("hter", 0.0)
            - val_metrics.get("loss", 0.0)
        )

        is_best = False
        if composite_score > self.best_checkpoint_score:
            self.best_checkpoint_score = composite_score
            self.best_epoch = epoch
            is_best = True

        if self.patience <= 0:
            return is_best, composite_score

        if composite_score > (self.best_score + self.min_delta):
            self.best_score = composite_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return is_best, composite_score

# ============================================================
# 6. TRAINING & EVALUATION LOOPS
# ============================================================
def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type=DEVICE.type, dtype=torch.bfloat16 if DEVICE.type == 'cuda' else torch.float32):
            out = model(images)
            loss = criterion(out["logits"], labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        preds = out["logits"].argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total*100:.2f}%"})

    return total_loss / total, correct / total

@torch.no_grad()
def evaluate_dataset(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_targets, all_scores = [], []

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        with torch.amp.autocast(device_type=DEVICE.type, dtype=torch.bfloat16 if DEVICE.type == 'cuda' else torch.float32):
            out = model(images)
            loss = criterion(out["logits"], labels)

        total_loss += loss.item() * images.size(0)
        probs = out["probs"][:, 1]  # Probabilitas Live
        all_targets.extend(labels.cpu().numpy().tolist())
        all_scores.extend(probs.cpu().numpy().tolist())

    metrics = calculate_metrics(all_targets, all_scores)
    metrics["loss"] = total_loss / len(all_targets)
    return metrics, all_targets, all_scores

# ============================================================
# 7. MAIN FUNCTION
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Pelatihan Single-Branch ViT + S-Adapter")
    parser.add_argument("--variant", type=str, default="small", choices=["tiny", "small", "base"], help="Varian ViT (tiny, small, base)")
    parser.add_argument("--data_dir", type=str, default="./CelebA_Spoof")
    parser.add_argument("--samples", type=int, default=0, help="0 = Full dataset")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr_adapter", type=float, default=1e-4)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--adapter_bottleneck", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=0.0005)
    parser.add_argument("--workers", type=int, default=4 if os.name != 'nt' else 0)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    variant_name = args.variant.lower()
    vit_model_name = f"vit_{variant_name}_patch16_224"
    exp_name = f"ViT_{variant_name.capitalize()}_SAdapter"
    save_dir = Path(args.save_dir if args.save_dir else f"./checkpoints_vit_{variant_name}")
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*75)
    print(f"MEMULAI PELATIHAN SINGLE-BRANCH: {exp_name.upper()}")
    print("="*75)
    print(f"Perangkat       : {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Direktori Simpan: {save_dir}")

    # 1. Dataset Manifest
    df = build_manifest(args.data_dir)
    print(f"Total file terdeteksi: {len(df):,}")

    if args.samples > 0 and len(df) > args.samples:
        n_half = args.samples // 2
        df_live = df[df.label == 1].sample(n=min(n_half, (df.label == 1).sum()), random_state=SEED)
        df_spoof = df[df.label == 0].sample(n=min(n_half, (df.label == 0).sum()), random_state=SEED)
        df = pd.concat([df_live, df_spoof]).reset_index(drop=True)

    # Subject-Independent Splitting: 70% Train, 15% Val, 15% Test
    subjs = list(df.subject.unique())
    if len(subjs) >= 3:
        train_subjs, test_subjs = train_test_split(subjs, test_size=0.15, random_state=SEED)
        train_subjs, val_subjs  = train_test_split(train_subjs, test_size=(0.15 / 0.85), random_state=SEED)
        df_train = df[df.subject.isin(train_subjs)].reset_index(drop=True)
        df_val   = df[df.subject.isin(val_subjs)].reset_index(drop=True)
        df_test  = df[df.subject.isin(test_subjs)].reset_index(drop=True)
    else:
        df_train, df_test = train_test_split(df, test_size=0.15, random_state=SEED)
        df_train, df_val  = train_test_split(df_train, test_size=(0.15 / 0.85), random_state=SEED)

    print(f"Data Split: Train={len(df_train):,} | Val={len(df_val):,} | Test={len(df_test):,}")

    train_ds = CelebASpoofDataset(df_train, train_transform, use_bbox=True)
    val_ds   = CelebASpoofDataset(df_val, val_transform, use_bbox=True)
    test_ds  = CelebASpoofDataset(df_test, val_transform, use_bbox=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=make_weighted_sampler(df_train),
        num_workers=args.workers, pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if args.workers > 0 else False,
        prefetch_factor=2 if args.workers > 0 else None
    )
    val_loader  = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if args.workers > 0 else False,
        prefetch_factor=2 if args.workers > 0 else None
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if args.workers > 0 else False,
        prefetch_factor=2 if args.workers > 0 else None
    )

    # 2. Inisialisasi Model
    model = ViTSAdapter(
        variant=variant_name,
        adapter_bottleneck=args.adapter_bottleneck,
        num_classes=2,
        pretrained=True,
        freeze_backbone=True,
    ).to(DEVICE)

    model.print_parameter_summary()

    # Parameter Stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    trainable_pct = 100.0 * trainable_params / total_params

    # Simpan Training Config JSON
    config_data = {
        "experiment_name": exp_name,
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {
            "data_dir": args.data_dir,
            "total_manifest": len(df),
            "train_samples": len(df_train),
            "val_samples": len(df_val),
            "test_samples": len(df_test),
            "device": DEVICE.type,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "mixed_precision": "bfloat16" if DEVICE.type == "cuda" else "fp32",
            "workers": args.workers,
        },
        "model_architecture": {
            "vit_name": vit_model_name,
            "variant": variant_name,
            "embedding_dim": model.embed_dim,
            "adapter_bottleneck": args.adapter_bottleneck,
            "num_classes": 2,
            "pretrained": True,
            "freeze_backbone": True,
        },
        "parameter_efficiency": {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": frozen_params,
            "trainable_percentage": f"{trainable_pct:.2f}%",
        },
        "training_hyperparameters": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr_adapter": args.lr_adapter,
            "lr_head": args.lr_head,
            "early_stopping_metric": "CompositeScore (AUC + Acc - HTER - Loss)",
            "early_stopping_patience": args.patience,
            "early_stopping_min_delta": args.min_delta,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR (T_max=25, eta_min=1e-6)",
            "loss": "FocalLoss (gamma=2.0, alpha=0.25)",
            "seed": SEED,
        }
    }
    with open(save_dir / "training_config.json", "w") as fp:
        json.dump(config_data, fp, indent=4)

    # 3. Setup Optimizer
    adapter_params = [p for p in model.adapters.parameters() if p.requires_grad]
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": adapter_params, "lr": args.lr_adapter, "weight_decay": 1e-4},
        {"params": head_params, "lr": args.lr_head, "weight_decay": 1e-2},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))
    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

    # 4. Training Loop
    history = []
    best_ckpt_path = save_dir / f"best_vit_{variant_name}_s_adapter.pth"
    t_start = time.time()

    print(f"\nMemulai Pelatihan {exp_name} (Max {args.epochs} Epochs)...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        val_metrics, _, _ = evaluate_dataset(model, val_loader, criterion)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed/60:.1f}m) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']*100:.2f}% "
              f"AUC: {val_metrics['auc']:.4f} HTER: {val_metrics['hter']*100:.2f}%")

        is_best, comp_score = early_stopping.check(val_metrics, epoch)
        if is_best:
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "variant": variant_name,
                "model_name": vit_model_name,
                "model_state_dict": model.state_dict(),
                "val_acc": val_metrics["accuracy"],
                "val_auc": val_metrics["auc"],
                "val_hter": val_metrics["hter"],
                "val_loss": val_metrics["loss"],
                "composite_score": comp_score,
            }, best_ckpt_path)
            print(f"  --> Checkpoint Terbaik Disimpan! (Val AUC: {val_metrics['auc']:.4f}, HTER: {val_metrics['hter']*100:.2f}%)")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_auc": val_metrics["auc"],
            "val_hter": val_metrics["hter"],
            "val_apcer": val_metrics["apcer"],
            "val_bpcer": val_metrics["bpcer"],
        })

        if early_stopping.early_stop:
            print(f"\n[Info] Early stopping terpicu pada epoch {epoch}!")
            break

    total_duration_minutes = (time.time() - t_start) / 60.0

    # 5. Evaluasi Final pada Test Set Unseen
    print("\n" + "="*75)
    print("EVALUASI FINAL PADA TEST SET UNSEEN (MENGGUNAKAN CHECKPOINT TERBAIK)")
    print("="*75)
    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Success] Memuat model terbaik dari Epoch {ckpt.get('epoch')}")

    test_metrics, test_targets, test_scores = evaluate_dataset(model, test_loader, criterion)
    print(f"Test Accuracy : {test_metrics['accuracy']*100:.2f}%")
    print(f"Test AUC      : {test_metrics['auc']:.4f}")
    print(f"Test HTER     : {test_metrics['hter']*100:.2f}%")
    print(f"Test APCER    : {test_metrics['apcer']*100:.2f}%")
    print(f"Test BPCER    : {test_metrics['bpcer']*100:.2f}%\n")

    # Simpan Ringkasan Akhir
    summary_data = {
        "experiment_name": exp_name,
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_duration_minutes": round(total_duration_minutes, 2),
        "early_stopping": {
            "triggered": early_stopping.early_stop,
            "stopped_epoch": len(history),
        },
        "best_checkpoint": {
            "best_epoch": early_stopping.best_epoch,
            "best_val_auc": round(float(ckpt.get("val_auc", 0.0)), 4) if best_ckpt_path.exists() else 0.0,
        },
        "final_test_set_evaluation": {
            "test_accuracy": round(float(test_metrics['accuracy']), 4),
            "test_accuracy_pct": f"{test_metrics['accuracy']*100:.2f}%",
            "test_auc": round(float(test_metrics['auc']), 4),
            "test_apcer": round(float(test_metrics['apcer']), 4),
            "test_apcer_pct": f"{test_metrics['apcer']*100:.2f}%",
            "test_bpcer": round(float(test_metrics['bpcer']), 4),
            "test_bpcer_pct": f"{test_metrics['bpcer']*100:.2f}%",
            "test_hter": round(float(test_metrics['hter']), 4),
            "test_hter_pct": f"{test_metrics['hter']*100:.2f}%",
            "test_loss": round(float(test_metrics['loss']), 4),
        }
    }
    with open(save_dir / "final_test_summary.json", "w") as fp:
        json.dump(summary_data, fp, indent=4)

    pd.DataFrame(history).to_csv(save_dir / "training_history.csv", index=False)
    with open(save_dir / "training_history.json", "w") as fp:
        json.dump(history, fp, indent=4)

    # Buat Plot Grafik Hasil Evaluasi
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot([h['epoch'] for h in history], [h['train_loss'] for h in history], label='Train Loss')
    axes[0].plot([h['epoch'] for h in history], [h['val_loss'] for h in history], label='Val Loss')
    axes[0].set_title(f'{exp_name} - Loss History')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot([h['epoch'] for h in history], [h['val_acc']*100 for h in history], label='Val Acc (%)')
    axes[1].plot([h['epoch'] for h in history], [h['val_auc']*100 for h in history], label='Val AUC (x100)')
    axes[1].plot([h['epoch'] for h in history], [h['val_hter']*100 for h in history], label='Val HTER (%)')
    axes[1].set_title(f'{exp_name} - Validation Metrics')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Percentage (%)')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_dir / f"training_history_{variant_name}.png", dpi=150)
    plt.close()

    print(f"\n[Selesai] Semua artefak evaluasi tersimpan di: {save_dir}")

if __name__ == "__main__":
    main()
