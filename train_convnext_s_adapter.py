"""
Training Script: Single-Branch ConvNeXt + S-Adapter (Face Anti-Spoofing)
========================================================================
Skrip pelatihan untuk studi ablasi khusus ConvNeXt + S-Adapter (tanpa ViT).
Mendukung 3 varian: Tiny, Small, dan Base.

Penggunaan:
  python train_convnext_s_adapter.py --variant tiny
  python train_convnext_s_adapter.py --variant small
  python train_convnext_s_adapter.py --variant base
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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image

from models.convnext_s_adapter import ConvNeXtSAdapter

# ============================================================
# 1. REPRODUCIBILITY & PERANGKAT
# ============================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ============================================================
# 2. DATASET LOADER CELEBA-SPOOF
# ============================================================
def scan_celeba_spoof(data_dir: Path) -> pd.DataFrame:
    records = []
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
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
                            if w1 > 10 and h1 > 10 and x2 > x1 and y2 > y1:
                                img = img[:, y1:y2, x1:x2]
        except Exception:
            img = torch.zeros((3, IMG_SIZE, IMG_SIZE), dtype=torch.uint8)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(row.label, dtype=torch.long)

# ============================================================
# 3. FOCAL LOSS & EARLY STOPPING
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
# 4. TRAINING & VALIDATION LOOPS
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32):
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
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_targets, all_probs, all_preds = [], [], []

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images, labels = images.to(device), labels.to(device)
        out = model(images)
        loss = criterion(out["logits"], labels)

        total_loss += loss.item() * images.size(0)
        probs = out["probs"][:, 1]
        preds = out["logits"].argmax(dim=-1)

        all_targets.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    acc = (all_preds == all_targets).mean()
    loss = total_loss / len(all_targets)

    try:
        auc = roc_auc_score(all_targets, all_probs)
        fpr, tpr, thresholds = roc_curve(all_targets, all_probs)
        fnr = 1.0 - tpr
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        th_eer = thresholds[eer_idx]
    except Exception:
        auc = 0.5
        th_eer = 0.5

    preds_at_th = (all_probs >= th_eer).astype(int)
    cm = confusion_matrix(all_targets, preds_at_th, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    apcer = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    bpcer = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    hter = 0.5 * (apcer + bpcer)

    return {
        "loss": loss,
        "accuracy": acc,
        "auc": auc,
        "apcer": apcer,
        "bpcer": bpcer,
        "hter": hter,
        "all_targets": all_targets,
        "all_probs": all_probs,
        "all_preds": preds_at_th,
    }

# ============================================================
# 5. FUNGSI UTAMA (MAIN RUNNER)
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Pelatihan Single-Branch ConvNeXt + S-Adapter")
    parser.add_argument("--variant", type=str, default="small", choices=["tiny", "small", "base"], help="Varian ConvNeXt (tiny, small, base)")
    parser.add_argument("--data_dir", type=str, default="./CelebA_Spoof", help="Path dataset CelebA-Spoof")
    parser.add_argument("--epochs", type=int, default=25, help="Jumlah epoch maksimal")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr_adapter", type=float, default=1e-4, help="Learning rate S-Adapter")
    parser.add_argument("--lr_head", type=float, default=1e-3, help="Learning rate Classifier Head")
    parser.add_argument("--patience", type=int, default=5, help="Patience early stopping")
    parser.add_argument("--dry_run", action="store_true", help="Uji coba 200 sampel untuk verifikasi cepat")
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Info] Menggunakan perangkat: {device}")

    # Direktori Checkpoint
    exp_name = f"ConvNeXt_{args.variant.capitalize()}_SAdapter"
    output_dir = Path(f"checkpoints_convnext_{args.variant.lower()}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Bangun Model
    model = ConvNeXtSAdapter(
        variant=args.variant,
        adapter_bottleneck=64,
        num_classes=2,
        pretrained=True,
        freeze_backbone=True,
    ).to(device)

    model.print_parameter_summary()

    # 2. Setup Dataset
    data_path = Path(args.data_dir)
    print(f"[Info] Memindai dataset di {data_path}...")
    manifest = scan_celeba_spoof(data_path)

    if manifest.empty or args.dry_run:
        print("[Warning] Dataset CelebA-Spoof lokal tidak ditemukan / Mode Dry Run aktif.")
        print("[Info] Membuat dataset sintetis 1000 citra untuk pengujian fungsional...")
        dummy_paths = [f"dummy_{i}.jpg" for i in range(600)]
        manifest = pd.DataFrame({
            "path": dummy_paths,
            "label": [i % 2 for i in range(600)],
            "split": ["train"] * 400 + ["val"] * 100 + ["test"] * 100,
            "subject": [f"subj_{i//2}" for i in range(600)]
        })

    train_df = manifest[manifest.split == 'train']
    val_df = manifest[manifest.split == 'val']
    test_df = manifest[manifest.split == 'test']

    print(f"[Dataset Split] Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    train_loader = DataLoader(CelebASpoofDataset(train_df, transform=train_transform, use_bbox=False), batch_size=args.batch_size, shuffle=True, num_workers=4 if os.name != 'nt' else 0, pin_memory=True)
    val_loader = DataLoader(CelebASpoofDataset(val_df, transform=val_transform, use_bbox=False), batch_size=args.batch_size, shuffle=False, num_workers=4 if os.name != 'nt' else 0, pin_memory=True)
    test_loader = DataLoader(CelebASpoofDataset(test_df, transform=val_transform, use_bbox=False), batch_size=args.batch_size, shuffle=False, num_workers=4 if os.name != 'nt' else 0, pin_memory=True)

    # 3. Setup Optimizer & Scheduler
    adapter_params = [p for p in model.spatial_adapter.parameters() if p.requires_grad]
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    # Unfrozen Stage 4 backbone params
    stage4_params = [p for p in model.stages[-1].parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": adapter_params + stage4_params, "lr": args.lr_adapter, "weight_decay": 1e-4},
        {"params": head_params, "lr": args.lr_head, "weight_decay": 1e-2},
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.0005)

    # 4. Loop Pelatihan
    history = []
    best_ckpt_path = output_dir / f"best_convnext_{args.variant.lower()}_s_adapter.pth"

    print(f"\nMemulai Pelatihan {exp_name} ({args.epochs} Epochs)...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']*100:.2f}% "
              f"AUC: {val_metrics['auc']:.4f} HTER: {val_metrics['hter']*100:.2f}%")

        is_best, comp_score = early_stopping.check(val_metrics, epoch)
        if is_best:
            torch.save({
                "epoch": epoch,
                "variant": args.variant,
                "model_state_dict": model.state_dict(),
                "val_acc": val_metrics["accuracy"],
                "val_auc": val_metrics["auc"],
                "val_hter": val_metrics["hter"],
            }, best_ckpt_path)
            print(f"  --> Checkpoint Terbaik Disimpan! (Val AUC: {val_metrics['auc']:.4f})")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_auc": val_metrics["auc"],
            "val_hter": val_metrics["hter"],
        })

        if early_stopping.early_stop:
            print(f"\n[Info] Early stopping terpicu pada epoch {epoch}!")
            break

    # 5. Evaluasi Final pada Test Set
    print("\n" + "="*70)
    print("EVALUASI FINAL PADA TEST SET (MODEL CHECKPOINT TERBAIK)")
    print("="*70)
    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Success] Memuat model terbaik dari Epoch {ckpt.get('epoch')}")

    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"Test Accuracy : {test_metrics['accuracy']*100:.2f}%")
    print(f"Test AUC      : {test_metrics['auc']:.4f}")
    print(f"Test HTER     : {test_metrics['hter']*100:.2f}%")
    print(f"Test APCER    : {test_metrics['apcer']*100:.2f}%")
    print(f"Test BPCER    : {test_metrics['bpcer']*100:.2f}%\n")

    # Simpan Ringkasan Evaluasi
    summary_data = {
        "experiment_name": exp_name,
        "variant": args.variant,
        "best_epoch": early_stopping.best_epoch,
        "test_evaluation": {
            "test_accuracy": round(float(test_metrics['accuracy']), 4),
            "test_accuracy_pct": f"{test_metrics['accuracy']*100:.2f}%",
            "test_auc": round(float(test_metrics['auc']), 4),
            "test_hter": round(float(test_metrics['hter']), 4),
            "test_hter_pct": f"{test_metrics['hter']*100:.2f}%",
            "test_apcer": round(float(test_metrics['apcer']), 4),
            "test_apcer_pct": f"{test_metrics['apcer']*100:.2f}%",
            "test_bpcer": round(float(test_metrics['bpcer']), 4),
            "test_bpcer_pct": f"{test_metrics['bpcer']*100:.2f}%",
        }
    }
    with open(output_dir / "final_test_summary.json", "w") as fp:
        json.dump(summary_data, fp, indent=4)

    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    print(f"[Done] Hasil tersimpan di folder: {output_dir}")

if __name__ == "__main__":
    main()

