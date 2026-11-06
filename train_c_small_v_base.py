"""
Training Script: Hybrid Token S-Adapter (ConvNeXt-Small + ViT-Base) for Face Anti-Spoofing
========================================================================================
Sesuai Gambar Arsitektur: Arsitektur TA.png
- Branch 1: ConvNeXt-Small (Stage 4 C4=768 -> TSR -> Tc 49x768)
- Branch 2: ViT-Base (Patch 16x16 -> 196 tokens -> S-Adapter -> Tv 196x768)
- Fusion   : Bidirectional Cross-Attention (12 Heads) -> Tr 245x768 -> Refinement
- Classifier : Global Pooling (768) -> Linear(768 -> 256) -> ReLU -> Dropout -> Linear(256 -> 2)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import math, random, time, warnings, argparse
import json, csv
from datetime import datetime
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
    """
    Dataset loader yang membaca gambar dan file anotasi *_BB.txt.
    Koordinat [x, y, w, h, score] dinormalisasi terhadap grid 224x224
    sesuai instruksi resmi dataset CelebA-Spoof.
    """
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
    """
    Menghitung AUC, APCER, BPCER, dan HTER (Half Total Error Rate).
    Label 1 = Live, 0 = Spoof.
    y_scores adalah probabilitas Live (kelas 1).
    """
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
# 5. EXPERIMENT LOGGER (POINT 1 GUIDANCE)
# ============================================================
class TrainingLogger:
    """
    Menangani Point 1: Pencatatan parameter konfigurasi, parameter PEFT,
    riwayat epoch (CSV & JSON secara realtime), serta ringkasan evaluasi akhir.
    """
    def __init__(self, save_dir, experiment_name="HybridToken_CSmall_VBase"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.csv_path = self.save_dir / "training_history.csv"
        self.json_path = self.save_dir / "training_history.json"
        self.config_path = self.save_dir / "training_config.json"
        self.summary_path = self.save_dir / "final_test_summary.json"
        
        self.records = []
        self._init_csv()

    def _init_csv(self):
        with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "elapsed_sec", "lr_adapter", "lr_head",
                "train_loss", "train_acc",
                "val_loss", "val_acc", "val_auc", "val_hter", "val_apcer", "val_bpcer",
                "is_best"
            ])

    def log_config(self, args_dict: dict, model_stats: dict, dataset_stats: dict):
        config_data = {
            "experiment_name": self.experiment_name,
            "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": dataset_stats,
            "model_architecture": {
                "vit_name": args_dict.get("vit_name"),
                "convnext_name": args_dict.get("convnext_name"),
                "adapter_bottleneck": args_dict.get("adapter_bottleneck"),
                "num_heads": args_dict.get("num_heads"),
                "num_classes": 2,
                "pretrained": True,
                "freeze_backbone": True,
            },
            "parameter_efficiency": model_stats,
            "training_hyperparameters": {
                "batch_size": args_dict.get("batch_size"),
                "epochs": args_dict.get("epochs"),
                "lr_adapter": args_dict.get("lr_adapter"),
                "lr_head": args_dict.get("lr_head"),
                "early_stopping_patience": args_dict.get("patience"),
                "early_stopping_min_delta": args_dict.get("min_delta"),
                "early_stopping_enabled": args_dict.get("patience", 0) > 0,
                "optimizer": "AdamW",
                "weight_decay_adapter": 1e-4,
                "weight_decay_head": 1e-2,
                "scheduler": f"CosineAnnealingLR (T_max={args_dict.get('epochs')}, eta_min=1e-6)",
                "loss": "FocalLoss (gamma=2.0, alpha=0.25)",
                "seed": 42,
            }
        }
        with open(self.config_path, mode="w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4)
        print(f"[LOGGER] Konfigurasi parameter disimpan di: {self.config_path}")

    def log_epoch(self, epoch: int, elapsed: float, lr_adapter: float, lr_head: float,
                  train_loss: float, train_acc: float, val_metrics: dict, is_best: bool):
        row = [
            epoch, f"{elapsed:.2f}", f"{lr_adapter:.6e}", f"{lr_head:.6e}",
            f"{train_loss:.4f}", f"{train_acc:.4f}",
            f"{val_metrics['loss']:.4f}", f"{val_metrics['accuracy']:.4f}",
            f"{val_metrics['auc']:.4f}", f"{val_metrics['hter']:.4f}",
            f"{val_metrics.get('apcer', 0.0):.4f}", f"{val_metrics.get('bpcer', 0.0):.4f}",
            1 if is_best else 0
        ]
        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
            f.flush()

        record = {
            "epoch": epoch,
            "elapsed_sec": round(elapsed, 2),
            "lr_adapter": lr_adapter,
            "lr_head": lr_head,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_acc": round(val_metrics["accuracy"], 4),
            "val_auc": round(val_metrics["auc"], 4),
            "val_hter": round(val_metrics["hter"], 4),
            "val_apcer": round(val_metrics.get("apcer", 0.0), 4),
            "val_bpcer": round(val_metrics.get("bpcer", 0.0), 4),
            "is_best": is_best
        }
        self.records.append(record)
        with open(self.json_path, mode="w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=4)

    def log_final_results(self, best_epoch: int, best_val_auc: float, test_metrics: dict, total_duration: float, early_stopped: bool = False, stopped_epoch: int = None):
        summary = {
            "experiment_name": self.experiment_name,
            "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration_minutes": round(total_duration / 60, 2),
            "early_stopping": {
                "triggered": early_stopped,
                "stopped_epoch": stopped_epoch if early_stopped else None
            },
            "best_checkpoint": {
                "best_epoch": best_epoch,
                "best_val_auc": round(best_val_auc, 4)
            },
            "final_test_set_evaluation": {
                "test_accuracy": round(test_metrics["accuracy"], 4),
                "test_accuracy_pct": f"{test_metrics['accuracy'] * 100:.2f}%",
                "test_auc": round(test_metrics["auc"], 4),
                "test_apcer": round(test_metrics["apcer"], 4),
                "test_apcer_pct": f"{test_metrics['apcer'] * 100:.2f}%",
                "test_bpcer": round(test_metrics["bpcer"], 4),
                "test_bpcer_pct": f"{test_metrics['bpcer'] * 100:.2f}%",
                "test_hter": round(test_metrics["hter"], 4),
                "test_hter_pct": f"{test_metrics['hter'] * 100:.2f}%",
                "test_loss": round(test_metrics.get("loss", 0.0), 4),
            }
        }
        with open(self.summary_path, mode="w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
        print(f"\n[LOGGER] Ringkasan hasil evaluasi akhir disimpan di: {self.summary_path}")
        print(f"[LOGGER] Log riwayat per-epoch lengkap CSV di: {self.csv_path}")

# ============================================================
# 6. EARLY STOPPING (POINT 2 GUIDANCE)
# ============================================================
class EarlyStopping:
    """
    Menangani Point 2: Early Stopping untuk memantau metrik (val_auc)
    dan menghentikan training otomatis jika performa tidak kunjung membaik
    setelah 'patience' epoch berturut-turut demi mencegah overfitting.
    """
    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = -float("inf") if mode == "max" else float("inf")
        self.early_stop = False
        self.best_epoch = 1

    def check(self, current_score: float, epoch: int) -> bool:
        """
        Mengecek apakah ada peningkatan skor melampaui min_delta.
        Return True jika tercatat rekor baru (is_best), False jika tidak.
        """
        if self.patience <= 0:
            if current_score > self.best_score:
                self.best_score = current_score
                self.best_epoch = epoch
                return True
            return False

        improved = False
        if self.mode == "max":
            if current_score > (self.best_score + self.min_delta):
                improved = True
        else:
            if current_score < (self.best_score - self.min_delta):
                improved = True

        if improved:
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

# ============================================================
# 7. TRAINING & EVALUATION (DENGAN AKSELERASI AMP MIXED PRECISION)
# ============================================================
def train_epoch(model, loader, criterion, optimizer, scaler, device, amp_dtype=torch.bfloat16):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    use_amp = (device.type == "cuda")

    pbar = tqdm(loader, desc="Training Batch (ConvNeXt-Small + ViT-Base)", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
            out = model(images)
            logits = out["logits"]
            loss = criterion(logits, labels)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
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
def evaluate_epoch(model, loader, criterion, device, amp_dtype=torch.bfloat16):
    model.eval()
    total_loss = 0.0
    all_targets, all_scores = [], []
    use_amp = (device.type == "cuda")

    for images, labels in tqdm(loader, desc="Eval Batch", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
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
# 8. MAIN PIPELINE
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Training Hybrid Token S-Adapter (ConvNeXt-Small + ViT-Base)")
    parser.add_argument("--data_dir", type=str, default="./CelebA_Spoof")
    parser.add_argument("--samples", type=int, default=0, help="Gunakan sample subset (0 = Full dataset)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr_adapter", type=float, default=1e-4)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--vit_name", type=str, default="vit_base_patch16_224", help="Backbone ViT (default: vit_base_patch16_224)")
    parser.add_argument("--convnext_name", type=str, default="convnext_small", help="Backbone ConvNeXt (default: convnext_small)")
    parser.add_argument("--adapter_bottleneck", type=int, default=64, help="Bottleneck dimensi S-Adapter (tetap 64)")
    parser.add_argument("--num_heads", type=int, default=12, help="Attention heads (default: 12)")
    parser.add_argument("--patience", type=int, default=5, help="Jumlah epoch toleransi Early Stopping (default: 5)")
    parser.add_argument("--min_delta", type=float, default=0.001, help="Peningkatan minimal AUC (default: 0.001)")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_c_small_v_base", help="Folder penyimpanan")
    parser.add_argument("--workers", type=int, default=4 if os.name != 'nt' else 0, help="Jumlah worker CPU DataLoader (default: 4 di Linux, hemat RAM)")
    args = parser.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "="*75)
    print("MEMBANGUN MANIFEST DATASET CELEBA-SPOOF (ConvNeXt-Small + ViT-Base)")
    print("="*75)
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

    print("\n" + "="*75)
    print("INISIALISASI MODEL HYBRID TOKEN S-ADAPTER (ConvNeXt-Small + ViT-Base)")
    print("="*75)
    print(f"Backbone 1 : ConvNeXt-Small (Stage 4 C4=768 -> TSR -> Tc 49x768)")
    print(f"Backbone 2 : ViT-Base (Patch 16x16 -> 196 tokens -> S-Adapter -> Tv 196x768)")
    print(f"Fusion     : Bidirectional Cross-Attention (12 Heads) -> Tr 245x768 -> Refinement")
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
    frozen_params = total_params - trainable_params
    trainable_pct = 100 * trainable_params / total_params
    print(f"Total Parameter      : {total_params:,}")
    print(f"Trainable Parameter  : {trainable_params:,} ({trainable_pct:.2f}%)")
    print(f"Frozen Parameter     : {frozen_params:,} ({100 - trainable_pct:.2f}%)")

    # Akselerasi Mixed Precision (bfloat16 Hopper / float16)
    use_cuda = (DEVICE.type == "cuda")
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(use_cuda and not use_bf16))

    if use_cuda:
        precision_type = "BFloat16 (Native Hopper Tensor Cores)" if use_bf16 else "Float16 (GradScaler)"
        print(f"[AKSELERASI H200] Mixed Precision AKTIF ({precision_type})")
        print(f"[AKSELERASI H200] Batch Size: {args.batch_size} | DataLoader Workers: {args.workers}")
    else:
        print("[AKSELERASI GPU] CPU Mode (Mixed Precision Nonaktif)")

    # [POINT 1]: Logger & Konfigurasi
    logger = TrainingLogger(save_dir=args.save_dir, experiment_name="HybridToken_CSmall_VBase")
    dataset_stats = {
        "data_dir": args.data_dir,
        "total_manifest": len(df),
        "samples_mode": args.samples if args.samples > 0 else "Full",
        "train_samples": len(df_train),
        "val_samples": len(df_val),
        "test_samples": len(df_test),
        "device": str(DEVICE),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None (CPU)",
        "mixed_precision": "bfloat16" if use_bf16 else ("float16" if use_cuda else "fp32"),
        "workers": args.workers
    }
    model_stats = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params,
        "trainable_percentage": f"{trainable_pct:.2f}%"
    }
    logger.log_config(vars(args), model_stats, dataset_stats)

    criterion = FocalLoss(gamma=2.0, alpha=0.25)
    optimizer = get_optimizer(model, lr_adapter=args.lr_adapter, lr_head=args.lr_head)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_auc = 0.0
    best_epoch = 1
    training_start_time = time.time()
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_auc": [], "val_hter": []}

    # [POINT 2]: Early Stopping
    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta, mode="max")
    if args.patience > 0:
        print(f"[EARLY STOPPING AKTIF] Monitoring: val_auc | Patience: {args.patience} epoch | Min Delta: {args.min_delta}")
    else:
        print("[EARLY STOPPING NONAKTIF] Model akan dilatih penuh hingga batas epoch.")

    print("\n" + "="*75)
    print("MEMULAI PROSES TRAINING (ConvNeXt-Small + ViT-Base)")
    print("="*75)
    best_ckpt_path = Path(args.save_dir) / "best_token_hybrid_c_small_v_base.pth"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE, amp_dtype)
        val_metrics, _, _ = evaluate_epoch(model, val_loader, criterion, DEVICE, amp_dtype)

        lr_adapter = optimizer.param_groups[0]['lr']
        lr_head = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else lr_adapter
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_auc"].append(val_metrics["auc"])
        history["val_hter"].append(val_metrics["hter"])

        is_best = early_stopping.check(val_metrics["auc"], epoch)
        if is_best:
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": best_val_auc,
                "val_acc": val_metrics["accuracy"],
                "val_hter": val_metrics["hter"],
                "arch": "HybridToken_CSmall_VBase"
            }, best_ckpt_path)
            print(f"  -> Checkpoint terbaik disimpan di: {best_ckpt_path} (Rekor Baru AUC: {best_val_auc:.4f})")
        else:
            if args.patience > 0:
                print(f"  -> [EarlyStopping] Tidak ada peningkatan selama {early_stopping.counter}/{args.patience} epoch (Rekor: {early_stopping.best_score:.4f} di Epoch {early_stopping.best_epoch})")

        logger.log_epoch(
            epoch=epoch, elapsed=elapsed, lr_adapter=lr_adapter, lr_head=lr_head,
            train_loss=train_loss, train_acc=train_acc, val_metrics=val_metrics, is_best=is_best
        )

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']*100:.2f}% "
            f"AUC: {val_metrics['auc']:.4f} HTER: {val_metrics['hter']*100:.2f}%"
        )

        if early_stopping.early_stop:
            print("\n" + "="*75)
            print(f"[EARLY STOPPING TRIGGERED] Training dihentikan otomatis pada Epoch {epoch}!")
            print(f"Alasan : Tidak ada peningkatan AUC lebih dari {args.min_delta} selama {args.patience} epoch berturut-turut.")
            print(f"Model  : Checkpoint terbaik yang dimuat adalah Epoch {early_stopping.best_epoch} (AUC: {early_stopping.best_score:.4f}).")
            print("="*75)
            break

    # Evaluasi Final pada Test Set
    print("\n" + "="*75)
    print("EVALUASI FINAL (ConvNeXt-Small + ViT-Base) PADA TEST SET")
    print("="*75)
    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Checkpoint terbaik (Epoch {ckpt['epoch']}) dimuat.")

    test_metrics, _, _ = evaluate_epoch(model, test_loader, criterion, DEVICE, amp_dtype)
    print("\n--- HASIL EVALUASI TEST SET (ConvNeXt-Small + ViT-Base) ---")
    print(f"Accuracy : {test_metrics['accuracy']*100:.2f}%")
    print(f"AUC      : {test_metrics['auc']:.4f}")
    print(f"APCER    : {test_metrics['apcer']*100:.2f}%")
    print(f"BPCER    : {test_metrics['bpcer']*100:.2f}%")
    print(f"HTER     : {test_metrics['hter']*100:.2f}%")

    total_training_duration = time.time() - training_start_time
    logger.log_final_results(
        best_epoch=best_epoch, best_val_auc=best_val_auc, test_metrics=test_metrics,
        total_duration=total_training_duration, early_stopped=early_stopping.early_stop, stopped_epoch=epoch
    )

    # Simpan Grafik Training
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(history["train_loss"], label="Train Loss", color="blue")
    axes[0].plot(history["val_loss"], label="Val Loss", color="red")
    axes[0].set_title("ConvNeXt-Small + ViT-Base: Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train Acc", color="blue")
    axes[1].plot(history["val_acc"], label="Val Acc", color="green")
    axes[1].set_title("ConvNeXt-Small + ViT-Base: Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(history["val_auc"], label="Val AUC", color="purple")
    axes[2].plot(history["val_hter"], label="Val HTER", color="orange")
    axes[2].set_title("ConvNeXt-Small + ViT-Base: AUC & HTER")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    plt.tight_layout()
    chart_ckpt_path = Path(args.save_dir) / "training_history_c_small_v_base.png"
    plt.savefig(chart_ckpt_path, dpi=150)
    plt.savefig("./training_history_c_small_v_base.png", dpi=150)
    plt.close()
    print(f"[SUCCESS] Grafik training disimpan di: {chart_ckpt_path} dan ./training_history_c_small_v_base.png")

if __name__ == "__main__":
    main()
