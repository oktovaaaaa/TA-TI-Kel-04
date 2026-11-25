"""
Training & Evaluasi dengan Split Resmi CelebA-Spoof (Per Subjek)
================================================================

Model yang didukung:
  --model c_tiny_v_tiny (atau h1)   : Hybrid ConvNeXt-Tiny + ViT-Tiny (H1)
  --model c_tiny_v_small (atau h2)  : Hybrid ConvNeXt-Tiny + ViT-Small (H2)
  --model c_tiny_v_base (atau h3)   : Hybrid ConvNeXt-Tiny + ViT-Base (H3)
  --model c_small_v_tiny (atau h4)  : Hybrid ConvNeXt-Small + ViT-Tiny (H4)
  --model hybrid / c_small_v_small (h5) : Hybrid ConvNeXt-Small + ViT-Small (H5)
  --model c_small_v_base (atau h6)  : Hybrid ConvNeXt-Small + ViT-Base (H6)
  --model convnext (atau b2)        : ConvNeXt-Small + S-Adapter (B2)
Mode:
  --mode freeze    : backbone dibekukan (PEFT: S-Adapter + ConvNeXt stage 4 + head dilatih)
  --mode fullft    : full fine-tuning, seluruh parameter dilatih

Yang dicatat SETIAP EPOCH (training_history.csv / .json):
  - waktu train, validation, test (detik) + throughput (citra/detik) + total waktu epoch
  - learning rate tiap kelompok parameter
  - metrik lengkap train / val / test: loss, accuracy, balanced accuracy, precision,
    recall, F1 (live, spoof, macro), APCER, BPCER, HTER, AUC, EER, TP/TN/FP/FN
  - skor komposit, status checkpoint terbaik, penghitung early stopping, memori GPU puncak

Catatan metodologi:
  Test set dievaluasi tiap epoch HANYA untuk analisis (kurva). Pemilihan checkpoint
  terbaik dan early stopping memakai VALIDATION saja. Hasil akhir yang dilaporkan
  adalah test set pada checkpoint terbaik (final_summary.json).

Contoh:
  python eksperimen_split_resmi/train.py --model hybrid   --mode freeze
  python eksperimen_split_resmi/train.py --model hybrid   --mode fullft
  python eksperimen_split_resmi/train.py --model convnext --mode freeze
  python eksperimen_split_resmi/train.py --model convnext --mode fullft
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import random
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    SplitDataset, train_transform, eval_transform, make_weighted_sampler,
    hitung_metrik, buat_model, buat_optimizer, statistik_parameter, FocalLoss,
    plot_confusion_matrix, plot_roc, plot_history, KONFIG_MODEL,
)

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent


class Tee:
    """Menulis output terminal sekaligus ke file log."""

    def __init__(self, path):
        self.file = open(path, "a", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sinkron(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def fmt_waktu(detik):
    detik = int(detik)
    return f"{detik // 3600:02d}:{(detik % 3600) // 60:02d}:{detik % 60:02d}"


class EarlyStopping:
    """Skor komposit = AUC + Accuracy - HTER - Loss (SAMA dengan 15 model lama)."""

    def __init__(self, patience=5, min_delta=0.0005):
        self.patience, self.min_delta = patience, min_delta
        self.counter = 0
        self.best_score = -float("inf")
        self.best_ckpt_score = -float("inf")
        self.best_epoch = 0
        self.early_stop = False

    @staticmethod
    def skor(m):
        return m["auc"] + m["accuracy"] - m["hter"] - m["loss"]

    def check(self, m, epoch):
        s = self.skor(m)
        is_best = s > self.best_ckpt_score
        if is_best:
            self.best_ckpt_score, self.best_epoch = s, epoch
        if self.patience > 0:
            if s > self.best_score + self.min_delta:
                self.best_score, self.counter = s, 0
            else:
                self.counter += 1
                self.early_stop = self.counter >= self.patience
        return is_best, s

    def state_dict(self):
        return dict(self.__dict__)

    def load_state_dict(self, d):
        self.__dict__.update(d)


def jalankan_epoch(model, loader, criterion, device, amp_dtype, optimizer=None, desc=""):
    """Satu putaran train (jika optimizer diberikan) atau evaluasi. Mengembalikan metrik + prediksi."""
    latih = optimizer is not None
    model.train(latih)
    use_amp = device.type == "cuda"
    total_loss, n = 0.0, 0
    idx_all, y_all, s_all = [], [], []

    sinkron(device)
    t0 = time.time()
    ctx = torch.enable_grad() if latih else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
        for images, labels, idx in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(images)["logits"]
                loss = criterion(logits.float(), labels)
            if latih:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            bs = labels.size(0)
            total_loss += loss.item() * bs
            n += bs
            prob_live = torch.softmax(logits.float().detach(), dim=-1)[:, 1]
            idx_all.append(idx.numpy())
            y_all.append(labels.cpu().numpy())
            s_all.append(prob_live.cpu().numpy())
            pbar.set_postfix(loss=f"{total_loss / n:.4f}")
    sinkron(device)
    durasi = time.time() - t0

    y_all, s_all, idx_all = np.concatenate(y_all), np.concatenate(s_all), np.concatenate(idx_all)
    m = hitung_metrik(y_all, s_all)
    m["loss"] = total_loss / max(n, 1)
    m["time_sec"] = durasi
    m["throughput_img_per_sec"] = n / durasi if durasi > 0 else 0.0
    return m, idx_all, y_all, s_all


def simpan_prediksi(df, idx, y, s, path):
    out = df.iloc[idx][["path", "subject", "label_name"]].copy()
    out["y_true"] = y
    out["prob_live"] = np.round(s, 6)
    out["y_pred"] = (s >= 0.5).astype(int)
    out["pred_name"] = np.where(out["y_pred"] == 1, "live", "spoof")
    out["benar"] = (out["y_true"] == out["y_pred"]).astype(int)
    out.to_csv(path, index=False)


def bulatkan(m):
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items()}


def main():
    p = argparse.ArgumentParser(description="Training split resmi CelebA-Spoof")
    p.add_argument("--model", choices=list(KONFIG_MODEL.keys()), required=True)
    p.add_argument("--mode", choices=["freeze", "fullft"], required=True)
    p.add_argument("--data_dir", default="./CelebA_Spoof")
    p.add_argument("--split_csv", default=str(HERE / "split" / "split.csv"))
    p.add_argument("--out_dir", default=None, help="Default: eksperimen_split_resmi/hasil/<model>_<mode>")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr_backbone", type=float, default=1e-5, help="Hanya dipakai saat --mode fullft")
    p.add_argument("--lr_adapter", type=float, default=1e-4)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5, help="0 = early stopping nonaktif")
    p.add_argument("--min_delta", type=float, default=0.0005)
    p.add_argument("--workers", type=int, default=8 if os.name != "nt" else 0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_test_each_epoch", action="store_true", help="Lewati evaluasi test tiap epoch")
    p.add_argument("--no_pretrained", action="store_true", help="Hanya untuk uji cepat tanpa internet")
    p.add_argument("--max_per_split", type=int, default=0, help="Uji cepat: batasi jumlah citra per bagian")
    p.add_argument("--resume", action="store_true", help="Lanjutkan dari last_checkpoint.pth")
    args = p.parse_args()

    nama_run = f"{args.model}_{args.mode}"
    out_dir = Path(args.out_dir) if args.out_dir else HERE / "hasil" / nama_run
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(out_dir / "log_training.txt")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    print("\n" + "=" * 80)
    print(f"EKSPERIMEN : {nama_run}  |  {KONFIG_MODEL[args.model]}  |  mode={args.mode}")
    print(f"Mulai      : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 80)

    # ---------------- DATA ----------------
    split_path = Path(args.split_csv)
    if not split_path.exists():
        sys.exit(f"split.csv tidak ditemukan di {split_path}. Jalankan dulu buat_split.py")
    df = pd.read_csv(split_path)
    bagian = {s: df[df.split == s].reset_index(drop=True) for s in ["train", "val", "test"]}
    if args.max_per_split > 0:
        bagian = {s: d.sample(n=min(args.max_per_split, len(d)), random_state=args.seed).reset_index(drop=True)
                  for s, d in bagian.items()}
        print(f"[UJI CEPAT] dibatasi {args.max_per_split} citra per bagian")

    # Verifikasi ulang: tidak ada subjek yang sama antar bagian
    subj = {s: set(d.subject) for s, d in bagian.items()}
    assert not (subj["train"] & subj["val"]) and not (subj["train"] & subj["test"]) \
        and not (subj["val"] & subj["test"]), "Ada subjek yang sama antar bagian!"

    for s, d in bagian.items():
        print(f"  {s:<5}: {len(d):>7,} citra | {d.subject.nunique():>5,} subjek | "
              f"live {int((d.label == 1).sum()):>7,} | spoof {int((d.label == 0).sum()):>7,}")

    pin = device.type == "cuda"
    kw = dict(num_workers=args.workers, pin_memory=pin,
              persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers > 0 else None)
    ds = {
        "train": SplitDataset(bagian["train"], args.data_dir, train_transform),
        "val": SplitDataset(bagian["val"], args.data_dir, eval_transform),
        "test": SplitDataset(bagian["test"], args.data_dir, eval_transform),
    }
    loader = {
        "train": DataLoader(ds["train"], batch_size=args.batch_size,
                            sampler=make_weighted_sampler(bagian["train"], args.seed), **kw),
        "val": DataLoader(ds["val"], batch_size=args.batch_size * 2, shuffle=False, **kw),
        "test": DataLoader(ds["test"], batch_size=args.batch_size * 2, shuffle=False, **kw),
    }

    # ---------------- MODEL ----------------
    model = buat_model(args.model, args.mode, pretrained=not args.no_pretrained).to(device)
    stat = statistik_parameter(model)
    optimizer, jumlah_grup = buat_optimizer(model, args.lr_backbone, args.lr_adapter, args.lr_head)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = FocalLoss(gamma=2.0, alpha=0.25)
    es = EarlyStopping(args.patience, args.min_delta)

    print(f"\nParameter total     : {stat['total_parameters']:,}")
    print(f"Parameter dilatih   : {stat['trainable_parameters']:,} ({stat['trainable_percentage']}%)")
    print(f"Parameter dibekukan : {stat['frozen_parameters']:,}")
    for g in optimizer.param_groups:
        print(f"  grup {g['name']:<8}: {jumlah_grup[g['name']]:>12,} parameter | lr awal {g['lr']:.0e}")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}) | "
          f"presisi: {'bf16' if use_bf16 else ('fp16' if device.type == 'cuda' else 'fp32')}")

    config = {
        "eksperimen": nama_run,
        "deskripsi_model": KONFIG_MODEL[args.model],
        "mode": args.mode,
        "args": vars(args),
        "parameter": stat,
        "parameter_per_grup_lr": jumlah_grup,
        "data": {s: {"citra": len(d), "subjek": int(d.subject.nunique()),
                     "live": int((d.label == 1).sum()), "spoof": int((d.label == 0).sum())}
                 for s, d in bagian.items()},
        "hyperparameter_tetap": {
            "optimizer": "AdamW", "weight_decay": {"backbone": 1e-4, "adapter": 1e-4, "head": 1e-2},
            "scheduler": f"CosineAnnealingLR(T_max={args.epochs}, eta_min=1e-6)",
            "loss": "FocalLoss(gamma=2.0, alpha=0.25)", "grad_clip": 1.0,
            "sampler": "WeightedRandomSampler (seimbang live/spoof)",
            "img_size": 224, "early_stopping_metric": "AUC + Accuracy - HTER - Loss (validation)",
        },
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    # ---------------- RESUME ----------------
    last_ckpt, best_ckpt = out_dir / "last_checkpoint.pth", out_dir / "best_checkpoint.pth"
    history, start_epoch, total_waktu_sebelumnya = [], 1, 0.0
    if args.resume and last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        es.load_state_dict(ck["early_stopping"])
        history = ck["history"]
        start_epoch = ck["epoch"] + 1
        total_waktu_sebelumnya = ck.get("total_waktu", 0.0)
        print(f"[RESUME] melanjutkan dari epoch {start_epoch}")

    # ---------------- TRAINING ----------------
    t_mulai = time.time()
    epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t_epoch = time.time()
        lr_now = {g["name"]: g["lr"] for g in optimizer.param_groups}

        m_tr, *_ = jalankan_epoch(model, loader["train"], criterion, device, amp_dtype,
                                  optimizer=optimizer, desc=f"Epoch {epoch} train")
        m_va, *_ = jalankan_epoch(model, loader["val"], criterion, device, amp_dtype, desc=f"Epoch {epoch} val")
        m_te = None
        if not args.no_test_each_epoch:
            m_te, *_ = jalankan_epoch(model, loader["test"], criterion, device, amp_dtype,
                                      desc=f"Epoch {epoch} test")
        scheduler.step()

        is_best, skor = es.check(m_va, epoch)
        if is_best:
            torch.save({"epoch": epoch, "model": model.state_dict(), "val_metrics": m_va,
                        "composite_score": skor, "config": config}, best_ckpt)

        durasi_epoch = time.time() - t_epoch
        rec = {"epoch": epoch, "epoch_time_sec": round(durasi_epoch, 2)}
        for k, v in lr_now.items():
            rec[f"lr_{k}"] = v
        for nama, m in [("train", m_tr), ("val", m_va), ("test", m_te)]:
            if m is None:
                continue
            for k, v in m.items():
                rec[f"{nama}_{k}"] = round(v, 6) if isinstance(v, float) else v
        rec.update({
            "composite_score": round(skor, 6), "is_best": int(is_best),
            "best_epoch_so_far": es.best_epoch, "early_stop_counter": es.counter,
            "gpu_peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else None,
        })
        history.append(rec)
        hist_df = pd.DataFrame(history)
        hist_df.to_csv(out_dir / "training_history.csv", index=False)
        with open(out_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        total_waktu = total_waktu_sebelumnya + (time.time() - t_mulai)
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "early_stopping": es.state_dict(),
                    "history": history, "total_waktu": total_waktu}, last_ckpt)

        sisa = (args.epochs - epoch) * (total_waktu / max(epoch - start_epoch + 1, 1))
        teks_test = (f" | TEST acc {m_te['accuracy'] * 100:.2f}% auc {m_te['auc']:.4f} "
                     f"hter {m_te['hter'] * 100:.2f}% ({m_te['time_sec']:.0f}s)") if m_te else ""
        print(f"Epoch {epoch:02d}/{args.epochs} [{fmt_waktu(durasi_epoch)}] "
              f"TRAIN loss {m_tr['loss']:.4f} acc {m_tr['accuracy'] * 100:.2f}% ({m_tr['time_sec']:.0f}s) | "
              f"VAL loss {m_va['loss']:.4f} acc {m_va['accuracy'] * 100:.2f}% auc {m_va['auc']:.4f} "
              f"hter {m_va['hter'] * 100:.2f}% ({m_va['time_sec']:.0f}s){teks_test} | "
              f"skor {skor:.4f}{' *BEST*' if is_best else ''} | ES {es.counter}/{args.patience} | "
              f"estimasi sisa maks {fmt_waktu(sisa)}")

        if es.early_stop:
            print(f"[EARLY STOPPING] berhenti di epoch {epoch}; checkpoint terbaik = epoch {es.best_epoch}")
            break

    total_training = total_waktu_sebelumnya + (time.time() - t_mulai)

    # ---------------- EVALUASI AKHIR (CHECKPOINT TERBAIK) ----------------
    print("\n" + "=" * 80)
    print(f"EVALUASI AKHIR memakai checkpoint terbaik (epoch {es.best_epoch})")
    print("=" * 80)
    ck = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])

    hasil_akhir = {}
    for nama in ["val", "test"]:
        m, idx, y, s = jalankan_epoch(model, loader[nama], criterion, device, amp_dtype, desc=f"final {nama}")
        simpan_prediksi(bagian[nama], idx, y, s, out_dir / f"predictions_{nama}.csv")
        judul = f"{nama_run} - {nama.upper()} (epoch terbaik {es.best_epoch})"
        plot_confusion_matrix(m, judul, out_dir / f"confusion_matrix_{nama}.png")
        plot_roc(y, s, m, judul, out_dir / f"roc_curve_{nama}.png")
        hasil_akhir[nama] = bulatkan(m)
        print(f"{nama.upper():<5} acc {m['accuracy'] * 100:.2f}% | AUC {m['auc']:.4f} | EER {m['eer'] * 100:.2f}% | "
              f"APCER {m['apcer'] * 100:.2f}% | BPCER {m['bpcer'] * 100:.2f}% | HTER {m['hter'] * 100:.2f}% | "
              f"F1-macro {m['f1_macro'] * 100:.2f}% | TP {m['tp']:,} TN {m['tn']:,} FP {m['fp']:,} FN {m['fn']:,} "
              f"| waktu {m['time_sec']:.1f}s ({m['throughput_img_per_sec']:.0f} citra/s)")

    hist_df = pd.DataFrame(history)
    plot_history(hist_df, f"Riwayat training {nama_run}", out_dir / "training_history.png")

    gagal = sum(d.n_gagal for d in ds.values())
    ringkasan = {
        "eksperimen": nama_run,
        "deskripsi_model": KONFIG_MODEL[args.model],
        "mode": args.mode,
        "selesai": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        "parameter": stat,
        "epoch_dijalankan": int(epoch),
        "epoch_terbaik": int(es.best_epoch),
        "early_stopping_terpicu": bool(es.early_stop),
        "waktu": {
            "total_training": fmt_waktu(total_training),
            "total_training_detik": round(total_training, 1),
            "rata2_train_per_epoch_detik": round(hist_df["train_time_sec"].mean(), 1),
            "rata2_val_per_epoch_detik": round(hist_df["val_time_sec"].mean(), 1),
            "rata2_test_per_epoch_detik": round(hist_df["test_time_sec"].mean(), 1) if "test_time_sec" in hist_df else None,
            "rata2_total_per_epoch_detik": round(hist_df["epoch_time_sec"].mean(), 1),
        },
        "hasil_validation_epoch_terbaik": hasil_akhir["val"],
        "hasil_test_epoch_terbaik": hasil_akhir["test"],
        "citra_gagal_dibaca": int(gagal),
        "catatan": "Checkpoint terbaik dipilih berdasarkan VALIDATION. Hasil test di atas adalah hasil resmi.",
    }
    with open(out_dir / "final_summary.json", "w", encoding="utf-8") as f:
        json.dump(ringkasan, f, indent=4)

    print(f"\nTotal waktu training : {fmt_waktu(total_training)}")
    if gagal:
        print(f"[PERINGATAN] {gagal} citra gagal dibaca (diganti citra hitam) - cek path dataset")
    print(f"Semua hasil disimpan di: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
