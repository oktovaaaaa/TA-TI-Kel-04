"""
Pembuatan Split Data Resmi CelebA-Spoof (Per Subjek)
====================================================

Aturan pembagian (disepakati dengan dosen pembimbing):
1. Sumber data & label  : file anotasi resmi `metas/intra_test/train_label.txt` dan
                          `test_label.txt` (BUKAN dari nama folder).
2. Test                 : seluruh data test resmi CelebA-Spoof, utuh (67.170 citra).
3. Validation           : sebagian SUBJEK (orang) dari data train resmi, dipilih acak
                          (seed tetap) sampai jumlah citranya ~ sama dengan test.
4. Train                : sisa subjek dari data train resmi.
5. Subjek tumpang tindih: subjek yang muncul di train resmi DAN test resmi dibuang
                          dari sisi train, agar tidak ada wajah yang sama antar bagian.

Konvensi label proyek : 1 = live, 0 = spoof
Konvensi label resmi  : 0 = live, 1 = spoof  (dibalik saat dibaca)

Output (folder ./split):
- split.csv                      : daftar seluruh citra + bagian (train/val/test)
- ringkasan_split.csv / .json    : tabel jumlah subjek, citra, live, spoof per bagian
- citra_dibuang_overlap.csv      : citra train dari subjek yang juga ada di test
- label_folder_vs_resmi.csv      : citra yang label foldernya berbeda dari label resmi

Cara pakai:
    python eksperimen_split_resmi/buat_split.py --data_dir ./CelebA_Spoof
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42


def baca_label_resmi(data_dir: Path, nama_file: str, split_resmi: str) -> pd.DataFrame:
    path = data_dir / "metas" / "intra_test" / nama_file
    if not path.exists():
        raise FileNotFoundError(f"File label resmi tidak ditemukan: {path}")
    df = pd.read_csv(path, sep=" ", header=None, names=["path", "label_resmi"], dtype={"path": str})
    # Path contoh: Data/train/4154/live/001021.jpg
    bagian = df["path"].str.split("/")
    df["subject"] = bagian.str[2].astype(int)
    df["folder"] = bagian.str[3]
    df["split_resmi"] = split_resmi
    # Balik ke konvensi proyek: 1 = live, 0 = spoof
    df["label"] = (df["label_resmi"] == 0).astype(int)
    df["label_name"] = np.where(df["label"] == 1, "live", "spoof")
    return df


def ringkas(df: pd.DataFrame, total_citra: int, total_subjek: int) -> list:
    baris = []
    for nama in ["train", "val", "test"]:
        d = df[df["split"] == nama]
        baris.append({
            "bagian": nama,
            "jumlah_subjek": int(d["subject"].nunique()),
            "persen_subjek": round(100 * d["subject"].nunique() / total_subjek, 2),
            "jumlah_citra": int(len(d)),
            "persen_citra": round(100 * len(d) / total_citra, 2),
            "jumlah_live": int((d["label"] == 1).sum()),
            "jumlah_spoof": int((d["label"] == 0).sum()),
            "rasio_live_persen": round(100 * (d["label"] == 1).mean(), 2) if len(d) else 0.0,
        })
    baris.append({
        "bagian": "total",
        "jumlah_subjek": int(df["subject"].nunique()),
        "persen_subjek": 100.0,
        "jumlah_citra": int(len(df)),
        "persen_citra": 100.0,
        "jumlah_live": int((df["label"] == 1).sum()),
        "jumlah_spoof": int((df["label"] == 0).sum()),
        "rasio_live_persen": round(100 * (df["label"] == 1).mean(), 2),
    })
    return baris


def main():
    parser = argparse.ArgumentParser(description="Buat split resmi CelebA-Spoof per subjek")
    parser.add_argument("--data_dir", type=str, default="./CelebA_Spoof")
    parser.add_argument("--out_dir", type=str, default=str(Path(__file__).parent / "split"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cek_file", action="store_true",
                        help="Cek bahwa setiap file gambar di split benar-benar ada di disk (lebih lama)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("MEMBUAT SPLIT RESMI CELEBA-SPOOF (PER SUBJEK)")
    print("=" * 75)

    train_resmi = baca_label_resmi(data_dir, "train_label.txt", "train")
    test_resmi = baca_label_resmi(data_dir, "test_label.txt", "test")
    print(f"[1] Label resmi dibaca : train={len(train_resmi):,} citra, test={len(test_resmi):,} citra")

    # --- Laporan ketidakcocokan label folder vs label resmi ---
    semua = pd.concat([train_resmi, test_resmi], ignore_index=True)
    beda = semua[semua["folder"] != semua["label_name"]]
    beda[["path", "subject", "split_resmi", "folder", "label_name"]].rename(
        columns={"label_name": "label_resmi_dipakai"}
    ).to_csv(out_dir / "label_folder_vs_resmi.csv", index=False)
    print(f"[2] Citra dengan label folder != label resmi : {len(beda):,} (memakai label resmi)")

    # --- Subjek tumpang tindih train resmi & test resmi ---
    subjek_overlap = sorted(set(train_resmi["subject"]) & set(test_resmi["subject"]))
    dibuang = train_resmi[train_resmi["subject"].isin(subjek_overlap)]
    dibuang[["path", "subject", "label_name"]].to_csv(out_dir / "citra_dibuang_overlap.csv", index=False)
    train_bersih = train_resmi[~train_resmi["subject"].isin(subjek_overlap)].copy()
    print(f"[3] Subjek ada di train & test resmi : {subjek_overlap} "
          f"-> {len(dibuang):,} citra train dibuang")

    # --- Pilih subjek validation dari train resmi sampai jumlah citra ~ test ---
    target_val = len(test_resmi)
    jumlah_per_subjek = train_bersih.groupby("subject").size().sort_index()
    rng = np.random.default_rng(args.seed)
    urutan = rng.permutation(jumlah_per_subjek.index.to_numpy())

    subjek_val, terkumpul = [], 0
    for sid in urutan:
        if terkumpul >= target_val:
            break
        subjek_val.append(int(sid))
        terkumpul += int(jumlah_per_subjek[sid])
    subjek_val = set(subjek_val)

    train_bersih["split"] = np.where(train_bersih["subject"].isin(subjek_val), "val", "train")
    test_resmi["split"] = "test"
    df = pd.concat([train_bersih, test_resmi], ignore_index=True)
    print(f"[4] Validation dipilih : {len(subjek_val):,} subjek, {terkumpul:,} citra "
          f"(target ~{target_val:,} = ukuran test)")

    # --- Verifikasi: tidak ada subjek yang sama antar bagian ---
    s_tr = set(df.loc[df.split == "train", "subject"])
    s_va = set(df.loc[df.split == "val", "subject"])
    s_te = set(df.loc[df.split == "test", "subject"])
    assert not (s_tr & s_va), "Ada subjek di train & val!"
    assert not (s_tr & s_te), "Ada subjek di train & test!"
    assert not (s_va & s_te), "Ada subjek di val & test!"
    assert df["path"].is_unique, "Ada path ganda!"
    print("[5] Verifikasi: TIDAK ada subjek yang sama antar train / val / test  ✔")

    if args.cek_file:
        hilang = [p for p in df["path"] if not (data_dir / p).exists()]
        print(f"[6] Cek file di disk : {len(hilang):,} file tidak ditemukan")
        if hilang:
            pd.Series(hilang, name="path").to_csv(out_dir / "file_tidak_ditemukan.csv", index=False)

    # --- Simpan ---
    kolom = ["path", "subject", "label", "label_name", "folder", "split_resmi", "split"]
    df = df[kolom].sort_values(["split", "subject", "path"]).reset_index(drop=True)
    df.to_csv(out_dir / "split.csv", index=False)

    total_citra_resmi = len(train_resmi) + len(test_resmi)
    tabel = ringkas(df, total_citra=len(df), total_subjek=df["subject"].nunique())
    pd.DataFrame(tabel).to_csv(out_dir / "ringkasan_split.csv", index=False)
    with open(out_dir / "ringkasan_split.json", "w", encoding="utf-8") as f:
        json.dump({
            "seed": args.seed,
            "sumber_label": "metas/intra_test/train_label.txt & test_label.txt (resmi)",
            "konvensi_label": "1 = live, 0 = spoof",
            "total_citra_resmi": total_citra_resmi,
            "citra_dibuang_overlap": int(len(dibuang)),
            "subjek_overlap_dibuang_dari_train": subjek_overlap,
            "citra_label_folder_berbeda_dari_resmi": int(len(beda)),
            "tabel": tabel,
        }, f, indent=4)

    print("\n" + "-" * 75)
    print(f"{'Bagian':<8}{'Subjek':>8}{'%Subj':>8}{'Citra':>10}{'%Citra':>8}{'Live':>9}{'Spoof':>9}")
    for r in tabel:
        print(f"{r['bagian']:<8}{r['jumlah_subjek']:>8,}{r['persen_subjek']:>7.2f}%"
              f"{r['jumlah_citra']:>10,}{r['persen_citra']:>7.2f}%"
              f"{r['jumlah_live']:>9,}{r['jumlah_spoof']:>9,}")
    print("-" * 75)
    print(f"Disimpan di: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
