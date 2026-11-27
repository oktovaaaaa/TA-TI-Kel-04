"""
Real-Time Face Anti-Spoofing dengan Google MediaPipe Face Tracking
==================================================================
Kombinasi SOTA Face Detection (Google MediaPipe BlazeFace) dengan
Model Dual-Branch Hybrid Token S-Adapter (ConvNeXt + Vision Transformer).

Fitur:
- Pelacakan wajah akurat, stabil, dan bebas getar (60+ FPS di CPU).
- Otomatis memotong wajah (ROI) dengan margin proporsional.
- Mendukung otomatis 9 varian model (Tiny, Small, Base).
- Tampilan UI informatif: Bounding box warna (Hijau: LIVE, Merah: SPOOF),
  Confidence Bar, FPS counter, dan label persentase real-time.

Penggunaan:
  python test_webcam_mediapipe.py
  python test_webcam_mediapipe.py --checkpoint checkpoints_c_small_v_small/best_token_hybrid_c_small_v_small.pth
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import time
import argparse
import urllib.request
from pathlib import Path
from PIL import Image

# Tambahkan root proyek ke sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent if (Path(__file__).resolve().parent.name == "evaluation") else Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from models.hybrid_convnext_vit_s_adapter import HybridTokenSAdapterConvNeXt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ─────────────────────────────────────────────────────────────
# 1. INISIALISASI MEDIAPIPE FACE DETECTOR (GOOGLE BLAZEFACE)
# ─────────────────────────────────────────────────────────────
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
MODEL_LOCAL = Path(__file__).resolve().parent / "blaze_face_short_range.tflite"

def get_mediapipe_detector():
    """Menginisialisasi Google MediaPipe Face Detector."""
    if not MODEL_LOCAL.exists():
        print(f"[Info] Mengunduh model Google BlazeFace ({MODEL_LOCAL.name})...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_LOCAL)
            print("[Success] Model BlazeFace berhasil diunduh!")
        except Exception as e:
            print(f"[Warning] Gagal mengunduh BlazeFace ({e}). Fallback ke Haar Cascade...")
            return None

    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(model_asset_path=str(MODEL_LOCAL))
        options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5)
        detector = vision.FaceDetector.create_from_options(options)
        print("[Success] Google MediaPipe Face Detector aktif (Tracking Akurat).")
        return detector
    except Exception as e:
        print(f"[Warning] Inisialisasi MediaPipe gagal ({e}). Fallback ke Haar Cascade...")
        return None

# ─────────────────────────────────────────────────────────────
# 2. PEMUAT CHECKPOINT MODEL HYBRID TOKEN
# ─────────────────────────────────────────────────────────────
def find_default_checkpoint():
    """Mencari checkpoint terbaik yang tersedia secara otomatis."""
    rel_candidates = [
        "checkpoints_c_small_v_small/best_token_hybrid_c_small_v_small.pth",
        "checkpoints_c_small_v_base/best_token_hybrid_c_small_v_base.pth",
        "checkpoints_c_tiny_v_small/best_token_hybrid_c_tiny_v_small.pth",
        "checkpoints_c_tiny_v_tiny/best_token_hybrid_c_tiny_v_tiny.pth",
        "checkpoints/best_token_hybrid_base_model.pth",
        "checkpoints/best_token_hybrid_model.pth",
    ]
    for rel in rel_candidates:
        p1 = Path(rel)
        if p1.exists():
            return p1
        p2 = ROOT_DIR / rel
        if p2.exists():
            return p2
    return Path(rel_candidates[0])

def load_trained_model(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists() and (ROOT_DIR / checkpoint_path).exists():
        checkpoint_path = ROOT_DIR / checkpoint_path
    if not checkpoint_path.exists():
        print(f"\n[Error] File checkpoint tidak ditemukan di: {checkpoint_path}")
        print("Silakan download terlebih dahulu model dari server dengan perintah scp.")
        sys.exit(1)

    print(f"\n[Info] Memuat checkpoint FAS: {checkpoint_path} ...")
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))

    cfg_file = checkpoint_path.parent / "training_config.json"
    vit_name = "vit_small_patch16_224"
    convnext_name = "convnext_small"
    num_heads = 6

    if cfg_file.exists():
        try:
            import json
            with open(cfg_file, "r") as f:
                cfg = json.load(f).get("model_architecture", {})
                vit_name = cfg.get("vit_name", vit_name)
                convnext_name = cfg.get("convnext_name", convnext_name)
                num_heads = cfg.get("num_heads", num_heads)
        except Exception:
            pass
    else:
        path_str = str(checkpoint_path).lower()
        if "c_tiny" in path_str:
            convnext_name = "convnext_tiny"
        elif "c_base" in path_str:
            convnext_name = "convnext_base"
        else:
            convnext_name = "convnext_small"

        if "v_tiny" in path_str:
            vit_name = "vit_tiny_patch16_224"
            num_heads = 3
        elif "v_base" in path_str:
            vit_name = "vit_base_patch16_224"
            num_heads = 12
        else:
            vit_name = "vit_small_patch16_224"
            num_heads = 6

    print(f"[Info] Arsitektur: {convnext_name} + {vit_name} ({num_heads} Heads)")
    model = HybridTokenSAdapterConvNeXt(
        vit_name=vit_name,
        convnext_name=convnext_name,
        adapter_bottleneck=8,
        num_heads=num_heads,
        num_classes=2,
        pretrained=False,
    ).to(DEVICE)

    model.load_state_dict(state_dict)
    model.eval()

    ep = ckpt.get("epoch", "N/A")
    acc = ckpt.get("val_acc", None)
    acc_str = f" | Val Acc: {acc*100:.2f}%" if acc else ""
    print(f"[Success] Model FAS berhasil dimuat (Epoch: {ep}{acc_str})!\n")
    return model, f"{convnext_name} + {vit_name}"

# ─────────────────────────────────────────────────────────────
# 3. LOOP INFERENSI KAMERA REAL-TIME
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Real-Time Face Anti-Spoofing (MediaPipe Tracking)")
    parser.add_argument("--checkpoint", type=str, default=str(find_default_checkpoint()), help="Path ke file model .pth")
    parser.add_argument("--cam", type=int, default=0, help="ID Webcam (default: 0)")
    args = parser.parse_args()

    model, arch_name = load_trained_model(args.checkpoint)
    mp_detector = get_mediapipe_detector()

    # Fallback OpenCV Haar Cascade
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    print(f"[Info] Membuka webcam (Device ID: {args.cam})...")
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[Error] Kamera webcam (ID: {args.cam}) tidak dapat diakses.")
        sys.exit(1)

    # Set resolusi kamera agar jernih
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    import mediapipe as mp
    prev_time = time.time()
    fps = 0.0

    print("="*65)
    print("SISTEM AKTIF! Tekan tombol 'q' pada jendela video untuk keluar.")
    print("="*65)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Warning] Gagal membaca frame kamera.")
            break

        # Hitung FPS
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, (curr_time - prev_time)))
        prev_time = curr_time

        h_frame, w_frame, _ = frame.shape
        faces_detected = []

        # --- A. DETEKSI WAJAH VIA GOOGLE MEDIAPIPE ---
        if mp_detector is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = mp_detector.detect(mp_image)

            for det in detection_result.detections:
                bb = det.bounding_box
                x = int(bb.origin_x)
                y = int(bb.origin_y)
                w = int(bb.width)
                h = int(bb.height)
                faces_detected.append((x, y, w, h))

        # --- B. FALLBACK JIKA MEDIAPIPE TIDAK AKTIF ---
        if mp_detector is None or len(faces_detected) == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            haar_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(80, 80))
            if len(faces_detected) == 0:
                faces_detected = [(x, y, w, h) for (x, y, w, h) in haar_faces]

        # --- C. PROSES SETIAP WAJAH TERDETEKSI ---
        for (x, y, w, h) in faces_detected:
            # Beri margin 12% agar pas dengan distribusi RetinaFace training
            margin_x = int(w * 0.12)
            margin_y = int(h * 0.12)
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(w_frame, x + w + margin_x)
            y2 = min(h_frame, y + h + margin_y)

            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size == 0 or (x2 - x1) < 20 or (y2 - y1) < 20:
                continue

            # Pra-pemrosesan citra wajah
            rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            pil_face = Image.fromarray(rgb_face)
            input_tensor = test_transform(pil_face).unsqueeze(0).to(DEVICE)

            # Inferensi model FAS
            with torch.no_grad():
                out = model(input_tensor)
                probs = out["probs"][0].cpu().numpy()

            spoof_prob = float(probs[0] * 100)
            live_prob  = float(probs[1] * 100)

            # Keputusan: Live jika probabilitas >= 50%
            is_live = live_prob >= 50.0
            color = (0, 220, 0) if is_live else (0, 0, 235)  # BGR: Hijau (Live), Merah (Spoof)
            label = f"LIVE: {live_prob:.1f}%" if is_live else f"SPOOF: {spoof_prob:.1f}%"

            # 1. Gambar Bounding Box Wajah dengan Sudut Stylish
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Sudut tebal (Corner Highlights)
            corner_len = int(min(w, h) * 0.15)
            # Kiri atas
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 4)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 4)
            # Kanan atas
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 4)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 4)
            # Kiri bawah
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 4)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 4)
            # Kanan bawah
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 4)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 4)

            # 2. Gambar Label Tag di Atas Wajah
            tag_w = max(180, int(w * 0.9))
            tag_h = 32
            tag_y1 = max(0, y1 - tag_h - 6)
            tag_y2 = tag_y1 + tag_h
            tag_x2 = min(w_frame, x1 + tag_w)

            cv2.rectangle(frame, (x1, tag_y1), (tag_x2, tag_y2), color, -1)
            cv2.putText(frame, label, (x1 + 8, tag_y2 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

            # 3. Bar Meter Indikator Keaslian di Bawah Bounding Box
            bar_y1 = min(h_frame - 10, y2 + 6)
            bar_y2 = bar_y1 + 8
            cv2.rectangle(frame, (x1, bar_y1), (x2, bar_y2), (50, 50, 50), -1)
            fill_w = int((x2 - x1) * (live_prob / 100.0))
            if fill_w > 0:
                cv2.rectangle(frame, (x1, bar_y1), (x1 + fill_w, bar_y2), color, -1)

        # --- D. HEADER DASHBOARD INFORMASI ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w_frame, 45), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        status_track = "Google MediaPipe BlazeFace (SOTA Tracking)" if mp_detector else "Haar Cascade (Fallback)"
        cv2.putText(frame, f"Model: {arch_name} | Tracker: {status_track}", (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.1f}", (w_frame - 120, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2, cv2.LINE_AA)

        cv2.imshow("Face Anti-Spoofing Live Camera", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' atau ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[Info] Aplikasi kamera ditutup.")

if __name__ == "__main__":
    main()

