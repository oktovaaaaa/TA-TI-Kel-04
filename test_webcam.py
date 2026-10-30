"""
Skrip Deteksi Wajah Real-Time Webcam (Face Anti-Spoofing Inference)
===================================================================
Arsitektur: Hybrid Token S-Adapter + ConvNeXt
Menggunakan checkpoint model (best_token_hybrid_model.pth) dan kamera webcam secara real-time.

Penggunaan:
  python test_webcam.py
  python test_webcam.py --checkpoint ./checkpoints/best_token_hybrid_model.pth

Tekan tombol 'q' pada jendela video untuk keluar.
"""

import os
import sys
import time
import argparse

# Prevent OpenMP duplicate DLL initialization conflict on Anaconda Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms

from models.s_adapter_convnext_hybrid import HybridTokenSAdapterConvNeXt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEFAULT_CHECKPOINT = Path('./checkpoints/best_token_hybrid_model.pth')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE = 224

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def load_trained_model(checkpoint_path=DEFAULT_CHECKPOINT):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"[Error] File checkpoint {checkpoint_path} tidak ditemukan.")
        print("Pastikan Anda telah menjalankan training terlebih dahulu:")
        print("  python train_hybrid_token.py")
        sys.exit(1)

    print(f"[Info] Memuat model arsitektur Hybrid Token S-Adapter dari: {checkpoint_path} ...")
    model = HybridTokenSAdapterConvNeXt(
        vit_name="vit_tiny_patch16_224",
        convnext_name="convnext_tiny",
        adapter_bottleneck=64,
        num_classes=2,
        pretrained=False,
    ).to(DEVICE)

    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
    model.load_state_dict(state_dict)
    model.eval()
    epoch_info = ckpt.get("epoch", "N/A")
    print(f"[Success] Model berhasil dimuat ke memori (Epoch: {epoch_info})!")
    return model

def main():
    parser = argparse.ArgumentParser(description="Real-Time Webcam Face Anti-Spoofing")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Path checkpoint model (.pth)")
    parser.add_argument("--cam", type=int, default=0, help="ID Device Webcam (default: 0)")
    args = parser.parse_args()

    model = load_trained_model(args.checkpoint)

    # Load OpenCV Face Detector (Haarcascade)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    print(f"\n[Info] Membuka webcam (Device ID: {args.cam})... Tekan tombol 'q' pada jendela video untuk keluar.")
    cap = cv2.VideoCapture(args.cam)

    if not cap.isOpened():
        print(f"[Error] Kamera webcam (ID: {args.cam}) tidak dapat diakses atau diizinkan.")
        sys.exit(1)

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Error] Gagal membaca frame dari webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60))

        # Jika tidak ada wajah terdeteksi, uji area tengah layar
        if len(faces) == 0:
            h_f, w_f, _ = frame.shape
            cx, cy = w_f // 2, h_f // 2
            sz = min(w_f, h_f) // 2
            faces = [(cx - sz//2, cy - sz//2, sz, sz)]

        for (x, y, w, h) in faces:
            margin = int(w * 0.1)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)

            face_roi = frame[y1:y2, x1:x2]

            if face_roi.size > 0:
                rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                pil_face = Image.fromarray(rgb_face)

                input_tensor = test_transform(pil_face).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    out = model(input_tensor)
                    if isinstance(out, dict):
                        probs = out["probs"][0].cpu().numpy()
                    else:
                        probs = F.softmax(out, dim=-1)[0].cpu().numpy()

                spoof_prob = float(probs[0] * 100)
                live_prob  = float(probs[1] * 100)

                if live_prob >= spoof_prob:
                    label = f"LIVE: {live_prob:.1f}%"
                    color = (0, 255, 0)
                else:
                    label = f"SPOOF: {spoof_prob:.1f}%"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(frame, (x1, y1 - 35), (x2, y1), color, -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "Tekan 'q' untuk Keluar", (10, frame.shape[0] - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Face Anti-Spoofing Realtime Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[Info] Pengujian realtime webcam selesai!")

if __name__ == "__main__":
    main()
