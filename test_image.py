"""
Skrip Pengujian Wajah Baru (Face Anti-Spoofing Inference)
=========================================================
Arsitektur: Hybrid Token S-Adapter + ConvNeXt
Penggunaan:
  python test_image.py --image path/ke/foto_wajah.jpg
  python test_image.py --image path/ke/foto_wajah.jpg --checkpoint ./checkpoints/best_token_hybrid_model.pth
"""

import os
import sys

# Prevent OpenMP duplicate DLL initialization conflict on Anaconda Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
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
        print("Silakan lakukan training terlebih dahulu:")
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

def predict_face(image_path, model):
    if not os.path.exists(image_path):
        print(f"[Error] File gambar {image_path} tidak ditemukan!")
        return

    print(f"\n[Info] Memeriksa gambar: {image_path} ...")
    pil_img = Image.open(image_path).convert('RGB')
    input_tensor = test_transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(input_tensor)
        if isinstance(out, dict):
            probs = out["probs"][0].cpu().numpy()
        else:
            probs = F.softmax(out, dim=-1)[0].cpu().numpy()

    spoof_prob = float(probs[0] * 100)
    live_prob  = float(probs[1] * 100)

    pred_label = "LIVE (WAJAH ASLI)" if live_prob >= spoof_prob else "SPOOF (WAJAH PALSU)"
    color = "green" if live_prob >= spoof_prob else "red"

    print("\n" + "=" * 55)
    print(f"  HASIL DETEKSI            : {pred_label}")
    print(f"  Confidence Live  (Asli)  : {live_prob:.2f}%")
    print(f"  Confidence Spoof (Palsu) : {spoof_prob:.2f}%")
    print("=" * 55 + "\n")

    plt.figure(figsize=(6, 6))
    plt.imshow(pil_img)
    plt.axis('off')
    plt.title(f"Prediksi: {pred_label}\nLive: {live_prob:.1f}% | Spoof: {spoof_prob:.1f}%", 
              fontsize=14, fontweight='bold', color=color)
    plt.tight_layout()
    plt.savefig('./hasil_prediksi.png', dpi=150)
    print("[Success] Gambar hasil prediksi telah disimpan ke: ./hasil_prediksi.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uji Anti-Spoofing Wajah (Hybrid Token S-Adapter + ConvNeXt)")
    parser.add_argument("--image", type=str, default="", help="Path file foto (.jpg / .png)")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Path checkpoint model (.pth)")
    args = parser.parse_args()

    img_path = args.image
    if not img_path:
        sample_imgs = list(Path('./CelebA_Spoof').rglob('*.jpg'))
        if sample_imgs:
            img_path = str(sample_imgs[0])
            print(f"[Info] Menugaskan sampel gambar otomatis: {img_path}")
        else:
            print("Silakan tentukan path gambar dengan argumen --image <path_foto>")
            sys.exit(0)

    model = load_trained_model(args.checkpoint)
    predict_face(img_path, model)
