"""
FastAPI Backend Server untuk Face Anti-Spoofing (Model H5 Champion)
==================================================================
Melayani permintaan inferensi dari Aplikasi Mobile Flutter via Tailscale / LAN.

Fitur:
- Stage 1: Face Detection (Google MediaPipe BlazeFace / OpenCV Haar Cascade)
  Menolak citra jika tidak ada wajah terdeteksi.
- Stage 2: Face Anti-Spoofing Model H5 (ConvNeXt-Small + ViT-Small)
  Mengklasifikasikan crop wajah menjadi LIVE vs SPOOF.
- Endpoint GET / : Status model dan sistem
- Endpoint GET /ping : Uji koneksi cepat dari HP
- Endpoint POST /predict : Menerima multipart image dari kamera HP / Galeri
- Endpoint POST /predict_base64 : Menerima JSON base64 image
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import io
import sys
import time
import base64
import argparse
from pathlib import Path
from PIL import Image, ImageOps

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Tambahkan root proyek ke sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.hybrid_convnext_vit_s_adapter import HybridTokenSAdapterConvNeXt

app = FastAPI(
    title="Face Anti-Spoofing AI API",
    description="Dual-Branch Hybrid Token S-Adapter (ConvNeXt-Small + ViT-Small) with Face Detection",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

MODEL = None
MODEL_INFO = {}
FACE_DETECTOR = None

HAAR_CASCADES = []

def init_face_detector():
    """Inisialisasi OpenCV Face Detector Cascades (Default, Alt2, Alt)."""
    global HAAR_CASCADES
    cascade_names = [
        'haarcascade_frontalface_default.xml',
        'haarcascade_frontalface_alt2.xml',
        'haarcascade_frontalface_alt.xml',
    ]
    HAAR_CASCADES = []
    for name in cascade_names:
        try:
            path = cv2.data.haarcascades + name
            cas = cv2.CascadeClassifier(path)
            if not cas.empty():
                HAAR_CASCADES.append(cas)
        except Exception:
            pass
    print(f"[Success] {len(HAAR_CASCADES)} OpenCV Face Detector Cascades siap!")

def detect_and_crop_face(pil_img: Image.Image):
    """
    Mendeteksi wajah manusia dengan dukungan multi-rotasi (0, 90, 270, 180 derajat)
    dan histogram equalization agar tahan terhadap orientasi kamera HP, pencahayaan keras, dan kacamata.
    Jika tidak ada wajah yang terdeteksi secara ketat, kembalikan None, None, False.
    """
    img = ImageOps.exif_transpose(pil_img)
    if img.mode != "RGB":
        img = img.convert("RGB")
        
    rotations = [0, 90, 270, 180]
    
    for rot in rotations:
        if rot == 0:
            test_img = img
        elif rot == 90:
            test_img = img.rotate(270, expand=True)
        elif rot == 270:
            test_img = img.rotate(90, expand=True)
        elif rot == 180:
            test_img = img.rotate(180, expand=True)
            
        img_w, img_h = test_img.size
        cv_img = cv2.cvtColor(np.array(test_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)
        
        for cas in HAAR_CASCADES:
            for g in (gray, gray_eq):
                faces = cas.detectMultiScale(g, scaleFactor=1.1, minNeighbors=3, minSize=(35, 35))
                if len(faces) > 0:
                    largest = max(faces, key=lambda f: f[2] * f[3])
                    x, y, w, h = largest
                    
                    margin_x = int(w * 0.15)
                    margin_y = int(h * 0.15)
                    x1 = max(0, x - margin_x)
                    y1 = max(0, y - margin_y)
                    x2 = min(img_w, x + w + margin_x)
                    y2 = min(img_h, y + h + margin_y)
                    
                    face_crop = test_img.crop((x1, y1, x2, y2))
                    return face_crop, [int(x1), int(y1), int(x2), int(y2)], True
                    
    # Jika tidak ada wajah frontal sama sekali (misal rak buku, dinding polos, dll)
    return None, None, False

def load_h5_model(checkpoint_path: str):
    global MODEL, MODEL_INFO
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan di {ckpt_path}")
        
    print(f"\n[Info] Memuat model Face Anti-Spoofing dari: {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
    
    model = HybridTokenSAdapterConvNeXt(
        vit_name="vit_small_patch16_224",
        convnext_name="convnext_small",
        adapter_bottleneck=8,
        num_heads=6,
        dropout=0.1,
        num_classes=2,
        pretrained=False,
        freeze_backbone=True
    ).to(DEVICE)
    
    model.load_state_dict(state_dict)
    model.eval()
    MODEL = model
    
    epoch = ckpt.get("epoch", "9")
    val_acc = ckpt.get("val_acc", 0.9961)
    MODEL_INFO = {
        "model_name": "H5: ConvNeXt-Small + ViT-Small [CHAMPION]",
        "checkpoint": str(ckpt_path.name),
        "device": str(DEVICE).upper(),
        "epoch": epoch,
        "val_accuracy": f"{val_acc*100:.2f}%" if val_acc else "99.61%",
        "test_accuracy": "99.67%",
        "test_hter": "0.43%",
        "status": "Ready for Mobile Inference"
    }
    print(f"[Success] Model berhasil dimuat ke memori ({MODEL_INFO['device']})!\n")

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Face Anti-Spoofing AI Server",
        "model_info": MODEL_INFO,
        "face_detector": f"{len(HAAR_CASCADES)} haar cascades"
    }

@app.get("/ping")
def ping():
    return {
        "status": "pong",
        "timestamp": time.time(),
        "message": "Koneksi ke Server AI Berhasil!"
    }

def run_inference(image: Image.Image):
    t_start = time.perf_counter()
    
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    # 1. Tahap 1: Deteksi & Pemotongan Wajah (Multi-Rotasi + Alt2 + Contrast Equalization)
    face_crop, box, face_found = detect_and_crop_face(image)
    
    # Jika TIDAK ADA wajah manusia sama sekali dalam gambar
    if not face_found or face_crop is None:
        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return {
            "prediction": "NO_FACE",
            "face_detected": False,
            "message": "Tidak ada wajah terdeteksi dalam bingkai!",
            "is_live": False,
            "confidence_live": 0.0,
            "confidence_spoof": 0.0,
            "latency_ms": round(t_elapsed_ms, 1),
            "device": str(DEVICE).upper()
        }
    
    # 2. Tahap 2: Face Anti-Spoofing pada Area Wajah
    input_tensor = transform(face_crop).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        out = MODEL(input_tensor)
        logits = out["logits"]
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
        
    t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    
    spoof_prob = float(probs[0] * 100.0)
    live_prob = float(probs[1] * 100.0)
    is_live = bool(live_prob >= spoof_prob)
    prediction = "LIVE" if is_live else "SPOOF"
    
    return {
        "prediction": prediction,
        "face_detected": True,
        "box": box,
        "is_live": is_live,
        "confidence_live": round(live_prob, 2),
        "confidence_spoof": round(spoof_prob, 2),
        "latency_ms": round(t_elapsed_ms, 1),
        "device": str(DEVICE).upper()
    }

@app.post("/predict")
async def predict_image(image: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model belum siap.")
        
    try:
        content = await image.read()
        pil_img = Image.open(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File gambar tidak valid: {str(e)}")
        
    result = run_inference(pil_img)
    return {
        "status": "success",
        "filename": image.filename,
        **result
    }

class Base64Payload(BaseModel):
    image_base64: str

@app.post("/predict_base64")
async def predict_base64(payload: Base64Payload):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model belum siap.")
        
    try:
        raw_b64 = payload.image_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",")[1]
        img_bytes = base64.b64decode(raw_b64)
        pil_img = Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Decode Base64 gagal: {str(e)}")
        
    result = run_inference(pil_img)
    return {
        "status": "success",
        **result
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000, help="Port server (default: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_c_small_v_small/best_token_hybrid_c_small_v_small.pth")
    args = parser.parse_args()
    
    init_face_detector()
    load_h5_model(args.checkpoint)
    print(f"[Info] Menjalankan API Server di http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
