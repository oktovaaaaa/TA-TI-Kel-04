import sys
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

from models.s_adapter_convnext_hybrid import HybridTokenSAdapterConvNeXt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT_PATH = Path('./checkpoints/best_token_hybrid_model.pth')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE = 224

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def test_single_frame():
    model = HybridTokenSAdapterConvNeXt(
        vit_name="vit_tiny_patch16_224",
        convnext_name="convnext_tiny",
        adapter_bottleneck=64,
        num_classes=2,
        pretrained=False,
    ).to(DEVICE)

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
    model.load_state_dict(state_dict)
    model.eval()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    print("Mengambil 1 frame dari webcam untuk audit...")
    for _ in range(10): # warmup
        ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Failed to capture frame")
        return

    # Simpan frame mentah
    cv2.imwrite("debug_webcam_frame.jpg", frame)
    print("Frame tersimpan di debug_webcam_frame.jpg")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    tensor = test_transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(tensor)
        logits = out["logits"][0].cpu().numpy()
        probs = out["probs"][0].cpu().numpy()

    print("\n--- HASIL AUDIT MODEL WEBCAM ---")
    print(f"Raw Logits: [{logits[0]:.4f}, {logits[1]:.4f}]")
    print(f"Probabilities: Index 0 = {probs[0]*100:.2f}% | Index 1 = {probs[1]*100:.2f}%")
    print(f"Epoch Checkpoint: {ckpt.get('epoch')}")
    print(f"Val AUC Checkpoint: {ckpt.get('val_auc')}")
    print("--------------------------------\n")

if __name__ == "__main__":
    test_single_frame()

