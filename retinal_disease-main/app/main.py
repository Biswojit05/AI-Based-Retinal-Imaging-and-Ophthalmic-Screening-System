"""
FastAPI backend for the eye-disease chatbot.

Loads the trained ResNet50 checkpoint (model/eye_disease_resnet50.pth)
and exposes a /predict endpoint that accepts an eye fundus photo and
returns the predicted condition with confidence scores.
"""
import io
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "eye_disease_resnet50.pth")
LABELS_PATH = os.path.join(BASE_DIR, "model", "labels.json")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

DISEASE_INFO = {
    "N": {
        "name": "Normal",
        "message": "Aapki eye image mein koi visible abnormality nahi dikh rahi — retina normal lag raha hai.",
    },
    "D": {
        "name": "Diabetic Retinopathy",
        "message": "Signs of diabetic retinopathy dikh rahe hain — yeh diabetes ki wajah se retina ki blood vessels ko hone wala damage hai.",
    },
    "G": {
        "name": "Glaucoma",
        "message": "Glaucoma ke signs dikh rahe hain — optic nerve par pressure se related condition, jo untreated rehne par vision loss kar sakti hai.",
    },
    "C": {
        "name": "Cataract",
        "message": "Cataract ke signs dikh rahe hain — eye lens cloudy hone lagta hai jisse vision blurry ho jaata hai.",
    },
    "A": {
        "name": "Age-related Macular Degeneration (AMD)",
        "message": "AMD ke signs dikh rahe hain — yeh macula (central retina) ko affect karta hai aur central vision ko dheere-dheere kamzor karta hai.",
    },
    "H": {
        "name": "Hypertensive Retinopathy",
        "message": "High blood pressure ki wajah se retina ki blood vessels mein changes dikh rahe hain.",
    },
    "M": {
        "name": "Pathological Myopia",
        "message": "Severe myopia (nearsightedness) ke signs dikh rahe hain jo retina ko bhi affect kar sakta hai.",
    },
    "O": {
        "name": "Other Abnormality",
        "message": "Kuch aur tarah ki abnormality dikh rahi hai jo upar ki common categories mein exactly fit nahi hoti.",
    },
}

DISCLAIMER = (
    "⚠️ Yeh sirf ek AI-based screening tool hai, medical diagnosis nahi. "
    "Please kisi qualified ophthalmologist se confirm karayein."
)

app = FastAPI(title="Eye Disease Screening Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
idx_to_label = None

inference_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def looks_like_fundus_photo(image: Image.Image) -> bool:
    """
    Rough heuristic sanity check. The training data (ODIR-5K) consists of
    fundus-camera photos: a bright, roughly circular retina scan centered
    in a near-black frame. A normal phone photo of an eye (eyelid,
    eyelashes, sclera, skin) does not have this dark-corners / bright-center
    pattern, so this catches the most obvious out-of-distribution uploads.
    It is not a reliable classifier — just an early sanity check.
    """
    gray = np.asarray(image.convert("L").resize((256, 256)), dtype=np.float32)
    patch = 24
    corners = np.concatenate(
        [
            gray[:patch, :patch].ravel(),
            gray[:patch, -patch:].ravel(),
            gray[-patch:, :patch].ravel(),
            gray[-patch:, -patch:].ravel(),
        ]
    )
    center = gray[96:160, 96:160]
    return bool(corners.mean() < 45 and (center.mean() - corners.mean()) > 25)


def load_model():
    global model, idx_to_label

    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        raise RuntimeError(
            f"Model checkpoint not found at {MODEL_PATH}. Run train_model.py first."
        )

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        idx_to_label = {int(k): v for k, v in json.load(f).items()}

    net = models.resnet50(weights=None)
    num_features = net.fc.in_features
    net.fc = nn.Linear(num_features, len(idx_to_label))
    net.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    net.to(device)
    net.eval()
    model = net


@app.on_event("startup")
def on_startup():
    load_model()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "device": str(device)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    raw_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image.")

    fundus_like = looks_like_fundus_photo(image)
    tensor = inference_transform(image).unsqueeze(0).to(device)

    start_time = time.time()
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    inference_time = time.time() - start_time

    ranked = sorted(
        (
            {
                "code": idx_to_label[i],
                "label": DISEASE_INFO.get(idx_to_label[i], {}).get("name", idx_to_label[i]),
                "confidence": round(probs[i].item() * 100, 2),
            }
            for i in range(len(idx_to_label))
        ),
        key=lambda x: x["confidence"],
        reverse=True,
    )

    top = ranked[0]
    top_code = top["code"]
    info = DISEASE_INFO.get(top_code, {"name": top_code, "message": ""})

    if fundus_like:
        reply = (
            f"Mujhe is image mein sabse zyada '{info['name']}' ({top['confidence']}% confidence) "
            f"ke signs dikh rahe hain. {info['message']}\n\n{DISCLAIMER}"
        )
    else:
        reply = (
            "⚠️ Yeh photo ek retina/fundus-camera scan jaisi nahi lag rahi (jis type ki photos par yeh model train hua hai). "
            "Agar yeh normal phone se li gayi eye photo hai (selfie/close-up), toh prediction ka koi matlab nahi hai — "
            "please neeche diye sample jaisi fundus photo try karein.\n\n"
            f"(Model ka raw guess tha: '{info['name']}' {top['confidence']}%, lekin isse ignore karein.)"
        )

    return {
        "prediction": top,
        "top_predictions": ranked[:3],
        "all_predictions": ranked,
        "reply": reply,
        "looks_like_fundus": fundus_like,
        "inference_time": round(inference_time, 3),
    }


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
