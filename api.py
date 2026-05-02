
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np, tensorflow as tf
from PIL import Image
import io

app   = FastAPI(title="API Classification de Riz", version="1.0")
model = tf.keras.models.load_model("modele_riz_final.h5")

CLASSES  = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]
IMG_SIZE = (224, 224)

app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def root():
    return {"message": "API Classification de Riz", "status": "ok"}

@app.get("/health")
def health():
    return {"status": "ok", "modele": "MobileNetV2", "classes": CLASSES}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents  = await file.read()
    img       = Image.open(io.BytesIO(contents)).convert("RGB")
    img       = img.resize(IMG_SIZE)
    arr       = np.array(img) / 255.0
    arr       = np.expand_dims(arr, axis=0).astype("float32")
    preds     = model.predict(arr, verbose=0)[0]
    pred_idx  = int(np.argmax(preds))
    return {
        "classe"       : CLASSES[pred_idx],
        "confiance"    : round(float(preds[pred_idx]) * 100, 2),
        "probabilites" : {c: round(float(p)*100, 2)
                          for c, p in zip(CLASSES, preds)}
    }
