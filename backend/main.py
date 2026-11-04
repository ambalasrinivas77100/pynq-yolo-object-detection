import os
import base64
from pathlib import Path
from collections import Counter
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI(title="PYNQ YOLO Object Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your Vercel URL after deployment
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv(
    "MODEL_PATH",
    str(BASE_DIR / "models" / "yolo11n.pt")
))

if not MODEL_PATH.exists():
    MODEL_PATH = BASE_DIR / "yolo11n.pt"

if not MODEL_PATH.exists():
    raise RuntimeError("YOLO model not found. Check models/yolo11n.pt")

model = YOLO(str(MODEL_PATH))


@app.get("/")
def home():
    return {"message": "PYNQ YOLO API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model": MODEL_PATH.name}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image")

    contents = await file.read()

    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be under 10 MB")

    try:
        image = Image.open(BytesIO(contents)).convert("RGB")
        image_array = np.array(image)
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        results = model.predict(image_bgr, conf=0.25, verbose=False)
        result = results[0]

        detections = []
        counts = Counter()

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                class_name = model.names[class_id]

                x1, y1, x2, y2 = [
                    round(float(v)) for v in box.xyxy[0].tolist()
                ]

                detections.append({
                    "class": class_name,
                    "confidence": round(confidence, 4),
                    "box": [x1, y1, x2, y2]
                })

                counts[class_name] += 1

        annotated = result.plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        output = BytesIO()
        Image.fromarray(annotated_rgb).save(output, format="JPEG", quality=85)

        encoded_image = base64.b64encode(output.getvalue()).decode("utf-8")

        return {
            "success": True,
            "total_objects": len(detections),
            "counts": dict(counts),
            "detections": detections,
            "annotated_image": encoded_image
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
