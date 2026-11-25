import asyncio
import gc
import json
import os
import base64
from pathlib import Path
from collections import Counter
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

try:
    import onnxruntime as ort
except ImportError:
    ort = None

app = FastAPI(title="PYNQ YOLO Object Detection API")

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:5177",
    "http://localhost:5178",
    "http://localhost:4173",
    "http://127.0.0.1:5173",
    "https://pynq-yolo-object-detection.vercel.app",
    FRONTEND_URL,
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin for origin in ALLOWED_ORIGINS if origin],
    allow_origin_regex=r"https://pynq-yolo-object-detection.*\.vercel\.app",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

PALETTE = [
    (34, 211, 238),
    (59, 130, 246),
    (16, 185, 129),
    (245, 158, 11),
    (244, 63, 94),
    (168, 85, 247),
    (236, 72, 153),
    (20, 184, 166),
]

CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45
MAX_DETECTIONS = 25
INPUT_SIZE = 640

BASE_DIR = Path(__file__).resolve().parent.parent


def _resolve_model_path() -> Path:
    env_path = os.getenv("MODEL_PATH")
    candidates = []
    if env_path:
        p = Path(env_path)
        if p.suffix.lower() == ".pt":
            candidates.append(p.with_suffix(".onnx"))
        candidates.append(p)
    candidates.extend([
        BASE_DIR / "models" / "yolo11n.onnx",
        BASE_DIR / "yolo11n.onnx",
    ])
    for candidate in candidates:
        if candidate.exists() and candidate.suffix.lower() == ".onnx":
            return candidate
    raise RuntimeError("YOLO ONNX model not found. Check models/yolo11n.onnx")


MODEL_PATH = _resolve_model_path()
CLASS_NAMES = list(COCO_CLASSES)

if ort is not None:
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    ort_session = ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    ORT_INPUT_NAME = ort_session.get_inputs()[0].name
    meta_names = ort_session.get_modelmeta().custom_metadata_map.get("names")
    if meta_names:
        try:
            parsed = json.loads(meta_names.replace("'", '"'))
            if isinstance(parsed, dict):
                CLASS_NAMES = [parsed[str(i)] for i in range(len(parsed))]
        except Exception:
            pass
    dnn_net = None
    INFERENCE_BACKEND = "onnxruntime"
else:
    ort_session = None
    ORT_INPUT_NAME = None
    dnn_net = cv2.dnn.readNetFromONNX(str(MODEL_PATH))
    INFERENCE_BACKEND = "opencv-dnn"


@app.get("/")
def home():
    return {"message": "PYNQ YOLO API is running", "runtime": INFERENCE_BACKEND}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": MODEL_PATH.name,
        "runtime": INFERENCE_BACKEND,
    }


def _letterbox(image_bgr: np.ndarray, new_size: int = INPUT_SIZE):
    h, w = image_bgr.shape[:2]
    scale = min(new_size / w, new_size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), 114, dtype=np.uint8)
    dw = (new_size - nw) / 2.0
    dh = (new_size - nh) / 2.0
    left = int(round(dw - 0.1))
    top = int(round(dh - 0.1))
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, scale, left, top


def _predict_raw(blob: np.ndarray) -> np.ndarray:
    if ort_session is not None:
        return ort_session.run(None, {ORT_INPUT_NAME: blob})[0]
    dnn_net.setInput(blob)
    return dnn_net.forward()


def _draw_detections(image_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    annotated = image_bgr.copy()
    img_h, img_w = annotated.shape[:2]
    thickness = max(2, round(min(img_h, img_w) / 320))
    font_scale = max(0.45, min(0.75, min(img_h, img_w) / 900))

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cls_id = det["class_id"]
        # PALETTE uses RGB tuples; convert to BGR for OpenCV
        r, g, b = PALETTE[cls_id % len(PALETTE)]
        color = (b, g, r)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        label = f"{det['class']} {det['confidence']:.2f}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        pad = 4
        label_y2 = y1 if y1 - th - pad * 2 >= 0 else min(img_h, y1 + th + pad * 2)
        label_y1 = max(0, label_y2 - th - pad * 2)
        label_x2 = min(img_w, x1 + tw + pad * 2)

        cv2.rectangle(annotated, (x1, label_y1), (label_x2, label_y2), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + pad, label_y2 - pad - baseline // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (10, 15, 28),
            max(1, thickness - 1),
            cv2.LINE_AA,
        )

    return annotated


def _run_detection_sync(contents: bytes):
    try:
        image = Image.open(BytesIO(contents))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Invalid or unsupported image file") from exc

    max_side = 800
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    image_array = np.array(image)
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    img_h, img_w = image_bgr.shape[:2]

    padded, scale, pad_x, pad_y = _letterbox(image_bgr, INPUT_SIZE)
    blob = cv2.dnn.blobFromImage(
        padded, 1.0 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False
    )

    preds = _predict_raw(blob)[0].T  # (8400, 4 + num_classes)
    class_scores = preds[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(len(class_scores)), class_ids]

    keep = confidences >= CONF_THRESHOLD
    preds = preds[keep]
    class_ids = class_ids[keep]
    confidences = confidences[keep]

    detections = []
    counts = Counter()

    if len(preds) > 0:
        nms_boxes = []
        xyxy_boxes = []
        for row in preds:
            cx, cy, bw, bh = row[:4]
            x1 = max(0.0, min(float(img_w), (cx - bw / 2.0 - pad_x) / scale))
            y1 = max(0.0, min(float(img_h), (cy - bh / 2.0 - pad_y) / scale))
            x2 = max(0.0, min(float(img_w), (cx + bw / 2.0 - pad_x) / scale))
            y2 = max(0.0, min(float(img_h), (cy + bh / 2.0 - pad_y) / scale))
            nms_boxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
            xyxy_boxes.append([round(x1), round(y1), round(x2), round(y2)])

        indices = cv2.dnn.NMSBoxes(
            nms_boxes, confidences.tolist(), CONF_THRESHOLD, NMS_THRESHOLD
        )

        if len(indices) > 0:
            selected = sorted(
                np.array(indices).flatten().tolist(),
                key=lambda i: float(confidences[i]),
                reverse=True,
            )[:MAX_DETECTIONS]

            for idx in selected:
                cid = int(class_ids[idx])
                conf = float(confidences[idx])
                cname = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"class_{cid}"
                detections.append({
                    "class": cname,
                    "class_id": cid,
                    "confidence": round(conf, 4),
                    "box": xyxy_boxes[idx],
                })
                counts[cname] += 1

    annotated_bgr = _draw_detections(image_bgr, detections)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # Strip internal class_id from API response payload
    response_detections = [
        {
            "class": d["class"],
            "confidence": d["confidence"],
            "box": d["box"],
        }
        for d in detections
    ]

    output = BytesIO()
    Image.fromarray(annotated_rgb).save(output, format="JPEG", quality=82)
    encoded_image = base64.b64encode(output.getvalue()).decode("utf-8")

    del image, image_array, image_bgr, padded, blob, preds, annotated_bgr, annotated_rgb, output
    gc.collect()

    return {
        "success": True,
        "total_objects": len(response_detections),
        "counts": dict(counts),
        "detections": response_detections,
        "annotated_image": encoded_image,
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image")

    contents = await file.read()

    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be under 10 MB")

    try:
        return await asyncio.to_thread(_run_detection_sync, contents)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
