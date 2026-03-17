"""
Greenhouse Guardians — Gradio Space (ZeroGPU, free)

Exposes two interfaces:
  1. Gradio UI at /          — visual demo for humans
  2. REST API at /api/classify — for the FastAPI backend to call

API usage:
  POST /api/classify
  Content-Type: multipart/form-data
  Body: file=<image bytes>  [, tomato_conf=0.30] [, flower_conf=0.25]

  Response 200:
  {
    "tomatoes": {
      "detections": [{"class_id", "label", "confidence", "bbox": {x1,y1,x2,y2}}, ...],
      "summary":    {"total": int, "by_class": {"Ripe": int, "Half_Ripe": int, "Unripe": int}}
    },
    "flowers": {
      "flowers":      [{"bounding_box": [x1,y1,x2,y2], "stage": int, "confidence": float}, ...],
      "total_flowers": int,
      "stage_counts":  {"0": int, "1": int, "2": int}
    },
    "annotated_image_b64": "<base64-encoded JPEG>"
  }
"""

import base64
import io
import os

import gradio as gr
import numpy as np
import spaces
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from huggingface_hub import hf_hub_download, login
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from ultralytics.models.sam import SAM3SemanticPredictor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

TOMATO_CLASSES = {0: "Unripe", 1: "Half_Ripe", 2: "Ripe"}
FLOWER_CLASSES = {0: "Bud", 1: "Anthesis", 2: "Post-Anthesis"}

TOMATO_COLORS = {
    "Unripe":    (46,  204, 113),
    "Half_Ripe": (243, 156,  18),
    "Ripe":      (231,  76,  60),
}
FLOWER_COLORS = {
    0: (52,  152, 219),
    1: (155,  89, 182),
    2: (230, 126,  34),
}

SAM3_PROMPTS = [
    "unripe green tomato",       # → class 0: Unripe
    "half ripe orange tomato",   # → class 1: Half_Ripe
    "ripe red tomato",           # → class 2: Ripe
]

# ---------------------------------------------------------------------------
# Model loading (CPU at startup; ZeroGPU moves to GPU during @spaces.GPU calls)
# ---------------------------------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN", "")
if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=False)

SAM3_PATH = os.path.join(MODELS_DIR, "sam3.pt")
if not os.path.exists(SAM3_PATH):
    print("Downloading SAM3 ...")
    hf_hub_download(repo_id="facebook/sam3", filename="sam3.pt", local_dir=MODELS_DIR)

sam3_predictor = SAM3SemanticPredictor(overrides=dict(
    conf=0.30, task="segment", mode="predict",
    model=SAM3_PATH, half=False, verbose=False,
))

FLOWER_PATH = os.path.join(MODELS_DIR, "best.pt")
if not os.path.exists(FLOWER_PATH):
    print("Downloading flower model ...")
    hf_hub_download(
        repo_id="deenp03/tomato_pollination_stage_classifier",
        filename="best.pt", local_dir=MODELS_DIR,
    )
flower_model = YOLO(FLOWER_PATH)

print("Models ready.")


# ---------------------------------------------------------------------------
# Drawing helper
# ---------------------------------------------------------------------------
def _draw_boxes(pil_img: Image.Image, boxes: list[dict]) -> Image.Image:
    img = pil_img.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for b in boxes:
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        color = b["color"]
        label = b["label"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        tb = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle([tb[0]-2, tb[1]-2, tb[2]+2, tb[3]+2], fill=color)
        brightness = 0.299*color[0] + 0.587*color[1] + 0.114*color[2]
        draw.text((x1, y1 - 2), label, fill=(0,0,0) if brightness > 128 else (255,255,255), font=font)
    return img


def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Core inference — @spaces.GPU means ZeroGPU allocates a free GPU only here
# Returns structured dicts so both the UI and the API can consume it.
# ---------------------------------------------------------------------------
@spaces.GPU(duration=120)
def _run_inference(img_np: np.ndarray, tomato_conf: float, flower_conf: float) -> dict:
    """
    Run SAM3 (tomatoes) + YOLOv8 (flowers).
    Returns a dict with full detection data + combined annotated PIL image.
    """
    pil_img = Image.fromarray(img_np)
    draw_boxes = []

    # --- SAM3: Tomato Ripeness ---
    tomato_detections = []
    tomato_by_class = {"Ripe": 0, "Half_Ripe": 0, "Unripe": 0}

    sam3_predictor.set_image(img_np)
    results = sam3_predictor(text=SAM3_PROMPTS)

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < tomato_conf:
                continue
            cls_id = int(box.cls[0])
            label  = TOMATO_CLASSES.get(cls_id, f"class_{cls_id}")
            x1, y1, x2, y2 = (round(float(v), 2) for v in box.xyxy[0])
            tomato_detections.append({
                "class_id": cls_id, "label": label,
                "confidence": round(conf, 4),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            })
            tomato_by_class[label] = tomato_by_class.get(label, 0) + 1
            draw_boxes.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "color": TOMATO_COLORS.get(label, (200, 200, 200)),
                "label": f"{label} {conf:.2f}",
            })

    # --- YOLOv8: Flower Stages ---
    flower_detections = []
    stage_counts = {"0": 0, "1": 0, "2": 0}

    for result in flower_model(img_np, verbose=False, conf=flower_conf):
        if result.boxes is None:
            continue
        for box in result.boxes:
            conf  = round(float(box.conf[0]), 4)
            stage = int(box.cls[0])
            x1, y1, x2, y2 = (round(float(v), 2) for v in box.xyxy[0])
            flower_detections.append({
                "bounding_box": [x1, y1, x2, y2],
                "stage": stage,
                "confidence": conf,
            })
            stage_counts[str(stage)] = stage_counts.get(str(stage), 0) + 1
            stage_name = FLOWER_CLASSES.get(stage, f"stage_{stage}")
            draw_boxes.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "color": FLOWER_COLORS.get(stage, (200, 200, 200)),
                "label": f"{stage_name} {conf:.2f}",
            })

    annotated_pil = _draw_boxes(pil_img, draw_boxes)

    return {
        "tomatoes": {
            "detections": tomato_detections,
            "summary": {"total": len(tomato_detections), "by_class": tomato_by_class},
        },
        "flowers": {
            "flowers": flower_detections,
            "total_flowers": len(flower_detections),
            "stage_counts": stage_counts,
        },
        "annotated_pil": annotated_pil,   # PIL image — stripped before JSON response
    }


# ---------------------------------------------------------------------------
# FastAPI app  (mounted alongside Gradio at /api/*)
# ---------------------------------------------------------------------------
fapp = FastAPI()


@fapp.post("/api/classify")
async def classify(
    file: UploadFile = File(..., description="Image file (jpg/png)"),
    tomato_conf: float = Form(0.30, description="SAM3 confidence threshold for tomatoes"),
    flower_conf: float = Form(0.25, description="YOLO confidence threshold for flowers"),
):
    """
    Classify tomato ripeness (SAM3) and flower stage (YOLOv8) in one call.

    Send as multipart/form-data:
        file        — the image
        tomato_conf — float 0-1, default 0.30
        flower_conf — float 0-1, default 0.25

    Returns JSON matching the structure used internally by classifier.py,
    with an additional annotated_image_b64 field (base64 JPEG).
    """
    image_bytes = await file.read()
    img_np = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))

    result = _run_inference(img_np, tomato_conf, flower_conf)

    annotated_b64 = _pil_to_b64(result.pop("annotated_pil"))

    return JSONResponse({
        **result,
        "annotated_image_b64": annotated_b64,
    })


@fapp.get("/api/health")
async def health():
    return {"status": "ok", "models": ["sam3", "yolov8-flower"]}


# ---------------------------------------------------------------------------
# Gradio UI (mounted at /)
# ---------------------------------------------------------------------------
def _gradio_predict(pil_img: Image.Image, tomato_conf: float, flower_conf: float):
    """Wrapper for the Gradio interface — formats output for display."""
    if pil_img is None:
        return None, "Upload an image first."

    img_np = np.array(pil_img.convert("RGB"))
    result = _run_inference(img_np, tomato_conf, flower_conf)

    tc = result["tomatoes"]["summary"]["by_class"]
    fc = result["flowers"]["stage_counts"]

    summary = f"""
## Results

### Tomatoes — {result['tomatoes']['summary']['total']} detected
| Stage | Count |
|---|---|
| 🟢 Unripe | {tc.get('Unripe', 0)} |
| 🟠 Half-Ripe | {tc.get('Half_Ripe', 0)} |
| 🔴 Ripe | {tc.get('Ripe', 0)} |

### Flowers — {result['flowers']['total_flowers']} detected
| Stage | Count |
|---|---|
| 🔵 Bud | {fc.get('0', 0)} |
| 🟣 Anthesis (ready to pollinate) | {fc.get('1', 0)} |
| 🟤 Post-Anthesis | {fc.get('2', 0)} |
"""
    return result["annotated_pil"], summary


with gr.Blocks(title="Greenhouse Guardians", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
# 🌿 Greenhouse Guardians
**Tomato Ripeness** (SAM3) + **Flower Stage** (YOLOv8) Detection
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Plant Image")
            with gr.Accordion("Confidence thresholds", open=False):
                tomato_conf_slider = gr.Slider(0.10, 0.80, value=0.30, step=0.05,
                    label="Tomato (SAM3)",
                    info="Raise to remove false positives on background/wall.")
                flower_conf_slider = gr.Slider(0.10, 0.80, value=0.25, step=0.05,
                    label="Flower (YOLOv8)")
            run_btn = gr.Button("Detect", variant="primary", size="lg")
        with gr.Column(scale=1):
            image_output = gr.Image(type="pil", label="Annotated Result")
            results_md   = gr.Markdown()

    run_btn.click(
        fn=_gradio_predict,
        inputs=[image_input, tomato_conf_slider, flower_conf_slider],
        outputs=[image_output, results_md],
    )

    gr.Markdown("""
---
**API:** `POST /api/classify` — see Space source for request/response format.
**Models:** SAM3 (tomatoes, zero-shot) · YOLOv8n-C3TR mAP@50=0.902 (flowers)
**Compute:** [ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu) (free)
    """)

# Mount Gradio on the FastAPI app — Gradio UI at /, API routes at /api/*
app = gr.mount_gradio_app(fapp, demo, path="/")
