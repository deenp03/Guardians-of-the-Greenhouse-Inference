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

Concurrency modes (set MODEL_POOL_SIZE env var):
  MODEL_POOL_SIZE=1  (default) — ZeroGPU mode: @spaces.GPU, lazy singleton, free H200
  MODEL_POOL_SIZE=4             — Dedicated GPU mode: 4 pre-loaded instances, no ZeroGPU needed
"""

import base64
import io
import os
import queue
import threading

import cv2
import gradio as gr
import numpy as np
import torch
from huggingface_hub import hf_hub_download, login
from PIL import Image, ImageDraw, ImageFont
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from ultralytics import YOLO
from ultralytics.models.sam import SAM3SemanticPredictor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODELS_DIR      = "models"
MODEL_POOL_SIZE = int(os.getenv("MODEL_POOL_SIZE", "1"))
ZEROGPU_MODE    = MODEL_POOL_SIZE == 1

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
    "ripe red tomato",
    "unripe green tomato",
    "tomato fruit",
]   # multiple descriptive prompts → better SAM3 grounding; ripeness via HSV later

TOMATO_LABEL_TO_ID = {"Unripe": 0, "Half_Ripe": 1, "Ripe": 2}

# ---------------------------------------------------------------------------
# Download weights at startup
# ---------------------------------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN", "")
if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=False)

import filelock

TOMATO_PATH = os.path.join(MODELS_DIR, "segment_ripeness.pt")
FLOWER_PATH = os.path.join(MODELS_DIR, "best.pt")

with filelock.FileLock(os.path.join(MODELS_DIR, "download.lock")):
    if not os.path.exists(TOMATO_PATH):
        print("Downloading tomato segmentation model ...")
        hf_hub_download(repo_id="deenp03/tomato-ripeness-classifier", filename="segment_ripeness.pt", local_dir=MODELS_DIR)
    if not os.path.exists(FLOWER_PATH):
        print("Downloading flower model ...")
        hf_hub_download(
            repo_id="deenp03/tomato_pollination_stage_classifier",
            filename="best.pt", local_dir=MODELS_DIR,
        )

print(f"Weights ready. Mode: {'ZeroGPU (pool=1)' if ZEROGPU_MODE else f'Dedicated GPU (pool={MODEL_POOL_SIZE})'}")

# ---------------------------------------------------------------------------
# Model management
#
# ZeroGPU mode  (MODEL_POOL_SIZE=1):
#   Lazy singleton loaded inside @spaces.GPU — GPU allocated per call by HF.
#
# Dedicated GPU mode (MODEL_POOL_SIZE>1):
#   Pool of N model pairs pre-loaded at startup. A threading.Semaphore gates
#   access so at most N requests run concurrently. No @spaces.GPU needed.
# ---------------------------------------------------------------------------
def _make_model_pair():
    sam3 = SAM3SemanticPredictor(overrides=dict(
        conf=0.30, task="segment", mode="predict",
        model=TOMATO_PATH, half=True, verbose=False,
    ))
    flower = YOLO(FLOWER_PATH)
    return sam3, flower


if ZEROGPU_MODE:
    import spaces

    # Lazy singletons — created inside the @spaces.GPU context
    _sam3_singleton   = None
    _flower_singleton = None

    def _get_zerogpu_models():
        global _sam3_singleton, _flower_singleton
        if _sam3_singleton is None:
            print("Loading SAM3 into GPU memory ...")
            _sam3_singleton, _flower_singleton = _make_model_pair()
        return _sam3_singleton, _flower_singleton

else:
    # Pre-load pool at startup (dedicated GPU has memory available immediately)
    print(f"Loading {MODEL_POOL_SIZE} model instance(s) into GPU memory ...")
    _pool: queue.Queue = queue.Queue()
    for _ in range(MODEL_POOL_SIZE):
        _pool.put(_make_model_pair())
    print("Model pool ready.")


# ---------------------------------------------------------------------------
# HSV ripeness classifier — called on each SAM3-detected tomato crop
# ---------------------------------------------------------------------------
def classify_ripeness_by_color(region_rgb: np.ndarray) -> str:
    """
    Classify a tomato crop as Unripe / Half_Ripe / Ripe by HSV hue distribution.

    HSV hue (OpenCV 0-179):
      Red/Ripe   : H in [0,10] or [170,180]
      Orange/Half: H in [10,25]
      Green/Unripe: H in [35,85]
    Only saturated+bright pixels (tomato skin) are counted.
    """
    hsv = cv2.cvtColor(region_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    mask     = (s > 60) & (v > 60)
    h_masked = h[mask]

    if len(h_masked) < 50:
        return "Unripe"

    total      = len(h_masked)
    red_pct    = np.sum((h_masked <= 10) | (h_masked >= 170)) / total
    orange_pct = np.sum((h_masked > 10)  & (h_masked <= 25))  / total

    if red_pct >= 0.35:
        return "Ripe"
    if orange_pct >= 0.25 or (red_pct >= 0.20 and orange_pct >= 0.15):
        return "Half_Ripe"
    return "Unripe"


# ---------------------------------------------------------------------------
# IoU dedupe — collapses duplicate boxes that come from multiple SAM3 prompts
# matching the same physical fruit
# ---------------------------------------------------------------------------
def _iou(a, b) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw  = max(0.0, ix2 - ix1)
    ih  = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _dedupe_boxes(items: list[dict], iou_thresh: float = 0.5) -> list[dict]:
    """Greedy NMS over confidence. Each item must have 'bbox' (xyxy tuple) and 'confidence'."""
    items = sorted(items, key=lambda d: -d["confidence"])
    kept: list[dict] = []
    for it in items:
        if all(_iou(it["bbox"], k["bbox"]) < iou_thresh for k in kept):
            kept.append(it)
    return kept


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def _draw_boxes(pil_img: Image.Image, boxes: list[dict]) -> Image.Image:
    img = pil_img.copy().convert("RGB")

    # Scale thickness and font to image size so boxes are visible at any resolution
    w, h   = img.size
    scale  = max(w, h) / 800
    lw     = max(3, int(4 * scale))
    fsize  = max(16, int(18 * scale))

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fsize)
    except Exception:
        font = ImageFont.load_default()

    # Semi-transparent fill layer
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ov      = ImageDraw.Draw(overlay)
    draw    = ImageDraw.Draw(img)

    for b in boxes:
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        color, label    = b["color"], b["label"]

        ov.rectangle([x1, y1, x2, y2], fill=(*color, 45))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)

        tb     = draw.textbbox((0, 0), label, font=font)
        lw2    = tb[2] - tb[0]
        lh     = tb[3] - tb[1]
        pad    = 4
        lx     = max(0, int(x1))
        ly     = max(0, int(y1) - lh - pad * 2 - lw)

        draw.rounded_rectangle([lx, ly, lx + lw2 + pad*2, ly + lh + pad*2],
                                radius=4, fill=color)
        brightness = 0.299*color[0] + 0.587*color[1] + 0.114*color[2]
        draw.text((lx + pad, ly + pad), label,
                  fill=(0, 0, 0) if brightness > 128 else (255, 255, 255),
                  font=font)

    img = Image.alpha_composite(img.convert("RGBA"), overlay)
    return img.convert("RGB")


def _draw_masks(pil_img: Image.Image, masks: list[dict]) -> Image.Image:
    """Draw segmentation outlines with leader lines to a side label column."""
    img = pil_img.copy().convert("RGB")

    w, h  = img.size
    scale = max(w, h) / 800
    lw    = max(2, int(3 * scale))
    fsize = max(14, int(16 * scale))
    pad   = 6

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fsize)
    except Exception:
        font = ImageFont.load_default()

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ov      = ImageDraw.Draw(overlay)
    draw    = ImageDraw.Draw(img)

    # First pass — draw polygons and collect centroids
    items = []
    for m in masks:
        pts   = [(float(x), float(y)) for x, y in m["polygon"]]
        color = m["color"]
        label = m["label"]
        if len(pts) < 3:
            continue

        ov.polygon(pts, fill=(*color, 80))
        draw.polygon(pts, outline=(0, 0, 0), width=lw + 4)
        draw.polygon(pts, outline=color, width=lw)

        cx = int(sum(x for x, _ in pts) / len(pts))
        cy = int(sum(y for _, y in pts) / len(pts))
        items.append((cx, cy, color, label))

    if not items:
        img = Image.alpha_composite(img.convert("RGBA"), overlay)
        return img.convert("RGB")

    # Sort top-to-bottom so label column reads naturally
    items.sort(key=lambda t: t[1])

    # Measure labels to size the column
    max_lw = max(draw.textbbox((0, 0), lbl, font=font)[2] for *_, lbl in items)
    lh     = draw.textbbox((0, 0), items[0][3], font=font)[3]
    row_h  = lh + pad * 2 + 4

    # Fit column height to image; shrink row spacing if needed
    total_h = len(items) * row_h
    if total_h > h - 20:
        row_h = max(lh + pad, (h - 20) // len(items))

    # Place column on whichever side has fewer centroid points
    left_count  = sum(1 for cx, *_ in items if cx < w // 2)
    right_count = len(items) - left_count
    margin = 8
    if right_count <= left_count:          # more room on the right
        col_x = w - max_lw - pad * 2 - margin
        anchor_side = "left"               # line attaches to left edge of label
    else:
        col_x = margin
        anchor_side = "right"

    start_y = max(margin, (h - len(items) * row_h) // 2)

    # Second pass — draw leader lines and labels
    dot_r = max(4, int(6 * scale))
    for i, (cx, cy, color, label) in enumerate(items):
        ly   = start_y + i * row_h
        tb   = draw.textbbox((0, 0), label, font=font)
        lbl_w = tb[2] - tb[0]

        # Label pill
        rx1 = col_x
        rx2 = col_x + lbl_w + pad * 2
        ry1 = ly
        ry2 = ly + lh + pad * 2
        draw.rounded_rectangle([rx1, ry1, rx2, ry2], radius=4, fill=color)
        brightness = 0.299*color[0] + 0.587*color[1] + 0.114*color[2]
        draw.text((rx1 + pad, ry1 + pad), label,
                  fill=(0, 0, 0) if brightness > 128 else (255, 255, 255),
                  font=font)

        # Leader line anchor point (middle of the near edge of the label)
        mid_y = (ry1 + ry2) // 2
        if anchor_side == "left":
            ax = rx1
        else:
            ax = rx2

        # Shadow then colored line
        draw.line([(ax, mid_y), (cx, cy)], fill=(0, 0, 0), width=3)
        draw.line([(ax, mid_y), (cx, cy)], fill=color,    width=2)

        # Dot at centroid
        draw.ellipse([cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r],
                     fill=color, outline=(0, 0, 0), width=2)

    img = Image.alpha_composite(img.convert("RGBA"), overlay)
    return img.convert("RGB")


def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Core inference logic (hardware-agnostic)
# mode: "tomatoes" | "flowers" | "both"
# ---------------------------------------------------------------------------
def _infer(sam3, flower, img_np: np.ndarray,
           tomato_conf: float, flower_conf: float,
           mode: str = "both") -> dict:
    pil_img    = Image.fromarray(img_np)
    draw_boxes = []
    result     = {}

    # --- SAM3: Tomato detection + HSV ripeness classification ---
    if mode in ("tomatoes", "both"):
        tomato_detections = []
        tomato_by_class   = {"Ripe": 0, "Half_Ripe": 0, "Unripe": 0}

        # Collect raw boxes from every SAM3 text prompt; multiple prompts often
        # detect the same fruit, so we dedupe by IoU before HSV-classifying.
        raw: list[dict] = []
        sam3.set_image(img_np)
        for r in sam3(text=SAM3_PROMPTS):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < tomato_conf:
                    continue
                x1, y1, x2, y2 = (round(float(v), 2) for v in box.xyxy[0])
                raw.append({"bbox": (x1, y1, x2, y2), "confidence": conf})

        for d in _dedupe_boxes(raw, iou_thresh=0.5):
            x1, y1, x2, y2 = d["bbox"]
            conf = d["confidence"]

            # Crop detected region and classify by HSV color
            ix1, iy1 = max(0, int(x1)), max(0, int(y1))
            ix2, iy2 = min(img_np.shape[1], int(x2)), min(img_np.shape[0], int(y2))
            crop  = img_np[iy1:iy2, ix1:ix2]
            label = classify_ripeness_by_color(crop) if crop.size > 0 else "Unripe"

            cls_id = TOMATO_LABEL_TO_ID[label]
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

        result["tomatoes"] = {
            "detections": tomato_detections,
            "summary": {"total": len(tomato_detections), "by_class": tomato_by_class},
        }

    # --- YOLOv8: Flower Stages ---
    if mode in ("flowers", "both"):
        flower_detections = []
        stage_counts      = {"0": 0, "1": 0, "2": 0}

        for r in flower(img_np, verbose=False, conf=flower_conf):
            if r.boxes is None:
                continue
            for box in r.boxes:
                conf  = round(float(box.conf[0]), 4)
                stage = int(box.cls[0])
                x1, y1, x2, y2 = (round(float(v), 2) for v in box.xyxy[0])
                flower_detections.append({"bounding_box": [x1, y1, x2, y2], "stage": stage, "confidence": conf})
                stage_counts[str(stage)] = stage_counts.get(str(stage), 0) + 1
                draw_boxes.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "color": FLOWER_COLORS.get(stage, (200, 200, 200)),
                    "label": f"{FLOWER_CLASSES.get(stage, f'stage_{stage}')} {conf:.2f}",
                })

        result["flowers"] = {
            "flowers": flower_detections,
            "total_flowers": len(flower_detections),
            "stage_counts": stage_counts,
        }

    result["annotated_pil"] = _draw_boxes(pil_img, draw_boxes)
    return result


def _infer_segment(sam3, img_np: np.ndarray, tomato_conf: float) -> dict:
    """SAM3 segmentation — returns precise polygon contours instead of bounding boxes."""
    pil_img    = Image.fromarray(img_np)
    masks_draw = []
    detections = []

    # Collect raw box+polygon pairs from every SAM3 prompt, then dedupe.
    raw: list[dict] = []
    sam3.set_image(img_np)
    for r in sam3(text=SAM3_PROMPTS):
        if r.boxes is None or len(r.boxes) == 0:
            continue

        # masks.xy is a list of (N,2) numpy arrays — one polygon per detection
        polygons = r.masks.xy if r.masks is not None else [None] * len(r.boxes)

        for i, box in enumerate(r.boxes):
            conf = float(box.conf[0])
            if conf < tomato_conf:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            polygon = polygons[i].tolist() if polygons[i] is not None else []
            raw.append({"bbox": (x1, y1, x2, y2), "confidence": conf, "polygon": polygon})

    for d in _dedupe_boxes(raw, iou_thresh=0.5):
        x1, y1, x2, y2 = d["bbox"]
        conf    = d["confidence"]
        polygon = d["polygon"]

        # Crop detected region and classify by HSV color
        ix1, iy1 = max(0, int(x1)), max(0, int(y1))
        ix2, iy2 = min(img_np.shape[1], int(x2)), min(img_np.shape[0], int(y2))
        crop  = img_np[iy1:iy2, ix1:ix2]
        label = classify_ripeness_by_color(crop) if crop.size > 0 else "Unripe"

        cls_id = TOMATO_LABEL_TO_ID[label]
        color  = TOMATO_COLORS.get(label, (200, 200, 200))

        detections.append({
            "class_id":   cls_id,
            "label":      label,
            "confidence": round(conf, 4),
            "polygon":    polygon,
        })
        masks_draw.append({"polygon": polygon, "color": color,
                            "label": f"{label} {conf:.2f}"})

    return {
        "detections":    detections,
        "total":         len(detections),
        "annotated_pil": _draw_masks(pil_img, masks_draw),
    }


# ---------------------------------------------------------------------------
# Public inference entry point — switches on mode
# ---------------------------------------------------------------------------
if ZEROGPU_MODE:
    @spaces.GPU(duration=30)
    def _run_inference(img_np: np.ndarray, tomato_conf: float, flower_conf: float,
                       mode: str = "both") -> dict:
        sam3, flower = _get_zerogpu_models()
        return _infer(sam3, flower, img_np, tomato_conf, flower_conf, mode)

    @spaces.GPU(duration=30)
    def _run_segment(img_np: np.ndarray, tomato_conf: float) -> dict:
        sam3, _ = _get_zerogpu_models()
        return _infer_segment(sam3, img_np, tomato_conf)

else:
    def _run_inference(img_np: np.ndarray, tomato_conf: float, flower_conf: float,
                       mode: str = "both") -> dict:
        sam3, flower = _pool.get()
        try:
            return _infer(sam3, flower, img_np, tomato_conf, flower_conf, mode)
        finally:
            _pool.put((sam3, flower))

    def _run_segment(img_np: np.ndarray, tomato_conf: float) -> dict:
        sam3, flower = _pool.get()
        try:
            return _infer_segment(sam3, img_np, tomato_conf)
        finally:
            _pool.put((sam3, flower))


# ---------------------------------------------------------------------------
# REST API helpers
# ---------------------------------------------------------------------------
async def _parse_request(request: Request):
    form        = await request.form()
    image_bytes = await form["file"].read()
    tomato_conf = float(form.get("tomato_conf", 0.30))
    flower_conf = float(form.get("flower_conf", 0.25))
    img_np      = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    return img_np, tomato_conf, flower_conf


# POST /api/segment — SAM3 precise polygon masks (tomatoes only)
async def _api_segment(request: Request):
    img_np, tomato_conf, _ = await _parse_request(request)
    result        = _run_segment(img_np, tomato_conf)
    annotated_b64 = _pil_to_b64(result.pop("annotated_pil"))
    return JSONResponse({**result, "annotated_image_b64": annotated_b64})


# POST /api/tomatoes — SAM3 ripeness only
async def _api_tomatoes(request: Request):
    img_np, tomato_conf, flower_conf = await _parse_request(request)
    result        = _run_inference(img_np, tomato_conf, flower_conf, mode="tomatoes")
    annotated_b64 = _pil_to_b64(result.pop("annotated_pil"))
    return JSONResponse({**result, "annotated_image_b64": annotated_b64})


# POST /api/flowers — YOLOv8 flower stages only
async def _api_flowers(request: Request):
    img_np, tomato_conf, flower_conf = await _parse_request(request)
    result        = _run_inference(img_np, tomato_conf, flower_conf, mode="flowers")
    annotated_b64 = _pil_to_b64(result.pop("annotated_pil"))
    return JSONResponse({**result, "annotated_image_b64": annotated_b64})


async def _api_health(request: Request):
    return JSONResponse({
        "status": "ok",
        "models": ["sam3", "yolov8-flower"],
        "mode": "zerogpu" if ZEROGPU_MODE else f"dedicated-pool-{MODEL_POOL_SIZE}",
    })


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def _gradio_predict(pil_img, tomato_conf, flower_conf, tomato_mode):
    if pil_img is None:
        return None, "Upload an image first."

    img_np = np.array(pil_img.convert("RGB"))

    if tomato_mode == "Segmentation masks":
        seg    = _run_segment(img_np, tomato_conf)
        boxes  = _run_inference(img_np, tomato_conf, flower_conf, mode="flowers")
        # Merge the two annotated images: draw flower boxes on top of the mask image
        flower_boxes = []
        for r in boxes["flowers"]["flowers"]:
            x1, y1, x2, y2 = r["bounding_box"]
            stage = r["stage"]
            flower_boxes.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "color": FLOWER_COLORS.get(stage, (200, 200, 200)),
                "label": f"{FLOWER_CLASSES.get(stage, f'stage_{stage}')} {r['confidence']:.2f}",
            })
        annotated = _draw_boxes(seg["annotated_pil"], flower_boxes)
        tc = {d["label"]: 0 for d in seg["detections"]}
        for d in seg["detections"]:
            tc[d["label"]] = tc.get(d["label"], 0) + 1
        fc = boxes["flowers"]["stage_counts"]
        total_tomatoes = seg["total"]
        total_flowers  = boxes["flowers"]["total_flowers"]
    else:
        result = _run_inference(img_np, tomato_conf, flower_conf, mode="both")
        annotated      = result["annotated_pil"]
        tc             = result["tomatoes"]["summary"]["by_class"]
        fc             = result["flowers"]["stage_counts"]
        total_tomatoes = result["tomatoes"]["summary"]["total"]
        total_flowers  = result["flowers"]["total_flowers"]

    summary = f"""
## Results
### Tomatoes — {total_tomatoes} detected
| Stage | Count |
|---|---|
| 🟢 Unripe | {tc.get('Unripe', 0)} |
| 🟠 Half-Ripe | {tc.get('Half_Ripe', 0)} |
| 🔴 Ripe | {tc.get('Ripe', 0)} |

### Flowers — {total_flowers} detected
| Stage | Count |
|---|---|
| 🔵 Bud | {fc.get('0', 0)} |
| 🟣 Anthesis (ready to pollinate) | {fc.get('1', 0)} |
| 🟤 Post-Anthesis | {fc.get('2', 0)} |
"""
    return annotated, summary


with gr.Blocks(title="Greenhouse Guardians", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌿 Greenhouse Guardians\n**Tomato Ripeness** (SAM3) + **Flower Stage** (YOLOv8) Detection")
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Plant Image")
            tomato_mode = gr.Radio(
                choices=["Bounding boxes", "Segmentation masks"],
                value="Bounding boxes",
                label="Tomato annotation style",
            )
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
    run_btn.click(fn=_gradio_predict,
                  inputs=[image_input, tomato_conf_slider, flower_conf_slider, tomato_mode],
                  outputs=[image_output, results_md])
    gr.Markdown("---\n**API:** `POST /api/tomatoes` · `POST /api/flowers` · `POST /api/segment` · **Compute:** [ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu) (free)")

demo.queue(max_size=20)
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    app_kwargs={
        "routes": [
            Route("/api/segment",  _api_segment,  methods=["POST"]),
            Route("/api/tomatoes", _api_tomatoes, methods=["POST"]),
            Route("/api/flowers",  _api_flowers,  methods=["POST"]),
            Route("/api/health",   _api_health,   methods=["GET"]),
        ]
    },
)
