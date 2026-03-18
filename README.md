---
title: Guardians of the Greenhouse Inference
emoji: 🍅
colorFrom: green
colorTo: red
sdk: gradio
sdk_version: 5.25.0
app_file: app.py
pinned: false
license: mit
---

# Guardians of the Greenhouse

Tomato ripeness detection (SAM3) + flower pollination stage detection (YOLOv8).

## API

Base URL: `https://deenp03-guardians-of-the-greenhouse-inference.hf.space`

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/segment` | SAM3 precise polygon masks (tomatoes) |
| `POST` | `/api/tomatoes` | SAM3 tomato ripeness bounding boxes |
| `POST` | `/api/flowers` | YOLOv8 flower stage bounding boxes |
| `GET` | `/api/health` | Service status |

---

### Shared request format

All `POST` endpoints accept `multipart/form-data`:

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `file` | image file | ✅ | — | JPEG/PNG plant image |
| `tomato_conf` | float | ❌ | `0.30` | SAM3 confidence threshold (0.10–0.80). Raise to reduce false positives. |
| `flower_conf` | float | ❌ | `0.25` | YOLOv8 confidence threshold (0.10–0.80). |

> `tomato_conf` is ignored by `/api/flowers` and `flower_conf` is ignored by `/api/tomatoes`.

---

### `POST /api/segment`

Returns SAM3's precise polygon contours around each tomato instead of bounding boxes. Uses the same SAM3 model but exposes the actual segmentation masks.

**Example**
```bash
curl -X POST .../api/segment -F "file=@plant.jpg" -F "tomato_conf=0.35"
```

**Response**
```json
{
  "total": 3,
  "detections": [
    {
      "class_id": 0,
      "label": "Unripe",
      "confidence": 0.8923,
      "polygon": [[120.5, 80.2], [135.1, 78.4], [150.3, 90.1], "..."]
    }
  ],
  "annotated_image_b64": "<base64 JPEG with polygon outlines>"
}
```

`polygon` is a list of `[x, y]` pixel coordinate pairs tracing the exact boundary of the tomato.

---

### `POST /api/tomatoes`

**Example**
```bash
curl -X POST .../api/tomatoes -F "file=@plant.jpg" -F "tomato_conf=0.35"
```

**Response**
```json
{
  "tomatoes": {
    "detections": [
      {
        "class_id": 0,
        "label": "Unripe",
        "confidence": 0.8923,
        "bbox": { "x1": 120.5, "y1": 80.2, "x2": 210.3, "y2": 175.6 }
      }
    ],
    "summary": {
      "total": 3,
      "by_class": { "Ripe": 0, "Half_Ripe": 1, "Unripe": 2 }
    }
  },
  "annotated_image_b64": "<base64 JPEG>"
}
```

**Labels:** `Unripe` · `Half_Ripe` · `Ripe`

---

### `POST /api/flowers`

**Example**
```bash
curl -X POST .../api/flowers -F "file=@plant.jpg" -F "flower_conf=0.25"
```

**Response**
```json
{
  "flowers": {
    "flowers": [
      {
        "bounding_box": [340.1, 220.4, 410.7, 295.2],
        "stage": 1,
        "confidence": 0.7654
      }
    ],
    "total_flowers": 1,
    "stage_counts": { "0": 0, "1": 1, "2": 0 }
  },
  "annotated_image_b64": "<base64 JPEG>"
}
```

**Stages:** `0` = Bud · `1` = Anthesis (ready to pollinate) · `2` = Post-Anthesis

---

### `GET /api/health`

```json
{ "status": "ok", "models": ["sam3", "yolov8-flower"], "mode": "zerogpu" }
```

## Scaling / Hardware

| Mode | How to enable | Hardware |
|---|---|---|
| ZeroGPU (default, free) | No env var needed | Free H200 via ZeroGPU |
| Dedicated GPU — 4 concurrent | Set `MODEL_POOL_SIZE=4` as a Space secret | 1× L40S ($1.80/hr) recommended |

To switch to dedicated GPU:
1. Go to Space Settings → Hardware → select **Nvidia 1xL40S**
2. Go to Space Settings → Variables and secrets → add `MODEL_POOL_SIZE` = `4`
3. Restart the Space — it will pre-load 4 SAM3 instances at startup for true parallel inference
