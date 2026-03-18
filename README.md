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

### `POST /api/classify`

Runs SAM3 tomato ripeness detection + YOLOv8 flower stage detection on an uploaded image.

**Request**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `file` | image file | ✅ | — | JPEG/PNG plant image |
| `tomato_conf` | float | ❌ | `0.30` | Confidence threshold for tomato detections (0.10–0.80). Raise to reduce false positives. |
| `flower_conf` | float | ❌ | `0.25` | Confidence threshold for flower detections (0.10–0.80). |

**Example request (curl)**
```bash
curl -X POST https://deenp03-guardians-of-the-greenhouse-inference.hf.space/api/classify \
  -F "file=@plant.jpg" \
  -F "tomato_conf=0.35" \
  -F "flower_conf=0.25"
```

**Example request (Python)**
```python
import httpx

with open("plant.jpg", "rb") as f:
    r = httpx.post(
        "https://deenp03-guardians-of-the-greenhouse-inference.hf.space/api/classify",
        files={"file": f},
        data={"tomato_conf": 0.35, "flower_conf": 0.25},
    )
result = r.json()
```

**Response `200 OK`**
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
  "annotated_image_b64": "<base64-encoded JPEG with bounding boxes drawn>"
}
```

**Tomato labels:** `Unripe` · `Half_Ripe` · `Ripe`

**Flower stages:** `0` = Bud · `1` = Anthesis (ready to pollinate) · `2` = Post-Anthesis

---

### `GET /api/health`

Returns service status.

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
