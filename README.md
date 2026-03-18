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

```
POST /api/classify
Content-Type: multipart/form-data
Body: file=<image>  [tomato_conf=0.30]  [flower_conf=0.25]
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
