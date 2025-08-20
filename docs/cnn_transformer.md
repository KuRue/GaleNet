# CNN-Transformer Design Note

This note outlines a hybrid convolutional--transformer architecture for hurricane forecasting.

## Input Channels
- Infrared satellite imagery (IR1, IR2)
- Water vapor channel
- Sea-surface temperature
- U/V wind at 10 m
- Sea-level pressure

## Model Depth
- CNN encoder: 6 convolutional blocks with residual connections
- Transformer core: 4 encoder and 4 decoder layers (model dim = 512, 8 heads)
- Prediction head: 2-layer MLP per forecast step

## Training Schedule
- 50 epochs with cosine learning-rate decay
- Batch size: 16 storms × 6 time steps
- Adam optimizer (LR 3e-4, weight decay 1e-5)
- 10% of training steps reserved for validation

## Target Metrics
Track and intensity errors evaluated over 24–72‑hour forecasts:

| Lead Time | Track RMSE | Intensity RMSE |
|-----------|------------|----------------|
| 24 h      | ≤ 80 km    | ≤ 10 kt        |
| 48 h      | ≤ 120 km   | ≤ 15 kt        |
| 72 h      | ≤ 150 km   | ≤ 20 kt        |

## Resource Budget
- ≤ 300 GPU-hours on 4× A100 40GB
- Peak GPU memory usage ≤ 32 GB per device

