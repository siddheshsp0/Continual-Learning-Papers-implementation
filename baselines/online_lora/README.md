# Online-LoRA — Task-free Online Continual Learning via Low Rank Adaptation

**Paper**: Online-LoRA: Task-free Online Continual Learning via Low Rank Adaptation  
**arXiv**: https://arxiv.org/abs/2411.05663

## Method Summary

Online-LoRA adds low-rank adapter matrices (A, B with rank r) to the QKV projection of each frozen ViT attention block. Parameter importance (Omega) is estimated online using an empirical Fisher approximation computed from current-batch gradients and a tiny hard buffer of 4 samples. A quadratic regularization penalty `λ/2 * Σ(Omega ⊙ W)²` prevents overwriting important weights. No task boundary information is required — this is a task-free online continual learning method.

## Reproduction Setup

| Setting | Value |
|---|---|
| Backbone | ViT-B/16 (timm `vit_base_patch16_224`, ImageNet-21k pretrained, frozen) |
| Dataset | Split CIFAR-100 (10 tasks × 10 classes) |
| LoRA rank r | 4 |
| Regularization λ | 2000 |
| Hard buffer size | 4 samples |
| LR | 0.0002 (Adam) |
| Epochs/task | 1 (online setting) |
| Batch size | 64 |

## Run

```bash
pip install -r requirements.txt
bash run.sh
# or
python train.py --config config.yaml
```

## Assumptions

- LoRA adapters applied to the combined QKV projection (one LoRALayer per attention block).
- Omega updated via exponential moving average of squared gradients (momentum 0.9).
- Hard buffer is a ring buffer updated each batch; buffer samples provide additional gradient signal for Fisher estimation.
- Class-incremental head grown per task.
- `epochs_per_task: 1` reproduces the online (single-pass) training regime from the paper.

## Unsupported Features

- ViT-S/16 backbone.
- Split ImageNet-R, Split ImageNet-S, Split CUB-200, CORe50.
- Loss-surface plateau detection for task boundary detection.
- A_UC metric (requires tracking accuracy at every training step).
- Multi-GPU training.

## Expected Runtime

| Hardware | Time |
|---|---|
| Single A100 GPU | ~5 min (10 tasks × 1 epoch, online) |
| Single RTX 3090 | ~10 min |
| CPU only | ~45 min |

## Checkpoint

```python
import torch
from model import OnlineLoRAModel
ckpt = torch.load('checkpoints/online_lora/final.pt')
model = OnlineLoRAModel(ckpt['cfg'])
model.load_state_dict(ckpt['model_state'])
```
