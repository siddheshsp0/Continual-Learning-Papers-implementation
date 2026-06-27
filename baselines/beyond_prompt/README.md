# Beyond Prompt Learning — C-ADA (Continual Adapter)

**Paper**: Beyond Prompt Learning: Continual Adapter for Efficient Rehearsal-Free Continual Learning  
**arXiv**: https://arxiv.org/abs/2407.10281

## Method Summary

C-ADA replaces prompt-based methods with lightweight adapter layers (CAL) inserted after every transformer FFN block. Each CAL performs a bottleneck projection (down → ReLU → up) that is shared across tasks. A Scale&Shift (S&S) module on the CLS token bridges the domain gap between ImageNet pretraining and the downstream dataset. An orthogonality loss on adapter up-projection weights mitigates interference between tasks. No task ID is needed at inference.

## Reproduction Setup

| Setting | Value |
|---|---|
| Backbone | ViT-B/16 (timm `vit_base_patch16_224`, ImageNet-21k pretrained, frozen) |
| Dataset | Split CIFAR-100 (10 tasks × 10 classes) |
| Adapter rank r | 64 |
| Ortho loss weight λ | 0.1 |
| LR | 0.001 (Adam) |
| Batch size | 64 |
| Epochs/task | 20 |

## Run

```bash
pip install -r requirements.txt
bash run.sh
# or
python train.py --config config.yaml
```

## Assumptions

- Single shared CAL per block (not task-expandable in this implementation).
- Orthogonality applied to the up-projection weight matrix per adapter.
- Scale&Shift applied to CLS token only (not all tokens).
- Class-incremental head grown per task; no task ID needed at test time.

## Unsupported Features

- Task-incremental CAL expansion (adding new adapter neurons per task).
- Split ImageNet-R, Split ImageNet-S.
- Domain-incremental evaluation setting.
- Comparison ablations (S&S ablation, CAL ablation).

## Expected Runtime

| Hardware | Time |
|---|---|
| Single A100 GPU | ~20 min (10 tasks × 20 epochs) |
| Single RTX 3090 | ~35 min |
| CPU only | ~4 hours |

## Checkpoint

```python
import torch
from model import ContinualAdapterModel
ckpt = torch.load('checkpoints/beyond_prompt/final.pt')
model = ContinualAdapterModel(ckpt['cfg'])
model.load_state_dict(ckpt['model_state'])
```
