# DualPrompt — Complementary Prompting for Rehearsal-free Continual Learning

**Paper**: DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning  
**arXiv**: https://arxiv.org/abs/2204.04799  
**Authors**: Zifeng Wang, Zizhao Zhang, Sayna Ebrahimi, Ruoxi Sun, Han Zhang, Chen-Yu Lee, Xiaoqi Ren, Guolong Su, Vincent Perez, Jarrod Kahn, Tomas Pfister

## Method Summary

DualPrompt decomposes prompt learning into two complementary components inspired by Complementary Learning Systems (CLS) theory. G-Prompts (General) encode task-invariant shared knowledge and are prepended to early transformer layers. E-Prompts (Expert) are task-specific and prepended to later layers, with one expert per task. Both are trained end-to-end with a frozen backbone; task identity is required at test time (task-incremental setting).

## Reproduction Setup

| Setting | Value |
|---|---|
| Backbone | ViT-B/16 (timm `vit_base_patch16_224`, ImageNet-21k pretrained, frozen) |
| Dataset | Split CIFAR-100 (10 tasks × 10 classes) |
| G-Prompt length | 5 tokens, injected at layers 0,1 |
| E-Prompt length | 20 tokens, injected at layers 2,3 |
| E-Prompt pool | 10 experts (one per task) |
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

- Task-incremental setting: task ID is provided at both train and test time.
- E-Prompt experts are indexed directly by task ID (no key-query matching).
- Class-incremental head (grows per task) but evaluation uses task ID to select E-Prompt.
- Prompt tokens are stripped from block output so subsequent blocks receive the original sequence length.

## Unsupported Features

- Task-agnostic evaluation (no task ID at test time).
- Split ImageNet-R, Split ImageNet-S, Split CUB-200.
- Key-based E-Prompt retrieval (paper's full implementation uses learned keys).
- ViT-L/16 backbone variant.

## Expected Runtime

| Hardware | Time |
|---|---|
| Single A100 GPU | ~15 min (10 tasks × 20 epochs) |
| Single RTX 3090 | ~25 min |
| CPU only | ~3 hours |

## Checkpoint

```python
import torch
from model import DualPromptModel
ckpt = torch.load('checkpoints/dualprompt/final.pt')
model = DualPromptModel(ckpt['cfg'])
model.load_state_dict(ckpt['model_state'])
```
