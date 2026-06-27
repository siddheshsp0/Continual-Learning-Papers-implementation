# L2P — Learning to Prompt for Continual Learning

**Paper**: Learning to Prompt for Continual Learning  
**arXiv**: https://arxiv.org/abs/2112.08654  
**Authors**: Zifeng Wang, Zizhao Zhang, Chen-Yu Lee, Han Zhang, Ruoxi Sun, Xiaoqi Ren, Guolong Su, Vincent Perez, Jarrod Kahn, Tomas Pfister

## Method Summary

L2P maintains a learnable prompt pool paired with keys. At each forward pass, an instance-wise query (CLS embedding) retrieves the top-k most similar prompts via cosine similarity. The retrieved prompts are prepended to the token sequence before the frozen ViT backbone, steering its representations without updating backbone weights. No task identity is required at test time.

## Reproduction Setup

| Setting | Value |
|---|---|
| Backbone | ViT-B/16 (timm `vit_base_patch16_224`, ImageNet-21k pretrained, frozen) |
| Dataset | Split CIFAR-100 (10 tasks × 10 classes) |
| Prompt pool size M | 10 |
| Prompt length L_p | 5 tokens |
| Top-k | 5 |
| Embedding dim | 768 |
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

- Split CIFAR-100 is the only dataset implemented (images resized to 224×224).
- Query vector uses CLS token after the first transformer block; the remaining blocks receive the concatenated prompt+token sequence.
- Class-incremental setting: head grows after each task, all classes share a single classifier.
- Pull loss encourages keys to align with queries of their assigned tasks.

## Unsupported Features

- Split ImageNet-R, Split ImageNet-S, Split CUB-200 (not implemented).
- Frequency-based diversity regularization from the paper's ablation.
- Multi-layer prompt insertion (prompts are only prepended before block 1 onward).
- ViT-S/16 backbone variant.

## Expected Runtime

| Hardware | Time |
|---|---|
| Single A100 GPU | ~15 min (10 tasks × 20 epochs) |
| Single RTX 3090 | ~25 min |
| CPU only | ~3 hours |

## Checkpoint

Checkpoints saved to `checkpoints/l2p/task_<N>.pt` after each task and `final.pt` at the end. Load with:

```python
import torch
from model import L2PModel
ckpt = torch.load('checkpoints/l2p/final.pt')
model = L2PModel(ckpt['cfg'])
model.load_state_dict(ckpt['model_state'])
```
