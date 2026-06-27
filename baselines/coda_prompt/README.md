# CODA-Prompt — COntinual Decomposed Attention-based Prompting

**Paper**: CODA-Prompt: COntinual Decomposed Attention-based Prompting for Rehearsal-Free Continual Learning  
**arXiv**: https://arxiv.org/abs/2211.13218  
**Authors**: James Seale Smith, Leonid Karlinsky, Vyshnavi Gutta, Paola Cascante-Bonilla, Donghyun Kim, Assaf Arbelle, Rameswar Panda, Rogerio Feris, Zsolt Kira

## Method Summary

CODA-Prompt decomposes the prompt into M learned components with associated keys. For each input, attention weights are computed between the CLS query and the keys, producing an input-conditioned composed prompt. This prompt is used as a K,V prefix in every transformer attention layer. An orthogonality loss on prompt components and keys prevents interference between tasks. No task ID is required at inference (task-agnostic).

## Reproduction Setup

| Setting | Value |
|---|---|
| Backbone | ViT-B/16 (timm `vit_base_patch16_224`, ImageNet-21k pretrained, frozen) |
| Dataset | Split CIFAR-100 (10 tasks × 10 classes) |
| Components M | 100 |
| Prompt length L_p | 8 tokens (K prefix + V prefix = 2×L_p) |
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

- Same composed prompt is applied to every transformer layer (the paper applies per-layer prompts; this uses shared components for simplicity).
- Orthogonality applied to flattened component vectors and key vectors.
- CLS token at position 0 (before any blocks run) is used as the query to compute attention weights.
- Class-incremental head grown per task; no task ID needed at test time.

## Unsupported Features

- Per-layer distinct prompt component sets.
- Progressive component expansion (adding M/N new components per task).
- Split ImageNet-R, Split ImageNet-S, Split CUB-200.
- Domain-incremental evaluation setting.

## Expected Runtime

| Hardware | Time |
|---|---|
| Single A100 GPU | ~20 min (10 tasks × 20 epochs) |
| Single RTX 3090 | ~35 min |
| CPU only | ~4 hours |

## Checkpoint

```python
import torch
from model import CODAPromptModel
ckpt = torch.load('checkpoints/coda_prompt/final.pt')
model = CODAPromptModel(ckpt['cfg'])
model.load_state_dict(ckpt['model_state'])
```
