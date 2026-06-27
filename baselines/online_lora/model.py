import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from collections import deque


class LoRALayer(nn.Module):
    """LoRA adapter: ΔW = B @ A applied as a residual to a frozen linear layer."""

    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.rank = rank
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        # Fisher-based importance (not nn.Parameter — updated manually)
        self.register_buffer('omega_A', torch.zeros_like(self.A))
        self.register_buffer('omega_B', torch.zeros_like(self.B))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.B @ self.A)

    def regularization_loss(self) -> torch.Tensor:
        loss_A = (self.omega_A * self.A ** 2).sum()
        loss_B = (self.omega_B * self.B ** 2).sum()
        return loss_A + loss_B

    @torch.no_grad()
    def update_omega(self, grad_A: torch.Tensor, grad_B: torch.Tensor, momentum: float = 0.9):
        # Normalize by element count so lambda_reg is scale-invariant across tensor sizes
        fisher_A = grad_A ** 2 / max(grad_A.numel(), 1)
        fisher_B = grad_B ** 2 / max(grad_B.numel(), 1)
        self.omega_A = momentum * self.omega_A + (1 - momentum) * fisher_A
        self.omega_B = momentum * self.omega_B + (1 - momentum) * fisher_B


class LoRAAttention(nn.Module):
    """Wraps a timm attention module to add LoRA to Q, K, V projections."""

    def __init__(self, attn: nn.Module, rank: int):
        super().__init__()
        self.attn = attn
        D = attn.qkv.in_features
        out3 = attn.qkv.out_features  # 3*D

        self.lora_qkv = LoRALayer(D, out3, rank)
        self.num_heads = attn.num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        head_dim = D // self.num_heads
        qkv = (self.attn.qkv(x) + self.lora_qkv(x)).reshape(B, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        scale = head_dim ** -0.5
        attn_w = (q @ k.transpose(-2, -1)) * scale
        attn_w = attn_w.softmax(dim=-1)
        attn_w = self.attn.attn_drop(attn_w)

        out = (attn_w @ v).transpose(1, 2).reshape(B, N, D)
        out = self.attn.proj(out)
        out = self.attn.proj_drop(out)
        return out

    def regularization_loss(self) -> torch.Tensor:
        return self.lora_qkv.regularization_loss()

    @torch.no_grad()
    def update_omega(self):
        if self.lora_qkv.A.grad is not None and self.lora_qkv.B.grad is not None:
            self.lora_qkv.update_omega(self.lora_qkv.A.grad, self.lora_qkv.B.grad)


class LoRABlock(nn.Module):
    """ViT block with LoRA-adapted attention."""

    def __init__(self, block: nn.Module, rank: int):
        super().__init__()
        # Store individual submodules directly — do NOT store self.block = block,
        # which would double-register norm1, norm2, mlp, etc. and cause
        # state_dict() key conflicts.
        self.norm1 = block.norm1
        self.norm2 = block.norm2
        self.mlp = block.mlp
        self.ls1 = getattr(block, 'ls1', nn.Identity())
        self.ls2 = getattr(block, 'ls2', nn.Identity())
        self.drop_path1 = getattr(block, 'drop_path1', nn.Identity())
        self.drop_path2 = getattr(block, 'drop_path2', nn.Identity())
        self.lora_attn = LoRAAttention(block.attn, rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.lora_attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

    def regularization_loss(self) -> torch.Tensor:
        return self.lora_attn.regularization_loss()

    @torch.no_grad()
    def update_omega(self):
        self.lora_attn.update_omega()


class HardBuffer:
    """Tiny ring-buffer of (x, y) samples for Fisher estimation."""

    def __init__(self, size: int):
        self.size = size
        self.buf: deque = deque(maxlen=size)

    def add(self, x: torch.Tensor, y: torch.Tensor):
        for xi, yi in zip(x.cpu(), y.cpu()):
            self.buf.append((xi, yi))

    def sample(self, device: torch.device):
        if not self.buf:
            return None, None
        xs = torch.stack([b[0] for b in self.buf]).to(device)
        ys = torch.tensor([b[1] for b in self.buf], device=device)
        return xs, ys


class OnlineLoRAModel(nn.Module):
    def __init__(self, cfg: dict, num_classes: int = 0):
        super().__init__()
        self.cfg = cfg
        rank = cfg['lora_rank']

        backbone = timm.create_model(cfg['backbone'], pretrained=True, num_classes=0)
        for p in backbone.parameters():
            p.requires_grad_(False)

        self.patch_embed = backbone.patch_embed
        self.cls_token = backbone.cls_token
        self.pos_embed = backbone.pos_embed
        self.norm = backbone.norm
        self.blocks = nn.ModuleList([LoRABlock(b, rank) for b in backbone.blocks])

        self.head = nn.Linear(768, num_classes) if num_classes > 0 else None
        self.buffer = HardBuffer(cfg['buffer_size'])

    def grow_head(self, total_classes: int):
        old = self.head
        self.head = nn.Linear(768, total_classes)
        if old is not None:
            with torch.no_grad():
                self.head.weight[:old.out_features] = old.weight
                self.head.bias[:old.out_features] = old.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        tokens = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return self.head(tokens[:, 0]) if self.head is not None else tokens[:, 0]

    def regularization_loss(self) -> torch.Tensor:
        losses = [b.regularization_loss() for b in self.blocks]
        return torch.stack(losses).sum()

    @torch.no_grad()
    def update_omega(self):
        for block in self.blocks:
            block.update_omega()
