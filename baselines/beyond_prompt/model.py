import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def orthogonal_loss(W: torch.Tensor) -> torch.Tensor:
    """||W W^T - I||_F  (W: N x D)."""
    gram = W @ W.T
    I = torch.eye(gram.shape[0], device=W.device)
    return torch.norm(gram - I, p='fro')


class ContinualAdapterLayer(nn.Module):
    """CAL: down-proj D→r, ReLU, up-proj r→D inserted after each FFN block."""

    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.kaiming_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(F.relu(self.down(x)))

    def ortho_loss(self) -> torch.Tensor:
        return orthogonal_loss(self.up.weight)


class AdaptedBlock(nn.Module):
    """Wraps a ViT block and appends a CAL after the FFN (MLP) sub-layer."""

    def __init__(self, block: nn.Module, dim: int, rank: int):
        super().__init__()
        self.block = block
        self.adapter = ContinualAdapterLayer(dim, rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        # adapter residual on every token
        x = x + self.adapter(x)
        return x

    def ortho_loss(self) -> torch.Tensor:
        return self.adapter.ortho_loss()


class ScaleShift(nn.Module):
    """Learnable affine transform on CLS feature to bridge pre-train/downstream gap."""

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma + self.beta


class ContinualAdapterModel(nn.Module):
    def __init__(self, cfg: dict, num_classes: int = 0):
        super().__init__()
        self.cfg = cfg
        D = 768  # ViT-B/16 embed dim
        r = cfg['adapter_rank']

        backbone = timm.create_model(cfg['backbone'], pretrained=True, num_classes=0)
        for p in backbone.parameters():
            p.requires_grad_(False)

        self.patch_embed = backbone.patch_embed
        self.cls_token = backbone.cls_token
        self.pos_embed = backbone.pos_embed
        self.norm = backbone.norm
        self.blocks = nn.ModuleList([AdaptedBlock(b, D, r) for b in backbone.blocks])

        self.scale_shift = ScaleShift(D)
        self.head = nn.Linear(D, num_classes) if num_classes > 0 else None

    def grow_head(self, total_classes: int):
        D = 768
        old = self.head
        self.head = nn.Linear(D, total_classes)
        if old is not None:
            with torch.no_grad():
                self.head.weight[:old.out_features] = old.weight
                self.head.bias[:old.out_features] = old.bias

    def ortho_loss(self) -> torch.Tensor:
        losses = [b.ortho_loss() for b in self.blocks]
        return torch.stack(losses).sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        tokens = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)
        cls_out = self.scale_shift(tokens[:, 0])
        return self.head(cls_out) if self.head is not None else cls_out
