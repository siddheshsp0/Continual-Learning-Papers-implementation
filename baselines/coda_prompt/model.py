import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional


def orthogonal_loss(W: torch.Tensor) -> torch.Tensor:
    """||W W^T - I||_F for W of shape (N, D)."""
    gram = W @ W.T
    I = torch.eye(gram.shape[0], device=W.device)
    return torch.norm(gram - I, p='fro')


class CODAPromptAttention(nn.Module):
    """Replaces one ViT attention block's QKV with prefix-tuned K,V from composed prompts."""

    def __init__(self, attn: nn.Module, prompt_length: int, embedding_dim: int):
        super().__init__()
        self.attn = attn
        self.prompt_length = prompt_length
        self.num_heads = attn.num_heads
        self.head_dim = embedding_dim // attn.num_heads
        self.prompt_k_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.prompt_v_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        with torch.no_grad():
            nn.init.eye_(self.prompt_k_proj.weight)
            nn.init.eye_(self.prompt_v_proj.weight)

    def forward(self, x: torch.Tensor, prompt_kv: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.attn.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if prompt_kv is not None:
            L_p = self.prompt_length
            pk = prompt_kv[:, :L_p, :]
            pv = prompt_kv[:, L_p:, :]
            pk = self.prompt_k_proj(pk).reshape(B, L_p, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            pv = self.prompt_v_proj(pv).reshape(B, L_p, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        scale = self.head_dim ** -0.5
        attn_w = (q @ k.transpose(-2, -1)) * scale
        attn_w = attn_w.softmax(dim=-1)
        attn_w = self.attn.attn_drop(attn_w)
        out = (attn_w @ v).transpose(1, 2).reshape(B, N, D)
        out = self.attn.proj(out)
        out = self.attn.proj_drop(out)
        return out


class CODAPromptBlock(nn.Module):
    """ViT block with CODA-Prompt prefix injection."""

    def __init__(self, block: nn.Module, prompt_length: int, embedding_dim: int):
        super().__init__()
        self.norm1 = block.norm1
        self.norm2 = block.norm2
        self.mlp = block.mlp
        self.ls1 = getattr(block, 'ls1', nn.Identity())
        self.ls2 = getattr(block, 'ls2', nn.Identity())
        self.drop_path1 = getattr(block, 'drop_path1', nn.Identity())
        self.drop_path2 = getattr(block, 'drop_path2', nn.Identity())
        self.coda_attn = CODAPromptAttention(block.attn, prompt_length, embedding_dim)

    def forward(self, x: torch.Tensor, prompt_kv: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.coda_attn(self.norm1(x), prompt_kv)
        x = x + self.drop_path1(self.ls1(attn_out))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class CODAPromptModel(nn.Module):
    def __init__(self, cfg: dict, num_classes: int = 0):
        super().__init__()
        self.cfg = cfg
        M = cfg['num_components']
        L_p = cfg['prompt_length']
        D = cfg['embedding_dim']

        backbone = timm.create_model(cfg['backbone'], pretrained=True, num_classes=0)
        for p in backbone.parameters():
            p.requires_grad_(False)

        self.patch_embed = backbone.patch_embed
        self.cls_token = backbone.cls_token
        self.pos_embed = backbone.pos_embed
        self.norm = backbone.norm
        self.blocks = nn.ModuleList([
            CODAPromptBlock(b, L_p, D) for b in backbone.blocks
        ])

        self.prompt_components = nn.Parameter(torch.randn(M, 2 * L_p, D) * 0.02)
        self.prompt_keys = nn.Parameter(torch.randn(M, D) * 0.02)

        self.head = nn.Linear(D, num_classes) if num_classes > 0 else None

    def grow_head(self, total_classes: int):
        D = self.cfg['embedding_dim']
        old = self.head
        self.head = nn.Linear(D, total_classes)
        if old is not None:
            with torch.no_grad():
                self.head.weight[:old.out_features] = old.weight
                self.head.bias[:old.out_features] = old.bias

    def compose_prompt(self, cls_query: torch.Tensor) -> torch.Tensor:
        q_norm = F.normalize(cls_query, dim=-1)
        k_norm = F.normalize(self.prompt_keys, dim=-1)
        attn = F.softmax(q_norm @ k_norm.T, dim=-1)
        composed = torch.einsum('bm,mld->bld', attn, self.prompt_components)
        return composed

    def ortho_loss(self) -> torch.Tensor:
        P = self.prompt_components.flatten(1)
        K = self.prompt_keys
        return orthogonal_loss(P) + orthogonal_loss(K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        tokens = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed

        cls_query = tokens[:, 0]
        prompt_kv = self.compose_prompt(cls_query)

        for block in self.blocks:
            tokens = block(tokens, prompt_kv)

        tokens = self.norm(tokens)
        cls_out = tokens[:, 0]
        return self.head(cls_out) if self.head is not None else cls_out
