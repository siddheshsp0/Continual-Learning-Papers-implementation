from typing import Optional

import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import Block


class PromptedBlock(nn.Module):
    """Wraps a ViT Block to prepend prompt tokens before the attention computation."""

    def __init__(self, block: Block):
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor, prompt: Optional[torch.Tensor] = None) -> torch.Tensor:
        if prompt is not None:
            # prompt: (B, L_p, D); prepend to sequence
            x = torch.cat([prompt, x], dim=1)
        out = self.block(x)
        if prompt is not None:
            out = out[:, prompt.shape[1]:]  # strip prompt tokens from output
        return out


class DualPromptModel(nn.Module):
    def __init__(self, cfg: dict, num_classes: int = 0):
        super().__init__()
        self.cfg = cfg
        D = cfg['embedding_dim']
        g_len = cfg['g_prompt_length']
        e_len = cfg['e_prompt_length']
        g_layers = cfg['g_prompt_layers']
        e_layers = cfg['e_prompt_layers']
        e_pool = cfg['e_pool_size']

        backbone = timm.create_model(cfg['backbone'], pretrained=True, num_classes=0)
        for p in backbone.parameters():
            p.requires_grad_(False)

        self.patch_embed = backbone.patch_embed
        self.cls_token = backbone.cls_token
        self.pos_embed = backbone.pos_embed
        self.norm = backbone.norm
        self.blocks = nn.ModuleList([PromptedBlock(b) for b in backbone.blocks])

        # g_layers and e_layers are sorted lists of 0-indexed block indices;
        # g_iter / e_iter count how many G/E prompt layers have been visited so far.
        self.g_layers = g_layers
        self.e_layers = e_layers

        # G-Prompt: one set shared across all tasks, one per G-layer
        self.g_prompts = nn.ParameterList([
            nn.Parameter(torch.randn(1, g_len, D) * 0.02) for _ in g_layers
        ])

        # E-Prompt pool: e_pool_size experts, one per E-layer
        self.e_prompts = nn.ParameterList([
            nn.Parameter(torch.randn(e_pool, e_len, D) * 0.02) for _ in e_layers
        ])

        self.head = nn.Linear(D, num_classes) if num_classes > 0 else None

    def grow_head(self, total_classes: int):
        D = self.cfg['embedding_dim']
        old = self.head
        self.head = nn.Linear(D, total_classes)
        if old is not None:
            with torch.no_grad():
                self.head.weight[:old.out_features] = old.weight
                self.head.bias[:old.out_features] = old.bias

    def forward(self, x: torch.Tensor, task_id: int = 0) -> torch.Tensor:
        B = x.shape[0]
        tokens = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed

        g_iter = 0
        e_iter = 0

        for i, prompted_block in enumerate(self.blocks):
            prompt = None
            if i in self.g_layers:
                g_p = self.g_prompts[g_iter].expand(B, -1, -1)
                g_iter += 1
                prompt = g_p
            if i in self.e_layers:
                e_p = self.e_prompts[e_iter][task_id].unsqueeze(0).expand(B, -1, -1)
                e_iter += 1
                prompt = e_p if prompt is None else torch.cat([prompt, e_p], dim=1)
            tokens = prompted_block(tokens, prompt)

        tokens = self.norm(tokens)
        cls_out = tokens[:, 0]
        return self.head(cls_out) if self.head is not None else cls_out
