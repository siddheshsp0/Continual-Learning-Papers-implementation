from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class PromptPool(nn.Module):
    def __init__(self, pool_size: int, prompt_length: int, embedding_dim: int, top_k: int):
        super().__init__()
        self.pool_size = pool_size
        self.prompt_length = prompt_length
        self.top_k = top_k

        self.prompts = nn.Parameter(torch.randn(pool_size, prompt_length, embedding_dim) * 0.02)
        self.keys = nn.Parameter(torch.randn(pool_size, embedding_dim) * 0.02)

    def forward(self, query: torch.Tensor):
        # query: (B, D)
        q_norm = F.normalize(query, dim=-1)
        k_norm = F.normalize(self.keys, dim=-1)
        sim = q_norm @ k_norm.T  # (B, M)

        top_idx = sim.topk(self.top_k, dim=-1).indices  # (B, top_k)

        # selected prompts: (B, top_k, L_p, D)
        selected_prompts = self.prompts[top_idx]  # (B, top_k, L_p, D)
        # flatten prompt tokens: (B, top_k*L_p, D)
        B = query.shape[0]
        selected_prompts = selected_prompts.reshape(B, self.top_k * self.prompt_length, -1)

        # pull loss: maximize similarity between query and selected keys
        selected_keys = self.keys[top_idx]  # (B, top_k, D)
        selected_keys_norm = F.normalize(selected_keys, dim=-1)
        pull_loss = -torch.sum(q_norm.unsqueeze(1) * selected_keys_norm, dim=-1).mean()

        return selected_prompts, pull_loss


class L2PModel(nn.Module):
    def __init__(self, cfg: dict, num_classes: int = 0):
        super().__init__()
        self.cfg = cfg
        embed_dim = cfg['embedding_dim']

        self.backbone = timm.create_model(
            cfg['backbone'], pretrained=True, num_classes=0
        )
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        self.prompt_pool = PromptPool(
            pool_size=cfg['prompt_pool_size'],
            prompt_length=cfg['prompt_length'],
            embedding_dim=embed_dim,
            top_k=cfg['top_k'],
        )

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else None

    def grow_head(self, total_classes: int):
        embed_dim = self.cfg['embedding_dim']
        old_head = self.head
        self.head = nn.Linear(embed_dim, total_classes)
        if old_head is not None:
            with torch.no_grad():
                self.head.weight[:old_head.out_features] = old_head.weight
                self.head.bias[:old_head.out_features] = old_head.bias

    def forward(self, x: torch.Tensor):
        # Step 1: patch embed + positional encoding
        patch_embed = self.backbone.patch_embed(x)
        B, N, D = patch_embed.shape
        cls_token = self.backbone.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_token, patch_embed], dim=1)
        tokens = tokens + self.backbone.pos_embed

        # Use CLS token embedding (before any block) as the query vector
        features = self.backbone.forward_features(x)
        query = features[:, 0]  # (B, D)

        # Step 2: get selected prompts and pull loss
        selected_prompts, pull_loss = self.prompt_pool(query)

        # Step 3: prepend prompts to full token sequence and run ALL blocks
        tokens = torch.cat([selected_prompts, tokens], dim=1)
        for block in self.backbone.blocks:
            tokens = block(tokens)

        tokens = self.backbone.norm(tokens)
        # CLS token is shifted right by the number of prepended prompt tokens
        # cls_out = tokens[:, self.cfg['top_k'] * self.cfg['prompt_length']]
        prompt_tokens = tokens[:, :self.cfg['top_k']*self.cfg['prompt_length']]

        pooled = prompt_tokens.mean(dim=1)

        if self.head is None:
            logits = pooled
        else:
            logits = self.head(pooled)

        # logits = self.head(cls_out) if self.head is not None else cls_out
        return logits, pull_loss
