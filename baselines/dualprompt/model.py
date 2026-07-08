from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.vision_transformer import Block


class PromptedBlock(nn.Module):
    """Wraps a ViT Block to prepend prompt tokens before the attention computation."""

    def __init__(self, block: Block):
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor, prompt: Optional[torch.Tensor] = None) -> torch.Tensor:
        if prompt is not None:
            # prompt: (B, L_p, D); prepend to sequence (Prompt Tuning / Pro-T)
            x = torch.cat([prompt, x], dim=1)
            
        out = self.block(x)
        
        if prompt is not None:
            # strip prompt tokens from output before passing to next layer
            out = out[:, prompt.shape[1]:]  
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
        
        # In DualPrompt, pool size explicitly matches the number of tasks
        num_tasks = cfg['num_tasks']

        backbone = timm.create_model(cfg['backbone'], pretrained=True, num_classes=0)
        
        # Strictly freeze the backbone as required by the algorithm
        for p in backbone.parameters():
            p.requires_grad_(False)

        self.patch_embed = backbone.patch_embed
        self.cls_token = backbone.cls_token
        self.pos_embed = backbone.pos_embed
        self.norm = backbone.norm
        
        # Wrap blocks to accept prompts
        self.blocks = nn.ModuleList([PromptedBlock(b) for b in backbone.blocks])

        self.g_layers = g_layers
        self.e_layers = e_layers

        # G-Prompt: Task-invariant instruction, one set shared across all tasks
        self.g_prompts = nn.ParameterList([
            nn.Parameter(torch.randn(1, g_len, D) * 0.02) for _ in g_layers
        ])

        # E-Prompt: Task-specific instruction experts
        self.e_prompts = nn.ParameterList([
            nn.Parameter(torch.randn(num_tasks, e_len, D) * 0.02) for _ in e_layers
        ])
        
        # Task Keys: Used to match query features to select the correct E-Prompt
        self.e_keys = nn.Parameter(torch.randn(num_tasks, D) * 0.02)

        self.head = nn.Linear(D, num_classes) if num_classes > 0 else None

    def grow_head(self, total_classes: int):
        """Dynamically expand the classification head for new tasks."""
        D = self.cfg['embedding_dim']
        old = self.head
        self.head = nn.Linear(D, total_classes)
        if old is not None:
            with torch.no_grad():
                self.head.weight[:old.out_features] = old.weight
                self.head.bias[:old.out_features] = old.bias

    def get_query(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the frozen backbone without any prompts to extract the query feature (q(x)).
        This strictly matches the f(x)[0] query formulation in the paper.
        """
        with torch.no_grad():
            B = x.shape[0]
            tokens = self.patch_embed(x)
            cls = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            tokens = tokens + self.pos_embed
            
            # Pass through standard blocks WITHOUT prompts
            for block in self.blocks:
                tokens = block(tokens, prompt=None)
                
            tokens = self.norm(tokens)
            return tokens[:, 0]  # Return [class] token feature

    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass handling both Training (known task) and Inference (unknown task).
        Returns: logits, match_loss
        """
        B = x.shape[0]
        
        # 1. Generate query feature from frozen backbone
        q_x = self.get_query(x)
        
        # 2. Determine Task ID & Compute Match Loss
        if task_id is None:
            # INFERENCE: Guess the task based on closest E-Key
            # q_x: (B, D), e_keys: (num_tasks, D)
            sims = F.cosine_similarity(q_x.unsqueeze(1), self.e_keys.unsqueeze(0), dim=-1)
            task_ids = sims.argmax(dim=-1) # (B,)
            
            # No match loss to optimize during inference
            match_loss = torch.tensor(0.0, device=x.device)
        else:
            # TRAINING: Use ground-truth task ID
            task_ids = torch.full((B,), task_id, dtype=torch.long, device=x.device)
            k_t = self.e_keys[task_ids]
            
            # Equation 3: L_match = 1 - cosine_similarity (we minimize this)
            match_loss = (1.0 - F.cosine_similarity(q_x, k_t, dim=-1)).mean()

        # 3. Prompted Forward Pass
        tokens = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed

        g_iter = 0
        e_iter = 0

        for i, prompted_block in enumerate(self.blocks):
            prompt = None
            
            if i in self.g_layers:
                # G-Prompt is shared, just expand to batch size
                g_p = self.g_prompts[g_iter].expand(B, -1, -1)
                g_iter += 1
                prompt = g_p if prompt is None else torch.cat([prompt, g_p], dim=1)
                
            if i in self.e_layers:
                # E-Prompt is task-specific, gather based on predicted/given task_ids
                # handles varying task_ids across the batch naturally
                e_p = self.e_prompts[e_iter][task_ids]
                e_iter += 1
                prompt = e_p if prompt is None else torch.cat([prompt, e_p], dim=1)
                
            tokens = prompted_block(tokens, prompt)

        tokens = self.norm(tokens)
        cls_out = tokens[:, 0]
        logits = self.head(cls_out) if self.head is not None else cls_out
        
        return logits, match_loss
