from typing import List
import argparse
import os
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
from pathlib import Path

from model import DualPromptModel

# Include datasets module's path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from datasets.builder import build_dataset

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def evaluate(model: DualPromptModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x, task_id=None) 
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


def train_task(model: DualPromptModel, loader: DataLoader, task_id: int, cfg: dict, device: torch.device, optimizer):
    ce = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(cfg['epochs_per_task']):
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, match_loss = model(x, task_id)

            loss = (
                ce(logits, y)
                + cfg["lambda"] * match_loss
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  epoch {epoch+1}/{cfg['epochs_per_task']}  loss={total_loss/len(loader):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from."
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run evaluation only."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint to evaluate."
    )
    args = parser.parse_args()
    if args.eval and args.checkpoint is None:
        raise ValueError(
            "--checkpoint must be provided when using --eval"
        )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg['seed'])
    generator = torch.Generator()
    generator.manual_seed(cfg["seed"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    splits = build_dataset(cfg)
    num_workers = cfg.get('num_workers', 4)
    os.makedirs(cfg['checkpoint_path'], exist_ok=True)

    model = DualPromptModel(cfg, num_classes=cfg['classes_per_task']).to(device)
    if args.eval:
        total_classes = cfg["num_tasks"] * cfg["classes_per_task"]
        model.grow_head(total_classes)
        model = model.to(device)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        # Evaluate all tasks
        for task_id in range(cfg["num_tasks"]):
            _, test_set = splits[task_id]
            test_loader = DataLoader(
                test_set,
                batch_size=cfg["batch_size"],
                shuffle=False,
                num_workers=num_workers,
                pin_memory=device.type == "cuda",
                persistent_workers=False,
                worker_init_fn=seed_worker,
                generator=generator,
            )

            acc = evaluate(model, test_loader, device)
            print(f"Task {task_id+1}: {acc:.2f}%")

        return
    start_task = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)

        start_task = checkpoint["task_id"] + 1
        total_classes = start_task * cfg["classes_per_task"]

        model.grow_head(total_classes)
        model = model.to(device)
        model.load_state_dict(checkpoint["model_state"])


        print(f"Resuming from task {start_task+1}")   
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        params,
        lr=cfg["lr"],
        betas=[cfg['beta1'], cfg['beta2']]
    )
    if args.resume is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state"]
        )
    task_accs: List[List[float]] = []

    for task_id in range(start_task, cfg['num_tasks']):
        print(f"\n=== Task {task_id+1}/{cfg['num_tasks']} ===")

        total_classes = (task_id + 1) * cfg['classes_per_task']
# Generated to persist momentum for g-prompts (common ones)
        # 1. Store the old head's parameter ID before growing
                # Correct way to preserve momentum while swapping head
        old_head_weight = model.head.weight if model.head is not None else None
        old_head_bias = model.head.bias if model.head is not None and model.head.bias is not None else None
        
        model.grow_head(total_classes)
        model = model.to(device)

        if task_id == start_task and args.resume is None:
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(params, lr=cfg["lr"], betas=[cfg['beta1'], cfg['beta2']])
        else:
            # Delete old states using the tensor references
            if old_head_weight is not None and old_head_weight in optimizer.state:
                del optimizer.state[old_head_weight]
            if old_head_bias is not None and old_head_bias in optimizer.state:
                del optimizer.state[old_head_bias]
                
            new_params = [p for p in model.parameters() if p.requires_grad]
            optimizer.param_groups[0]['params'] = new_params

        # params = [p for p in model.parameters() if p.requires_grad]
        # optimizer = torch.optim.Adam(
        #     params,
        #     lr=cfg["lr"],
        #     betas=[cfg['beta1'], cfg['beta2']]
        # )

        train_set, _ = splits[task_id]
        loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True, num_workers=num_workers, pin_memory=device.type == "cuda",
                            persistent_workers=False,
                            worker_init_fn=seed_worker,
                            generator=generator)
        train_task(model, loader, task_id, cfg, device, optimizer)

        torch.save({'task_id': task_id, 'model_state': model.state_dict(), 'cfg': cfg, "optimizer_state": optimizer.state_dict(),},
                   os.path.join(cfg['checkpoint_path'], f'task_{task_id+1}.pt'))

        accs = []
        for prev_t in range(task_id + 1):
            _, test_set = splits[prev_t]
            test_loader = DataLoader(test_set, batch_size=cfg['batch_size'], shuffle=False, num_workers=num_workers, pin_memory=device.type=="cuda",
                                     persistent_workers=False,
                                     worker_init_fn=seed_worker,
                                     generator=generator)
            acc = evaluate(model, test_loader, device)
            accs.append(acc)
            print(f"  Task {prev_t+1} acc: {acc:.2f}%")
        task_accs.append(accs)

    last_accs = task_accs[-1]
    avg_acc = sum(last_accs) / len(last_accs)
    forgetting_list = [
        max(task_accs[s][t] for s in range(t, cfg['num_tasks'])) - task_accs[-1][t]
        for t in range(cfg['num_tasks'] - 1)
    ]
    avg_forgetting = sum(forgetting_list) / len(forgetting_list) if forgetting_list else 0.0

    print(f"\n=== Final Results ===")
    print(f"Average Accuracy : {avg_acc:.2f}%")
    print(f"Average Forgetting: {avg_forgetting:.2f}%")

    torch.save({
        "task_id": cfg["num_tasks"] - 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg,
    }, os.path.join(cfg["checkpoint_path"], "final.pt"))


if __name__ == '__main__':
    main()
