import argparse
import os
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
from typing import List
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

from model import OnlineLoRAModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_split_cifar100(root: str, num_tasks: int, classes_per_task: int):
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    train_tf = T.Compose([T.Resize(224), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])
    test_tf = T.Compose([T.Resize(224), T.ToTensor(), T.Normalize(mean, std)])
    train_full = torchvision.datasets.CIFAR100(root, train=True, download=True, transform=train_tf)
    test_full = torchvision.datasets.CIFAR100(root, train=False, download=True, transform=test_tf)
    splits = []
    for t in range(num_tasks):
        classes = list(range(t * classes_per_task, (t + 1) * classes_per_task))
        tr_idx = [i for i, (_, y) in enumerate(train_full) if y in classes]
        te_idx = [i for i, (_, y) in enumerate(test_full) if y in classes]
        splits.append((Subset(train_full, tr_idx), Subset(test_full, te_idx)))
    return splits


def evaluate(model: OnlineLoRAModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


def train_task(model: OnlineLoRAModel, loader: DataLoader, cfg: dict, device: torch.device,
               task_id: int = 0):
    # Regularization protects previous-task knowledge; skip on first task (omega still zero)
    lam = cfg['lambda_reg'] if task_id > 0 else 0.0
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg['lr'])
    ce = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(cfg['epochs_per_task']):
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # --- main batch forward + backward ---
            opt.zero_grad()
            logits = model(x)
            reg = model.regularization_loss() if lam > 0 else torch.tensor(0.0, device=device)
            loss = ce(logits, y) + (lam / 2.0) * reg
            loss.backward()
            # update Fisher importance from main-batch gradients (grads still live)
            model.update_omega()
            # optimizer step using main-batch gradients
            opt.step()

            # add current batch to replay buffer
            model.buffer.add(x.detach(), y.detach())

            # --- buffer pass: update Fisher only, no optimizer step ---
            bx, by = model.buffer.sample(device)
            if bx is not None:
                model.zero_grad()
                buf_logits = model(bx)
                buf_loss = ce(buf_logits, by)
                buf_loss.backward()
                model.update_omega()
                model.zero_grad()  # discard buffer grads — Fisher already captured

            total_loss += loss.item()

        print(f"  epoch {epoch+1}/{cfg['epochs_per_task']}  loss={total_loss/len(loader):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    splits = get_split_cifar100(cfg['data_root'], cfg['num_tasks'], cfg['classes_per_task'])
    os.makedirs(cfg['checkpoint_path'], exist_ok=True)

    model = OnlineLoRAModel(cfg, num_classes=cfg['classes_per_task']).to(device)
    task_accs: List[List[float]] = []

    num_workers = cfg.get('num_workers', 4)

    for task_id in range(cfg['num_tasks']):
        print(f"\n=== Task {task_id+1}/{cfg['num_tasks']} ===")

        total_classes = (task_id + 1) * cfg['classes_per_task']
        model.grow_head(total_classes)
        model = model.to(device)

        train_set, _ = splits[task_id]
        loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True, num_workers=num_workers, pin_memory=True)
        train_task(model, loader, cfg, device, task_id=task_id)

        torch.save({'task_id': task_id, 'model_state': model.state_dict(), 'cfg': cfg},
                   os.path.join(cfg['checkpoint_path'], f'task_{task_id+1}.pt'))

        accs = []
        for prev_t in range(task_id + 1):
            _, test_set = splits[prev_t]
            test_loader = DataLoader(test_set, batch_size=cfg['batch_size'], shuffle=False, num_workers=num_workers, pin_memory=True)
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

    torch.save({'model_state': model.state_dict(), 'cfg': cfg},
               os.path.join(cfg['checkpoint_path'], 'final.pt'))


if __name__ == '__main__':
    main()
