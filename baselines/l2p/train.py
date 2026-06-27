import argparse
import os
import random
from typing import List
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

from model import L2PModel


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
    train_tf = T.Compose([
        T.Resize(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tf = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    train_full = torchvision.datasets.CIFAR100(root, train=True, download=True, transform=train_tf)
    test_full = torchvision.datasets.CIFAR100(root, train=False, download=True, transform=test_tf)

    task_splits = []
    for t in range(num_tasks):
        classes = list(range(t * classes_per_task, (t + 1) * classes_per_task))
        train_idx = [i for i, (_, y) in enumerate(train_full) if y in classes]
        test_idx = [i for i, (_, y) in enumerate(test_full) if y in classes]
        task_splits.append((Subset(train_full, train_idx), Subset(test_full, test_idx)))
    return task_splits


def evaluate(model: L2PModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


def train_task(model: L2PModel, loader: DataLoader, cfg: dict, device: torch.device):
    pull_w = cfg['pull_loss_weight']
    lr = cfg['lr']
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    ce = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(cfg['epochs_per_task']):
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, pull_loss = model(x)
            loss = ce(logits, y) + pull_w * pull_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
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

    task_splits = get_split_cifar100(
        cfg['data_root'], cfg['num_tasks'], cfg['classes_per_task']
    )

    os.makedirs(cfg['checkpoint_path'], exist_ok=True)

    model = L2PModel(cfg, num_classes=cfg['classes_per_task'])
    model = model.to(device)

    num_workers = cfg.get('num_workers', 4)
    task_accs: List[List[float]] = []  # task_accs[t][i] = acc on task i after training task t

    for task_id in range(cfg['num_tasks']):
        print(f"\n=== Task {task_id+1}/{cfg['num_tasks']} ===")

        total_classes = (task_id + 1) * cfg['classes_per_task']
        model.grow_head(total_classes)
        model = model.to(device)

        train_set, _ = task_splits[task_id]
        train_loader = DataLoader(
            train_set, batch_size=cfg['batch_size'], shuffle=True,
            num_workers=num_workers, pin_memory=True, persistent_workers=False
        )

        train_task(model, train_loader, cfg, device)

        # checkpoint
        ckpt_path = os.path.join(cfg['checkpoint_path'], f'task_{task_id+1}.pt')
        torch.save({
            'task_id': task_id,
            'model_state': model.state_dict(),
            'cfg': cfg,
        }, ckpt_path)

        # evaluate all tasks seen so far
        accs = []
        for prev_t in range(task_id + 1):
            _, test_set = task_splits[prev_t]
            test_loader = DataLoader(
                test_set, batch_size=cfg['batch_size'], shuffle=False,
                num_workers=num_workers, pin_memory=True, persistent_workers=False
            )
            acc = evaluate(model, test_loader, device)
            accs.append(acc)
            print(f"  Task {prev_t+1} acc: {acc:.2f}%")
        task_accs.append(accs)

    # compute Average Accuracy and Forgetting
    last_accs = task_accs[-1]
    avg_acc = sum(last_accs) / len(last_accs)

    forgetting_list = []
    for t in range(cfg['num_tasks'] - 1):
        best = max(task_accs[s][t] for s in range(t, cfg['num_tasks']))
        last = task_accs[-1][t]
        forgetting_list.append(best - last)
    avg_forgetting = sum(forgetting_list) / len(forgetting_list) if forgetting_list else 0.0

    print(f"\n=== Final Results ===")
    print(f"Average Accuracy : {avg_acc:.2f}%")
    print(f"Average Forgetting: {avg_forgetting:.2f}%")

    torch.save({'model_state': model.state_dict(), 'cfg': cfg},
               os.path.join(cfg['checkpoint_path'], 'final.pt'))


if __name__ == '__main__':
    main()
