# datasets/splitter.py

from torch.utils.data import Subset


def split_by_classes(train_set, test_set, num_tasks, classes_per_task):
    task_splits = []

    for task in range(num_tasks):

        classes = range(
            task * classes_per_task,
            (task + 1) * classes_per_task
        )

        train_idx = [
            i for i, y in enumerate(train_set.targets)
            if y in classes
        ]

        test_idx = [
            i for i, y in enumerate(test_set.targets)
            if y in classes
        ]

        task_splits.append((
            Subset(train_set, train_idx),
            Subset(test_set, test_idx),
        ))

    return task_splits
