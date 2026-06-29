# datasets/transforms.py

import torchvision.transforms as T


def vit_train_transform():

    return T.Compose([
        T.Resize(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(
            (0.5071,0.4867,0.4408),
            (0.2675,0.2565,0.2761)
        ),
    ])


def vit_test_transform():

    return T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(
            (0.5071,0.4867,0.4408),
            (0.2675,0.2565,0.2761)
        ),
    ])
