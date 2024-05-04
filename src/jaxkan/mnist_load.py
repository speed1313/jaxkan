import numpy as np
import torchvision


def mnist_transform(x):
    return np.expand_dims(np.array(x, dtype=np.float32), axis=2) / 255.0


def get_dataset_torch(class_num, sample_per_class):
    mnist = {
        "train": torchvision.datasets.MNIST("./data", train=True, download=True),
        "test": torchvision.datasets.MNIST("./data", train=False, download=True),
    }

    ds = {}

    for split in ["train", "test"]:
        # only 0 and 1
        idx_list = []
        for class_number in range(class_num):
            idx = np.where(mnist[split].targets == class_number)[0][:sample_per_class]
            idx_list.append(idx)
        idx = np.concatenate(idx_list)
        ds[split] = {
            "image": mnist[split].data[idx],
            "label": mnist[split].targets[idx],
        }

        ds[split]["image"] = mnist_transform(ds[split]["image"])
        ds[split]["label"] = np.array(ds[split]["label"], dtype=np.int32)
        # expand dim
        ds[split]["image"] = np.expand_dims(ds[split]["image"], axis=3)
    return ds["train"], ds["test"]
