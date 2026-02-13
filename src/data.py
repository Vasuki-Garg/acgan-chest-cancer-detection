import os
from collections import Counter
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def build_gan_transform(image_size=112):
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.RandomResizedCrop((image_size, image_size), scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

def build_cnn_transform(image_size=112):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Lambda(lambda x: x.convert('RGB')),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

def load_imagefolder(data_dir, batch_size=32, num_workers=2, image_size=112, mode="gan", shuffle=True):
    if mode == "gan":
        tfm = build_gan_transform(image_size=image_size)
    else:
        tfm = build_cnn_transform(image_size=image_size)

    ds = datasets.ImageFolder(root=data_dir, transform=tfm)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return ds, dl

def count_images_per_class(train_dir, image_exts=('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')):
    counts = {}
    for class_name in sorted(os.listdir(train_dir)):
        class_path = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        counts[class_name] = sum(1 for f in os.listdir(class_path) if f.lower().endswith(image_exts))
    return counts

def scan_rgba_images(root_dir, splits=("train", "val", "test")):
    rgba_paths = []
    total = 0
    rgba = 0

    for split in splits:
        split_path = os.path.join(root_dir, split)
        if not os.path.exists(split_path):
            continue

        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if not os.path.isdir(class_path):
                continue

            for f in os.listdir(class_path):
                if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                total += 1
                p = os.path.join(class_path, f)
                try:
                    with Image.open(p) as img:
                        if img.mode == "RGBA":
                            rgba += 1
                            rgba_paths.append((p, img.size))
                except Exception:
                    pass

    return {"total_images": total, "total_rgba": rgba, "rgba_paths": rgba_paths}

def get_class_counts_from_loader(loader):
    targets = loader.dataset.targets
    class_counts = Counter(targets)
    class_names = loader.dataset.classes
    return {class_names[i]: class_counts[i] for i in class_counts}
