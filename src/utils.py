import os
import random
import numpy as np
import torch

def set_seed(seed: int = 999):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_acc(preds, labels):
    # preds: probability/logits shape [B, C]
    correct = preds.data.max(1)[1].eq(labels.data).cpu().sum()
    return float(correct) / float(len(labels.data)) * 100.0
