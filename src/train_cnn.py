import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.models import SimpleCNN
from src.data import load_imagefolder
from src.utils import get_device, set_seed

def train_model(model, criterion, optimizer, train_loader, val_loader, n_epochs=20, ckpt_path="best_model.pth"):
    device = get_device()
    model.to(device)

    train_losslist, valid_losslist = [], []
    valid_loss_min = np.inf

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * images.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(val_loader.dataset)
        train_losslist.append(train_loss)
        valid_losslist.append(valid_loss)

        print(f"Epoch {epoch}: Train Loss {train_loss:.6f} | Val Loss {valid_loss:.6f}")

        if valid_loss < valid_loss_min:
            print(f"Saving best model: {valid_loss_min:.6f} -> {valid_loss:.6f}")
            torch.save(model.state_dict(), ckpt_path)
            valid_loss_min = valid_loss

    return train_losslist, valid_losslist

def evaluate(model, loader, class_names, show_cm=True):
    device = get_device()
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    cm = confusion_matrix(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)

    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    precision_weighted = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall_weighted = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print("Overall Performance:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro P/R/F1: {precision_macro:.4f} {recall_macro:.4f} {f1_macro:.4f}")
    print(f"Weighted P/R/F1: {precision_weighted:.4f} {recall_weighted:.4f} {f1_weighted:.4f}")

    if show_cm:
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--test_dir", required=True)

    ap.add_argument("--image_size", type=int, default=112)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=999)

    ap.add_argument("--ckpt_path", default="best_model.pth")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()

    train_ds, train_loader = load_imagefolder(args.train_dir, args.batch_size, args.num_workers,
                                             args.image_size, mode="cnn", shuffle=True)
    val_ds, val_loader = load_imagefolder(args.val_dir, args.batch_size, args.num_workers,
                                         args.image_size, mode="cnn", shuffle=False)
    test_ds, test_loader = load_imagefolder(args.test_dir, args.batch_size, args.num_workers,
                                           args.image_size, mode="cnn", shuffle=False)

    class_names = train_ds.classes
    model = SimpleCNN(num_classes=len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    train_loss, val_loss = train_model(model, criterion, optimizer, train_loader, val_loader,
                                       n_epochs=args.epochs, ckpt_path=args.ckpt_path)

    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    evaluate(model, test_loader, class_names, show_cm=True)

if __name__ == "__main__":
    main()
