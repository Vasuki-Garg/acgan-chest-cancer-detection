# ACGAN Augmentation + CNN Classification (Chest Cancer, 4 Classes)

This repository contains the pipeline to:
1) Load a 4-class chest cancer dataset in `ImageFolder` format (train/val/test)
2) Train an ACGAN (class-conditional GAN)
3) Tune hyperparameters with Optuna (short trials)
4) Generate synthetic images (fixed-per-class or class-balanced)
5) Train and evaluate a simple CNN baseline classifier

Designed to run in **Google Colab**, with data and outputs stored on **Google Drive**.

---

## Repository Structure

```text
acgan-lung-cancer-augmentation/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ src/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ data.py
│  ├─ models.py
│  ├─ utils.py
│  ├─ tune_optuna.py
│  ├─ train_acgan.py
│  ├─ generate_images.py
│  └─ train_cnn.py
└─ notebooks/
   └─ colab_runner.ipynb

## Dataset Format (Required)
path_to_filtered_dataset/
├─ train/
│  ├─ Adenocarcinoma/
│  ├─ Large Cell Carcinoma/
│  ├─ Normal/
│  └─ Squamous Cell Carcinoma/
├─ val/
└─ test/
