# ACGAN Augmentation + CNN Classification (Lung Cancer, 4 Classes)

This repository contains the pipeline to:
1) Load a 4-class lung cancer dataset in `ImageFolder` format (train/val/test)
2) Train an ACGAN (class-conditional GAN)
3) Tune hyperparameters with Optuna (short trials)
4) Generate synthetic images (fixed-per-class or class-balanced)
5) Train and evaluate a simple CNN baseline classifier

Designed to run in **Google Colab**, with data and outputs stored on **Google Drive**.

---

## Repository Structure

