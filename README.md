# ACGAN Augmentation + CNN Classification (Chest Cancer, 4 Classes)

This repository contains the pipeline to:
1) Load a 4-class chest cancer dataset in `ImageFolder` format (train/val/test)
2) Train an ACGAN (class-conditional GAN)
3) Tune hyperparameters with Optuna (short trials)
4) Generate synthetic images (fixed-per-class or class-balanced)
5) Train and evaluate a simple CNN baseline classifier

Designed to run in **Google Colab**, with data and outputs stored on **Google Drive**.

---

## Credit
This code is adapted from the official repository:
[- https://github.com/harrylui1995/ASP_E2EPO](https://github.com/yacinebouaouni/ACGAN-Xray-Generation-Tensorflow)

Related paper:
Waheed, A., Goyal, M., Gupta, D., Khanna, A., Al-Turjman, F., & Pinheiro, P. R. (2020). CovidGAN: Data augmentation using auxiliary classifier GAN for improved COVID-19 detection. IEEE Access, 8, 91916–91923. https://doi.org/10.1109/ACCESS.2020.2994762

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
