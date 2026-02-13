import argparse
import json
import os

import optuna
import torch
import torch.nn as nn
import torch.optim as optim

from src.models import Generator, Discriminator, weights_init
from src.data import load_imagefolder
from src.utils import get_device, compute_acc, ensure_dir, set_seed

def objective_factory(train_dir, num_classes, lr, beta1, ngpu, image_size, num_workers):
    def objective(trial):
        ngf = trial.suggest_categorical("ngf", [32, 64, 128])
        ndf = trial.suggest_categorical("ndf", [32, 64, 128])
        lambda_class = trial.suggest_float("lambda_class", 0.5, 3.0)
        label_smooth_real = trial.suggest_float("label_smooth_real", 0.7, 1.0)
        nz = trial.suggest_categorical("nz", [64, 100, 128, 256])
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        device = get_device()

        _, dataloader = load_imagefolder(
            train_dir, batch_size=batch_size, num_workers=num_workers,
            image_size=image_size, mode="gan", shuffle=True
        )

        netG = Generator(ngpu=ngpu, latent_dim=nz, num_classes=num_classes, ngf=ngf).to(device)
        netD = Discriminator(ngpu=ngpu, num_classes=num_classes, nc=3, ndf=ndf).to(device)
        netG.apply(weights_init)
        netD.apply(weights_init)

        criterion_source = nn.BCELoss()
        criterion_label = nn.CrossEntropyLoss()

        optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

        try:
            for epoch in range(2):  # short trial
                for i, (real_images, real_labels) in enumerate(dataloader):
                    if i >= 20:
                        break
                    curr_batch = real_images.size(0)
                    real_images = real_images.to(device)
                    real_labels = real_labels.to(device)

                    real_label_tensor = torch.full((curr_batch, 1), label_smooth_real, device=device)
                    fake_label_tensor = torch.full((curr_batch, 1), 0.0, device=device)

                    # D: real
                    netD.zero_grad()
                    source_out_real, class_out_real = netD(real_images)
                    loss_real_source = criterion_source(source_out_real, real_label_tensor)
                    loss_real_class = criterion_label(class_out_real, real_labels)
                    loss_real = loss_real_source + loss_real_class
                    loss_real.backward()

                    # D: fake
                    noise = torch.randn(curr_batch, nz, device=device)
                    fake_labels = torch.randint(0, num_classes, (curr_batch,), device=device)
                    fake_images = netG(fake_labels, noise)

                    source_out_fake, class_out_fake = netD(fake_images.detach())
                    loss_fake_source = criterion_source(source_out_fake, fake_label_tensor)
                    loss_fake_class = criterion_label(class_out_fake, fake_labels)
                    loss_fake = loss_fake_source + loss_fake_class
                    loss_fake.backward()
                    optimizerD.step()

                    # G
                    netG.zero_grad()
                    source_out_gen, class_out_gen = netD(fake_images)
                    loss_gen_source = criterion_source(source_out_gen, real_label_tensor)
                    loss_gen_class = criterion_label(class_out_gen, fake_labels)
                    loss_gen = loss_gen_source + lambda_class * loss_gen_class
                    loss_gen.backward()
                    optimizerG.step()

            acc = compute_acc(class_out_real, real_labels)
            return acc
        except Exception:
            return 0.0

    return objective

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--out_dir", default="artifacts")
    ap.add_argument("--n_trials", type=int, default=20)
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.0002)
    ap.add_argument("--beta1", type=float, default=0.5)
    ap.add_argument("--ngpu", type=int, default=1)
    ap.add_argument("--image_size", type=int, default=112)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=999)
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    study = optuna.create_study(direction="maximize")
    obj = objective_factory(
        args.train_dir, args.num_classes, args.lr, args.beta1, args.ngpu, args.image_size, args.num_workers
    )
    study.optimize(obj, n_trials=args.n_trials)

    best = {"value": float(study.best_trial.value), "params": study.best_trial.params}
    out_path = os.path.join(args.out_dir, "best_params.json")
    with open(out_path, "w") as f:
        json.dump(best, f, indent=2)

    print("Best accuracy:", best["value"])
    print("Saved:", out_path)
    print("Best params:", best["params"])

if __name__ == "__main__":
    main()
