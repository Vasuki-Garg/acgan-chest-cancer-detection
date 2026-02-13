import argparse
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

from src.models import Generator, Discriminator, weights_init
from src.data import load_imagefolder
from src.utils import get_device, ensure_dir, compute_acc, set_seed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--out_dir", default="artifacts")
    ap.add_argument("--generated_dir", default="generated_outputs")
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--model_dir", default="saved_models")
    ap.add_argument("--num_epochs", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--image_size", type=int, default=112)
    ap.add_argument("--num_workers", type=int, default=2)

    ap.add_argument("--nc", type=int, default=3)
    ap.add_argument("--nz", type=int, default=100)
    ap.add_argument("--ngf", type=int, default=32)
    ap.add_argument("--ndf", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.0002)
    ap.add_argument("--beta1", type=float, default=0.5)
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--ngpu", type=int, default=1)

    ap.add_argument("--lambda_class", type=float, default=2.0)
    ap.add_argument("--label_smooth_real", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=999)

    ap.add_argument("--best_params_json", default=None, help="Optional path to artifacts/best_params.json")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()

    # optional: override from best_params.json
    if args.best_params_json and os.path.exists(args.best_params_json):
        with open(args.best_params_json, "r") as f:
            best = json.load(f)["params"]
        args.ngf = best.get("ngf", args.ngf)
        args.ndf = best.get("ndf", args.ndf)
        args.nz = best.get("nz", args.nz)
        args.batch_size = best.get("batch_size", args.batch_size)
        args.lambda_class = best.get("lambda_class", args.lambda_class)
        args.label_smooth_real = best.get("label_smooth_real", args.label_smooth_real)

    ensure_dir(args.generated_dir)
    ensure_dir(os.path.join(args.generated_dir, "classwise"))
    ensure_dir(args.model_dir)
    ensure_dir(args.runs_dir)

    writer = SummaryWriter(log_dir=args.runs_dir)

    _, dataloader = load_imagefolder(
        args.train_dir, batch_size=args.batch_size, num_workers=args.num_workers,
        image_size=args.image_size, mode="gan", shuffle=True
    )

    netG = Generator(ngpu=args.ngpu, latent_dim=args.nz, num_classes=args.num_classes, ngf=args.ngf).to(device)
    netD = Discriminator(ngpu=args.ngpu, num_classes=args.num_classes, nc=args.nc, ndf=args.ndf).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion_source = nn.BCELoss()
    criterion_label = nn.CrossEntropyLoss()

    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # fixed sampling
    eval_noise = torch.randn(args.batch_size, args.nz, device=device)
    eval_label = torch.randint(0, args.num_classes, (args.batch_size,), device=device)

    fixed_noise = torch.randn(args.num_classes, args.nz, device=device)
    fixed_labels = torch.arange(0, args.num_classes, dtype=torch.long, device=device)

    best_accuracy = 0.0
    best_model_path = os.path.join(args.model_dir, "best_netG.pt")
    iters = 0

    for epoch in range(args.num_epochs):
        D_loss_epoch = 0.0
        G_loss_epoch = 0.0

        for i, (real_images, real_labels) in enumerate(dataloader):
            curr_batch = real_images.size(0)
            real_images = real_images.to(device)
            real_labels = real_labels.to(device)

            real_label_tensor = torch.full((curr_batch, 1), args.label_smooth_real, device=device)
            fake_label_tensor = torch.full((curr_batch, 1), 0.0, device=device)

            # --- Train D on real ---
            netD.zero_grad()
            source_out_real, class_out_real = netD(real_images)
            loss_real_source = criterion_source(source_out_real, real_label_tensor)
            loss_real_class = criterion_label(class_out_real, real_labels)
            loss_real = loss_real_source + loss_real_class
            loss_real.backward()

            # --- Train D on fake ---
            noise = torch.randn(curr_batch, args.nz, device=device)
            fake_class_labels = torch.randint(0, args.num_classes, (curr_batch,), device=device)
            fake_images = netG(fake_class_labels, noise)

            source_out_fake, class_out_fake = netD(fake_images.detach())
            loss_fake_source = criterion_source(source_out_fake, fake_label_tensor)
            loss_fake_class = criterion_label(class_out_fake, fake_class_labels)
            loss_fake = loss_fake_source + loss_fake_class
            loss_fake.backward()

            optimizerD.step()
            D_loss_epoch += float((loss_real + loss_fake).item())

            # --- Train G ---
            netG.zero_grad()
            source_out_gen, class_out_gen = netD(fake_images)
            loss_gen_source = criterion_source(source_out_gen, real_label_tensor)
            loss_gen_class = criterion_label(class_out_gen, fake_class_labels)
            loss_gen = loss_gen_source + args.lambda_class * loss_gen_class
            loss_gen.backward()
            optimizerG.step()
            G_loss_epoch += float(loss_gen.item())

            if i % 50 == 0:
                acc = compute_acc(class_out_real, real_labels)
                print(f"Epoch [{epoch+1}/{args.num_epochs}] Step [{i}] "
                      f"D_loss: {(loss_real+loss_fake).item():.4f} "
                      f"G_loss: {loss_gen.item():.4f} Acc: {acc:.2f}%")

                writer.add_scalar('Batch/Disc_Source', (loss_fake_source + loss_real_source).item(), iters)
                writer.add_scalar('Batch/Disc_Class', (loss_fake_class + loss_real_class).item(), iters)
                writer.add_scalar('Batch/Gen', loss_gen.item(), iters)
                writer.add_scalar('Batch/Acc', acc, iters)

                with torch.no_grad():
                    constructed = netG(eval_label, eval_noise)
                vutils.save_image(
                    constructed.data,
                    os.path.join(args.generated_dir, f"result_epoch{epoch:03d}.png"),
                    normalize=True
                )

            iters += 1

        avg_D = D_loss_epoch / max(1, len(dataloader))
        avg_G = G_loss_epoch / max(1, len(dataloader))
        writer.add_scalar('Epoch/D_loss', avg_D, epoch + 1)
        writer.add_scalar('Epoch/G_loss', avg_G, epoch + 1)
        print(f"✔️ Epoch {epoch+1} complete. Avg G: {avg_G:.4f} Avg D: {avg_D:.4f}")

        # classwise samples
        with torch.no_grad():
            class_samples = netG(fixed_labels, fixed_noise).detach().cpu()
        vutils.save_image(
            class_samples,
            os.path.join(args.generated_dir, "classwise", f"classwise_epoch{epoch:03d}.png"),
            nrow=args.num_classes,
            normalize=True
        )

        # save best based on last logged acc (or compute fresh)
        # minimal: reuse acc from latest log; if no log, set to 0
        acc_to_compare = acc if "acc" in locals() else 0.0
        if acc_to_compare > best_accuracy:
            best_accuracy = acc_to_compare
            torch.save({
                'netG': netG.state_dict(),
                'gen_loss': avg_G,
                'accuracy': best_accuracy,
                'epoch': epoch + 1,
                'args': vars(args)
            }, best_model_path)
            print(f"💾 Saved new best netG: {best_accuracy:.2f}% at epoch {epoch+1}")

    print("Done. Best accuracy:", best_accuracy)
    writer.close()

if __name__ == "__main__":
    main()
