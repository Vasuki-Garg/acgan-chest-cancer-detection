import argparse
import os

import torch
from torchvision.utils import save_image

from src.models import Generator
from src.data import count_images_per_class
from src.utils import get_device, ensure_dir

def load_generator(checkpoint_path, nz, num_classes, ngf, ngpu):
    device = get_device()
    netG = Generator(ngpu=ngpu, latent_dim=nz, num_classes=num_classes, ngf=ngf).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    netG.load_state_dict(ckpt["netG"])
    netG.eval()
    return netG, device

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train_dir", required=True, help="Real train dir (ImageFolder structure) for class names/counts")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--mode", choices=["fixed", "balance"], default="fixed")
    ap.add_argument("--samples_per_class", type=int, default=500)  # for fixed mode
    ap.add_argument("--batch_gen", type=int, default=32)

    ap.add_argument("--nz", type=int, default=100)
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--ngf", type=int, default=32)
    ap.add_argument("--ngpu", type=int, default=1)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    netG, device = load_generator(args.checkpoint, args.nz, args.num_classes, args.ngf, args.ngpu)

    class_names = [d for d in sorted(os.listdir(args.train_dir)) if os.path.isdir(os.path.join(args.train_dir, d))]
    label_to_class = {i: class_names[i] for i in range(min(args.num_classes, len(class_names)))}

    if args.mode == "fixed":
        for class_id in range(args.num_classes):
            class_name = label_to_class[class_id]
            class_dir = os.path.join(args.out_dir, class_name)
            ensure_dir(class_dir)

            n = args.samples_per_class
            noise = torch.randn(n, args.nz, device=device)
            labels = torch.full((n,), class_id, dtype=torch.long, device=device)

            with torch.no_grad():
                images = netG(labels, noise)

            for i, img in enumerate(images):
                save_image(img, os.path.join(class_dir, f"sample_{i:04d}.png"), normalize=True)

            print(f"✅ Generated {n} for class '{class_name}'")

    else:  # balance
        counts = count_images_per_class(args.train_dir)
        max_count = max(counts.values())
        print("Real counts:", counts)
        print("Max:", max_count)

        # generate only for classes below max
        for class_id in range(args.num_classes):
            class_name = label_to_class[class_id]
            real_count = counts.get(class_name, 0)
            to_generate = max_count - real_count
            if to_generate <= 0:
                print(f"Skip '{class_name}' already max.")
                continue

            class_dir = os.path.join(args.out_dir, class_name)
            ensure_dir(class_dir)
            print(f"⚙️ Generating {to_generate} for '{class_name}'")

            made = 0
            with torch.no_grad():
                while made < to_generate:
                    b = min(args.batch_gen, to_generate - made)
                    noise = torch.randn(b, args.nz, device=device)
                    labels = torch.full((b,), class_id, dtype=torch.long, device=device)
                    imgs = netG(labels, noise)
                    for j, img in enumerate(imgs):
                        save_image(img, os.path.join(class_dir, f"synthetic_{made+j:04d}.png"), normalize=True)
                    made += b

        print("✅ Balanced generation complete")

if __name__ == "__main__":
    main()
