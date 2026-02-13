import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, ngpu=1, latent_dim=100, num_classes=4, ngf=32):
        super().__init__()
        self.ngpu = ngpu
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.ngf = ngf

        self.NoiseBranch = nn.Sequential(
            nn.Linear(in_features=self.latent_dim, out_features=1024*7*7, bias=False),
            nn.ReLU(),
            nn.Unflatten(1, (1024, 7, 7))
        )

        self.LabelBranch = nn.Sequential(
            nn.Embedding(num_classes, 1),
            nn.Linear(in_features=1, out_features=49, bias=False),
            nn.Unflatten(1, (1, 7, 7))
        )

        self.main = nn.Sequential(
            nn.ConvTranspose2d(1025, ngf*16, 5, 2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf*16),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*16, ngf*8, 5, 2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*8, ngf*4, 5, 2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*4, 3, 5, 2, padding=2, output_padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, label, noise):
        noise_branch = self.NoiseBranch(noise)
        label_branch = self.LabelBranch(label)
        combined = torch.cat((noise_branch, label_branch), dim=1)
        return self.main(combined)

class Discriminator(nn.Module):
    def __init__(self, ngpu=1, num_classes=4, nc=3, ndf=32):
        super().__init__()
        self.ngpu = ngpu
        self.num_classes = num_classes
        self.nc = nc
        self.ndf = ndf

        k = 3
        conv_padding = (k // 2 + (k - 2 * (k // 2)) - 1, k // 2)

        self.main = nn.Sequential(
            nn.ZeroPad2d(conv_padding),
            nn.Conv2d(nc, ndf, 3, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.5),

            nn.ZeroPad2d(conv_padding),
            nn.Conv2d(ndf, ndf * 2, 3, 2, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.5),

            nn.ZeroPad2d(conv_padding),
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.5),

            nn.ZeroPad2d(conv_padding),
            nn.Conv2d(ndf * 4, ndf * 8, 3, 2, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.5),

            nn.ZeroPad2d(conv_padding),
            nn.Conv2d(ndf * 8, ndf * 16, 3, 2, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.5),

            nn.Flatten()
        )

        # NOTE: 41472 assumes input 112x112 and this exact conv stack.
        self.dense_val = nn.Linear(41472, 1, bias=False)
        self.dense_label = nn.Linear(41472, num_classes, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        features = self.main(x)
        validity = self.sigmoid(self.dense_val(features))
        label_probs = self.softmax(self.dense_label(features))
        return validity, label_probs

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # This assumes 112x112 input (same as your notebook)
        self.fc1 = nn.Linear(16 * 25 * 25, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 25 * 25)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
