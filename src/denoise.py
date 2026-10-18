"""Task 8 (bonus): CNN encoder-decoder denoiser vs 3x3 Gaussian blur; PSNR before/after."""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import DATA_DIR, SEED
from train import set_seed, save_table, FIG_DIR


class DenoiseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU())
        self.dec = nn.Sequential(nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
                                 nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid())

    def forward(self, x):
        return self.dec(self.enc(x))


def psnr(a, b):
    mse = ((a - b) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return (10 * torch.log10(1.0 / mse)).mean().item()


def noisy(x, sigma, gen=None):
    return (x + sigma * torch.randn(x.shape, generator=gen)).clamp(0, 1)


def run(device, epochs=5):
    tf = transforms.ToTensor()
    tr = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=tf)
    te = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=tf)
    xte = torch.stack([te[i][0] for i in range(len(te))])
    blur_k = (torch.tensor([[1., 2, 1], [2, 4, 2], [1, 2, 1]]) / 16).view(1, 1, 3, 3)
    rows, fig, ax = [], *plt.subplots(2, 4, figsize=(10, 5.2))
    for r, sigma in enumerate((0.1, 0.3)):
        set_seed(SEED)
        net = DenoiseNet().to(device); opt = torch.optim.Adam(net.parameters(), 1e-3)
        for _ in range(epochs):
            for x, _y in DataLoader(tr, batch_size=128, shuffle=True):
                x = x.to(device)
                loss = F.mse_loss(net(noisy(x.cpu(), sigma).to(device)), x)
                opt.zero_grad(); loss.backward(); opt.step()
        g = torch.Generator().manual_seed(SEED)
        xn = noisy(xte, sigma, g)
        with torch.no_grad():
            xd = net.eval()(xn.to(device)).cpu()
        xb = F.conv2d(xn, blur_k, padding=1)
        rows.append(dict(sigma=sigma, psnr_noisy=round(psnr(xn, xte), 2), psnr_gaussian_blur=round(psnr(xb, xte), 2),
                         psnr_cnn=round(psnr(xd, xte), 2)))
        for c, (im, t) in enumerate(((xte[0], "clean"), (xn[0], f"noisy σ={sigma}"), (xb[0], "Gaussian blur"), (xd[0], "CNN"))):
            ax[r, c].imshow(im[0], cmap="gray", vmin=0, vmax=1); ax[r, c].set_title(t, fontsize=9); ax[r, c].axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "T8_denoising.png"), dpi=140); plt.close(fig)
    save_table(rows, "T8_denoising_psnr")
    return rows
