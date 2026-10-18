"""Task 1: transforms, splits, dataloaders, filtering demo, device utility."""
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.path.join(ROOT, "data")                 # git-ignored; downloaded automatically
SEED = 42                                             # split / subset / init seed (reported in README)
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
FM_MEAN, FM_STD = (0.2860,), (0.3530,)                # Fashion-MNIST train-set statistics
IN_MEAN, IN_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]   # ImageNet statistics


# ---- 1(d) device-agnostic utility --------------------------------------------------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---- 1(b) transforms ---------------------------------------------------------------
def custom_transform(augment):
    """Augmentations are label-preserving for clothing: horizontal flip (garments are roughly
    left-right symmetric), small rotation/translation/scale (camera framing variation).
    NOT used: vertical flip or large rotations - they turn a Trouser into an upside-down, sideways
    shape that never occurs in the data and blurs Trouser/Pullover/Dress distinctions."""
    base = [transforms.ToTensor(), transforms.Normalize(FM_MEAN, FM_STD)]
    if augment:
        aug = [transforms.RandomHorizontalFlip(),
               transforms.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.9, 1.1))]
        return transforms.Compose(aug + base)
    return transforms.Compose(base)


def resnet_transform():
    """4(a): 64x64, 3 channels, ImageNet statistics (the pretrained filters expect that input scale)."""
    return transforms.Compose([transforms.Resize(64), transforms.Grayscale(num_output_channels=3),
                               transforms.ToTensor(), transforms.Normalize(IN_MEAN, IN_STD)])


# ---- 1(c) splits and loaders (custom CNN, 28x28) -----------------------------------
def get_custom_loaders(batch_size=128, augment=True, seed=SEED, num_workers=0):
    """80/20 split of the 60k training images with a fixed seed. Augmentation is applied to the
    TRAIN split only; validation uses the plain transform so it measures performance on the real
    data distribution and stays comparable across runs. 'train_eval' = clean (un-augmented) view of
    the training split, used to measure the generalisation gap fairly."""
    full_aug = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=custom_transform(augment))
    full_plain = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=custom_transform(False))
    g = torch.Generator().manual_seed(seed)
    tr, va = random_split(range(len(full_plain)), [48000, 12000], generator=g)
    gen = torch.Generator().manual_seed(seed)
    mk = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh, num_workers=num_workers,
                                   generator=gen if sh else None)
    return dict(train=mk(Subset(full_aug, tr.indices), True),
                val=mk(Subset(full_plain, va.indices), False),
                train_eval=mk(Subset(full_plain, tr.indices), False))


def get_test_loader(kind="custom", batch_size=128):
    """Official 10k test set. Only called in Task 5 / 6, after all models are frozen."""
    tf = custom_transform(False) if kind == "custom" else resnet_transform()
    ds = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False), ds


# ---- Task 4: stratified subset for transfer learning -------------------------------
def get_transfer_loaders(n_train_per_class=1000, n_val_per_class=200, batch_size=32, seed=SEED):
    """10,000 train / 2,000 val images, stratified by class, disjoint, fixed seed."""
    ds = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=resnet_transform())
    t = np.array(ds.targets)
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for c in range(10):
        idx = rng.permutation(np.where(t == c)[0])
        tr += idx[:n_train_per_class].tolist()
        va += idx[n_train_per_class:n_train_per_class + n_val_per_class].tolist()
    gen = torch.Generator().manual_seed(seed)
    return dict(train=DataLoader(Subset(ds, tr), batch_size=batch_size, shuffle=True, generator=gen),
                val=DataLoader(Subset(ds, va), batch_size=batch_size, shuffle=False))


# ---- 1(a) convolution as filtering -------------------------------------------------
def conv_filter_demo(path):
    ds = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=transforms.ToTensor())
    x = ds[0][0].unsqueeze(0)                                     # (1,1,28,28) in [0,1]
    sx = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernels = {"Sobel-x (high-pass)": sx, "Sobel-y (high-pass)": sx.t(),
               "Gaussian blur (low-pass)": torch.tensor([[1., 2, 1], [2, 4, 2], [1, 2, 1]]) / 16}
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.2))
    ax[0].imshow(x[0, 0], cmap="gray"); ax[0].set_title("original")
    for a, (name, k) in zip(ax[1:], kernels.items()):
        # F.conv2d computes cross-correlation; for the (anti)symmetric kernels here it only flips a sign.
        a.imshow(F.conv2d(x, k.view(1, 1, 3, 3), padding=1)[0, 0], cmap="gray"); a.set_title(name, fontsize=9)
    for a in ax:
        a.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def augmentation_preview(path, n=7):
    ds = datasets.FashionMNIST(DATA_DIR, train=True, download=True)
    img = ds[1][0]
    tf = transforms.Compose([transforms.RandomHorizontalFlip(),
                             transforms.RandomAffine(10, translate=(0.08, 0.08), scale=(0.9, 1.1))])
    fig, ax = plt.subplots(1, n + 1, figsize=(2 * (n + 1), 2.2))
    ax[0].imshow(img, cmap="gray"); ax[0].set_title("original", fontsize=8)
    for a in ax[1:]:
        a.imshow(tf(img), cmap="gray"); a.set_title("augmented", fontsize=8)
    for a in ax:
        a.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
