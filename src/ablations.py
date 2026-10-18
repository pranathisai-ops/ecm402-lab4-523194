"""Task 3: controlled ablations. Seed, optimizer, LR, epochs identical across variants; one factor changes."""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import SEED, get_custom_loaders
from models import CustomCNN, MLP, count_params
from train import set_seed, make_optimizer, fit, save_table, FIG_DIR

LR = 1e-3
BASE = dict(n_blocks=2, pool="max")


def _run(kind, cfg, augment, epochs, device, tag):
    set_seed(SEED)
    loaders = get_custom_loaders(augment=augment, seed=SEED)
    set_seed(SEED)                                   # identical initialisation for identical architectures
    model = (MLP() if kind == "mlp" else CustomCNN(**cfg)).to(device)
    hist = fit(model, loaders, epochs, make_optimizer(model, lr=LR), device, tag=tag)
    return dict(model=model, hist=hist, cfg=cfg, augment=augment, kind=kind, params=count_params(model))


def cfg_name(cfg):
    return f"CNN pool={cfg['pool']}, blocks={cfg['n_blocks']}"


def run_all(device, epochs=15):
    runs = {}
    runs["MLP baseline"] = _run("mlp", None, False, epochs, device, "[MLP]")
    variants = [BASE, dict(n_blocks=2, pool="avg"), dict(n_blocks=2, pool="none"),
                dict(n_blocks=1, pool="max"), dict(n_blocks=3, pool="max"), dict(n_blocks=4, pool="max")]
    for cfg in variants:
        runs[cfg_name(cfg)] = _run("cnn", cfg, False, epochs, device, f"[{cfg_name(cfg)}]")

    # 3(b) pooling figure + 3(c) depth figure
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for pool in ("max", "avg", "none"):
        ax[0].plot(runs[cfg_name(dict(n_blocks=2, pool=pool))]["hist"]["val_acc"], label=f"pool={pool}")
    ax[0].set(xlabel="epoch", ylabel="validation accuracy", title="3(b) Pooling variants"); ax[0].legend()
    depths = [1, 2, 3, 4]
    ax[1].plot(depths, [runs[cfg_name(dict(n_blocks=d, pool="max"))]["hist"]["best_val_acc"] for d in depths], "o-")
    ax[1].set(xlabel="number of conv blocks", ylabel="best validation accuracy", title="3(c) Depth"); ax[1].set_xticks(depths)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "T3bc_pooling_depth.png"), dpi=140); plt.close(fig)

    # 3(d) augmentation on the best architecture (selected on validation accuracy)
    cnn_keys = [k for k, r in runs.items() if r["kind"] == "cnn"]
    best_key = max(cnn_keys, key=lambda k: runs[k]["hist"]["best_val_acc"])
    best_cfg = runs[best_key]["cfg"]
    runs[best_key + " + augmentation"] = _run("cnn", best_cfg, True, epochs, device, "[best+aug]")
    a, b = runs[best_key]["hist"], runs[best_key + " + augmentation"]["hist"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(a["val_acc"], label="no augmentation"); ax[0].plot(b["val_acc"], label="augmentation")
    ax[0].set(xlabel="epoch", ylabel="validation accuracy", title="Validation accuracy"); ax[0].legend()
    gap = lambda h: np.array(h["train_acc_clean"]) - np.array(h["val_acc"])
    ax[1].plot(gap(a), label="no augmentation"); ax[1].plot(gap(b), label="augmentation")
    ax[1].set(xlabel="epoch", ylabel="train acc (clean) - val acc", title="Generalisation gap"); ax[1].legend()
    ax[2].plot(a["train_loss"], label="no augmentation"); ax[2].plot(b["train_loss"], label="augmentation")
    ax[2].set(xlabel="epoch", ylabel="training loss", title="Training loss"); ax[2].legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "T3d_augmentation.png"), dpi=140); plt.close(fig)

    rows = []
    for k, r in runs.items():
        h = r["hist"]
        rows.append(dict(variant=k, params=r["params"], best_val_acc=round(h["best_val_acc"], 4),
                         best_epoch=h["best_epoch"], sec_per_epoch=round(float(np.mean(h["epoch_time"])), 2),
                         final_gap=round(float(h["train_acc_clean"][-1] - h["val_acc"][-1]), 4)))
    save_table(rows, "T3_ablation_summary")
    return runs, best_key
