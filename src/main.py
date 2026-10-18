"""Reproduces all results/figures:  python src/main.py [--quick] [--bonus]
--quick : tiny epoch/subset budget to smoke-test the pipeline (numbers are NOT for the report)."""
import argparse
import os
import platform
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ablations as abl
from data import (CLASSES, DATA_DIR, SEED, augmentation_preview, conv_filter_demo, get_custom_loaders,
                  get_device, get_test_loader, get_transfer_loaders)
from gradcam import GradCAM, cam_on_object, denorm_gray, overlay
from models import (CustomCNN, count_params, hand_param_count, layer_table, make_resnet18, receptive_field)
from train import FIG_DIR, TAB_DIR, evaluate, fit, make_optimizer, save_table, set_seed

LR_TL = 1e-3


def fig_path(n):
    return os.path.join(FIG_DIR, n)


def hardware_info(device):
    name = torch.cuda.get_device_name(0) if device.type == "cuda" else f"CPU-only ({platform.processor() or platform.machine()}, {torch.get_num_threads()} threads)"
    return name


# ------------------------------------------------------------------ Task 1
def task1():
    conv_filter_demo(fig_path("T1a_filtering.png"))
    augmentation_preview(fig_path("T1b_augmentation.png"))
    L = get_custom_loaders()
    print(f"[Task1] split sizes: train={len(L['train'].dataset)}, val={len(L['val'].dataset)}, seed={SEED}")


# ------------------------------------------------------------------ Task 2
def task2(runs, device):
    key = abl.cfg_name(abl.BASE)
    run = runs[key]
    model, h, cfg = run["model"], run["hist"], run["cfg"]
    # 2(b) bookkeeping
    rows = layer_table(model)
    save_table([{k: str(v) for k, v in r.items()} for r in rows], "T2b_layer_table")
    hand, auto = hand_param_count(cfg), count_params(model)
    msg = f"hand-computed params = {hand}, sum(p.numel()) = {auto}, match = {hand == auto}"
    print("[Task2b]", msg)
    # 2(c) receptive field
    rf, steps = receptive_field(cfg)
    with open(os.path.join(TAB_DIR, "T2bc_bookkeeping.txt"), "w") as f:
        f.write(msg + "\n\nReceptive field of a last-conv unit = %d input pixels\n" % rf + "\n".join(steps) + "\n")
    print(f"[Task2c] receptive field of last conv unit = {rf} px")
    # 2(d) curves
    ep = np.arange(1, len(h["val_loss"]) + 1)
    best_loss_ep = int(np.argmin(h["val_loss"])) + 1
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, h["train_loss"], label="train loss"); ax[0].plot(ep, h["val_loss"], label="val loss")
    ax[0].axvline(best_loss_ep, ls="--", c="gray", label=f"min val loss (epoch {best_loss_ep})")
    ax[0].set(xlabel="epoch", ylabel="loss"); ax[0].legend()
    ax[1].plot(ep, h["val_acc"]); ax[1].set(xlabel="epoch", ylabel="validation accuracy")
    fig.tight_layout(); fig.savefig(fig_path("T2d_training_curves.png"), dpi=140); plt.close(fig)
    print(f"[Task2d] validation loss minimum at epoch {best_loss_ep}")
    # 2(e) learned filters and feature maps
    W = model.features[0].weight.detach().cpu()                    # (16,1,3,3)
    fig, ax = plt.subplots(2, 8, figsize=(12, 3.2))
    for i, a in enumerate(ax.ravel()):
        a.imshow(W[i, 0], cmap="gray"); a.set_title(f"f{i}", fontsize=7); a.axis("off")
    fig.suptitle("Learned first-layer 3x3 kernels"); fig.tight_layout(); fig.savefig(fig_path("T2e_filters.png"), dpi=140); plt.close(fig)
    x, _ = get_custom_loaders()["val"].dataset[0]
    with torch.no_grad():
        fm = model.features[0](x.unsqueeze(0).to(device))[0].cpu()  # pre-ReLU feature maps
    fig, ax = plt.subplots(2, 8, figsize=(12, 3.2))
    for i, a in enumerate(ax.ravel()):
        a.imshow(fm[i], cmap="gray"); a.axis("off")
    fig.suptitle("First-layer feature maps (one validation image)"); fig.tight_layout()
    fig.savefig(fig_path("T2e_feature_maps.png"), dpi=140); plt.close(fig)


# ------------------------------------------------------------------ Task 4
def task4(device, epochs, quick):
    L = get_transfer_loaders(*( (100, 20) if quick else (1000, 200) ))
    m = make_resnet18("feature_extraction").eval()
    with torch.no_grad():
        x = torch.zeros(1, 3, 64, 64)
        feat = torch.nn.Sequential(*list(m.children())[:-2])(x).shape
        out = m(x).shape
    txt = f"64x64 input -> backbone feature map {tuple(feat)} -> AdaptiveAvgPool2d -> logits {tuple(out)}"
    print("[Task4b]", txt)
    results, rows = {}, []
    for s in ("feature_extraction", "gradual", "differential"):
        set_seed(SEED)
        model = make_resnet18(s).to(device)
        h = fit(model, L, epochs, make_optimizer(model, s, LR_TL), device, tag=f"[{s}]")
        results[s] = dict(model=model, hist=h, trainable=count_params(model))
        rows.append(dict(strategy=s, trainable_params=count_params(model), best_val_acc=round(h["best_val_acc"], 4),
                         val_acc_per_epoch=[round(v, 4) for v in h["val_acc"]],
                         sec_per_epoch=round(float(np.mean(h["epoch_time"])), 1)))
    best_s = max(results, key=lambda k: results[k]["hist"]["best_val_acc"])
    # 4(d) cosine schedule on the best strategy
    set_seed(SEED)
    model = make_resnet18(best_s).to(device)
    opt = make_optimizer(model, best_s, LR_TL)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    h = fit(model, L, epochs, opt, device, scheduler=sch, tag=f"[{best_s}+cosine]")
    results[best_s + "+cosine"] = dict(model=model, hist=h, trainable=count_params(model))
    rows.append(dict(strategy=best_s + "+cosine", trainable_params=count_params(model), best_val_acc=round(h["best_val_acc"], 4),
                     val_acc_per_epoch=[round(v, 4) for v in h["val_acc"]], sec_per_epoch=round(float(np.mean(h["epoch_time"])), 1)))
    save_table(rows, "T4_transfer_strategies")
    with open(os.path.join(TAB_DIR, "T4b_resnet_shape.txt"), "w") as f:
        f.write(txt + "\n")
    plt.figure(figsize=(6.5, 4.2))
    for k, r in results.items():
        plt.plot(range(1, epochs + 1), r["hist"]["val_acc"], "o-", label=k)
    plt.xlabel("epoch"); plt.ylabel("validation accuracy (2k subset)"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(fig_path("T4c_strategies.png"), dpi=140); plt.close()
    best_key = max(results, key=lambda k: results[k]["hist"]["best_val_acc"])   # selected on validation only
    print(f"[Task4] best transfer model (validation): {best_key}")
    return results[best_key], best_key


# ------------------------------------------------------------------ Task 5 (test set touched here, once)
def task5(runs, tl_best, device):
    cnn_keys = [k for k, r in runs.items() if r["kind"] == "cnn"]
    cnn_key = max(cnn_keys, key=lambda k: runs[k]["hist"]["best_val_acc"])
    print(f"[Task5] best custom CNN (validation): {cnn_key}")
    tl_c, _ = get_test_loader("custom"); tl_r, ds_r = get_test_loader("resnet", batch_size=64)
    res = {}
    for name, model, loader, hist, params in (
            ("MLP baseline", runs["MLP baseline"]["model"], tl_c, runs["MLP baseline"]["hist"], runs["MLP baseline"]["params"]),
            (f"Custom CNN ({cnn_key})", runs[cnn_key]["model"], tl_c, runs[cnn_key]["hist"], runs[cnn_key]["params"]),
            ("ResNet-18 (best strategy)", tl_best["model"], tl_r, tl_best["hist"], tl_best["trainable"])):
        _, acc, preds, labels = evaluate(model, loader, device)
        res[name] = dict(acc=acc, f1=f1_score(labels, preds, average="macro"), preds=preds, labels=labels,
                         params=params, t=float(np.mean(hist["epoch_time"])))
    save_table([dict(model=n, trainable_params=r["params"], sec_per_epoch=round(r["t"], 2), test_acc=round(r["acc"], 4),
                     macro_f1=round(r["f1"], 4), acc_per_1k_params=round(1000 * r["acc"] / r["params"], 4)) for n, r in res.items()],
               "T5c_efficiency")
    better = max((n for n in res if n != "MLP baseline"), key=lambda n: res[n]["acc"])
    cm = confusion_matrix(res[better]["labels"], res[better]["preds"])
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(10), yticks=range(10), xticklabels=CLASSES, yticklabels=CLASSES, xlabel="predicted", ylabel="true", title=f"Test confusion matrix - {better}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color="w" if cm[i, j] > cm.max() / 2 else "k")
    fig.tight_layout(); fig.savefig(fig_path("T5a_confusion_matrix.png"), dpi=140); plt.close(fig)
    off = cm + cm.T; np.fill_diagonal(off, 0)
    a, b = np.unravel_index(off.argmax(), off.shape)
    print(f"[Task5b] most confused pair: {CLASSES[a]} <-> {CLASSES[b]} ({off[a, b]} errors); test acc {better}: {res[better]['acc']:.4f}")
    # example images of the confused pair (raw 28x28 test images)
    raw = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=transforms.ToTensor())
    fig, ax = plt.subplots(2, 6, figsize=(10, 3.8))
    for r, c in enumerate((a, b)):
        idx = np.where(res[better]["labels"] == c)[0][:6]
        for k, i in enumerate(idx):
            ax[r, k].imshow(raw[i][0][0], cmap="gray"); ax[r, k].set_title(CLASSES[c], fontsize=7); ax[r, k].axis("off")
    fig.tight_layout(); fig.savefig(fig_path("T5b_confused_examples.png"), dpi=140); plt.close(fig)
    return res, better, ds_r, (a, b)


# ------------------------------------------------------------------ Task 6
def task6(tl_best, res_tl, ds_r, pair, device):
    model = tl_best["model"].to(device)
    cam = GradCAM(model, model.layer4[-1])                  # block output (post-ReLU), see gradcam.py
    preds, labels = res_tl["preds"], res_tl["labels"]
    coat = int(np.where((labels == 4) & (preds == 4))[0][0]); shirt = int(np.where((labels == 6) & (preds == 6))[0][0])
    wrong = np.where(np.isin(labels, pair) & (preds != labels) & np.isin(preds, pair))[0]
    if len(wrong) == 0:
        wrong = np.where(preds != labels)[0]
    items = [("correct Coat", coat), ("correct Shirt", shirt)] + [(f"misclassified #{k+1}", int(i)) for k, i in enumerate(wrong[:3])]
    fig, ax = plt.subplots(2, len(items), figsize=(3 * len(items), 6))
    rows = []
    for c, (name, i) in enumerate(items):
        x = ds_r[i][0].unsqueeze(0).to(device)
        heat, _, pred = cam(x)
        img = denorm_gray(x[0])
        ax[0, c].imshow(img, cmap="gray"); ax[0, c].axis("off"); ax[0, c].set_title(f"{name}\ntrue={CLASSES[labels[i]]}", fontsize=8)
        overlay(ax[1, c], img, heat[0], f"pred={CLASSES[pred[0]]}")
        rows.append(dict(sample=name, test_index=i, true=CLASSES[labels[i]], pred=CLASSES[pred[0]], cam_mass_on_garment=round(cam_on_object(img, heat[0]), 3)))
    fig.tight_layout(); fig.savefig(fig_path("T6_gradcam.png"), dpi=140); plt.close(fig)
    save_table(rows, "T6_gradcam_summary")
    cam.remove()


# ------------------------------------------------------------------ driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--bonus", action="store_true")
    args = ap.parse_args()
    device = get_device()
    ep_cnn, ep_tl = (2, 1) if args.quick else (15, 6)
    print(f"Device: {device} ({hardware_info(device)}); seed={SEED}; epochs CNN={ep_cnn}, TL={ep_tl}")
    task1()
    runs, best_key = abl.run_all(device, ep_cnn)                     # Task 3 (also provides Task 2 base model)
    task2(runs, device)
    tl_best, tl_name = task4(device, ep_tl, args.quick)
    res, better, ds_r, pair = task5(runs, tl_best, device)
    task6(tl_best, res["ResNet-18 (best strategy)"], ds_r, pair, device)
    if args.bonus:
        import denoise
        print("[Bonus]", denoise.run(device, epochs=1 if args.quick else 5))
    with open(os.path.join(TAB_DIR, "hardware_and_runtimes.md"), "w") as f:
        f.write(f"Hardware: {hardware_info(device)}\nSeed: {SEED}\n\n| run | mean sec/epoch |\n|---|---|\n")
        for k, r in runs.items():
            f.write(f"| {k} | {np.mean(r['hist']['epoch_time']):.2f} |\n")
        f.write(f"| ResNet-18 {tl_name} (64x64, 10k subset) | {np.mean(tl_best['hist']['epoch_time']):.1f} |\n")
    print("Done. Figures in results/figures, tables in results/tables.")


if __name__ == "__main__":
    main()
