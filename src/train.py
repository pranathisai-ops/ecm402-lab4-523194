"""Training / validation loops, optimizers, table helpers."""
import copy
import csv
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FIG_DIR, TAB_DIR = os.path.join(ROOT, "results", "figures"), os.path.join(ROOT, "results", "tables")
os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(TAB_DIR, exist_ok=True)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _set_train(model):
    """model.train(), but keep BatchNorm layers whose parameters are frozen in eval mode so the
    pretrained running statistics are not disturbed (feature extraction / frozen backbone)."""
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
            m.eval()


def run_epoch(model, loader, criterion, optimizer, device):
    _set_train(model)
    tot_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item() * len(y); correct += (out.argmax(1) == y).sum().item(); n += len(y)
    return tot_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    tot, correct, n, preds, labels = 0.0, 0, 0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        tot += crit(out, y).item(); p = out.argmax(1)
        correct += (p == y).sum().item(); n += len(y)
        preds.append(p.cpu()); labels.append(y.cpu())
    return tot / n, correct / n, torch.cat(preds).numpy(), torch.cat(labels).numpy()


def make_optimizer(model, strategy=None, lr=1e-3):
    """Adam. For strategy='differential': backbone lr = 0.1 x classifier lr (parameter groups)."""
    if strategy == "differential":
        backbone = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
        head = [p for n, p in model.named_parameters() if n.startswith("fc.")]
        return torch.optim.Adam([{"params": backbone, "lr": 0.1 * lr}, {"params": head, "lr": lr}])
    return torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)


def fit(model, loaders, epochs, optimizer, device, scheduler=None, verbose=True, tag=""):
    """Train on loaders['train'], select by VALIDATION accuracy (best weights restored at the end).
    epoch_time covers training + validation only (not the optional clean train-set evaluation)."""
    crit = nn.CrossEntropyLoss()
    h = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "train_acc_clean", "epoch_time")}
    best_acc, best_state = -1.0, None
    for ep in range(epochs):
        t0 = time.perf_counter()
        trl, tra = run_epoch(model, loaders["train"], crit, optimizer, device)
        vl, va, _, _ = evaluate(model, loaders["val"], device)
        h["epoch_time"].append(time.perf_counter() - t0)
        if scheduler is not None:
            scheduler.step()
        if "train_eval" in loaders:
            h["train_acc_clean"].append(evaluate(model, loaders["train_eval"], device)[1])
        h["train_loss"].append(trl); h["train_acc"].append(tra); h["val_loss"].append(vl); h["val_acc"].append(va)
        if va > best_acc:
            best_acc, best_state, h["best_epoch"] = va, copy.deepcopy(model.state_dict()), ep + 1
        if verbose:
            print(f"{tag} epoch {ep+1:2d}/{epochs}  train {trl:.4f}/{tra:.4f}  val {vl:.4f}/{va:.4f}  {h['epoch_time'][-1]:.1f}s", flush=True)
    model.load_state_dict(best_state)
    h["best_val_acc"] = best_acc
    return h


def save_table(rows, name):
    """Write a list of dicts as results/tables/<name>.csv and .md."""
    keys = list(rows[0].keys())
    with open(os.path.join(TAB_DIR, name + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    with open(os.path.join(TAB_DIR, name + ".md"), "w") as f:
        f.write("| " + " | ".join(keys) + " |\n|" + "---|" * len(keys) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[k]) for k in keys) + " |\n")
