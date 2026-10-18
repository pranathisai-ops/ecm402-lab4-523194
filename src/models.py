"""Custom CNN, MLP baseline, ResNet-18 setup, and by-hand bookkeeping utilities (Task 2b, 2c)."""
import torch
import torch.nn as nn
from torchvision import models

CHANNELS = (16, 32, 64, 64)


class CustomCNN(nn.Module):
    """n_blocks x [Conv3x3(pad1) -> ReLU -> Pool2] -> Flatten -> Dropout -> FC -> ReLU -> Dropout -> FC.
    pool in {'max','avg','none'}; 'none' replaces the pool by a stride-2 convolution (same spatial size)."""

    def __init__(self, n_blocks=2, pool="max", hidden=32, dropout=0.3, n_classes=10, in_size=28):
        super().__init__()
        self.cfg = dict(n_blocks=n_blocks, pool=pool, hidden=hidden, dropout=dropout)
        layers, cin = [], 1
        for i in range(n_blocks):
            cout = CHANNELS[i]
            if pool == "none":
                layers += [nn.Conv2d(cin, cout, 3, stride=2, padding=1), nn.ReLU()]
            else:
                layers += [nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(),
                           nn.MaxPool2d(2) if pool == "max" else nn.AvgPool2d(2)]
            cin = cout
        self.features = nn.Sequential(*layers)
        with torch.no_grad():
            flat = self.features(torch.zeros(1, 1, in_size, in_size)).numel()
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(flat, hidden),
                                        nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


class MLP(nn.Module):
    """Fully-connected baseline with a parameter count comparable to the default CNN (~53k vs ~55k)."""

    def __init__(self, hidden=(64, 32), n_classes=10):
        super().__init__()
        layers, d = [nn.Flatten()], 28 * 28
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---- 2(b) bookkeeping ---------------------------------------------------------------
def hand_param_count(cfg, in_size=28, n_classes=10):
    """Independent closed-form count: conv = (k*k*Cin + 1)*Cout ; fc = (in + 1)*out."""
    s, cin, total = in_size, 1, 0
    for i in range(cfg["n_blocks"]):
        cout = CHANNELS[i]
        total += (3 * 3 * cin + 1) * cout
        s = (s + 2 - 3) // 2 + 1 if cfg["pool"] == "none" else s // 2
        cin = cout
    flat = cin * s * s
    return total + (flat + 1) * cfg["hidden"] + (cfg["hidden"] + 1) * n_classes


def layer_table(model, in_size=28):
    """Per-layer table (input shape, kernel, stride, padding, filters, output shape, params) via hooks."""
    rows, hooks = [], []
    first = lambda v: v[0] if isinstance(v, (tuple, list)) else v

    def hook(m, inp, out):
        r = dict(layer=m.__class__.__name__, input_shape=tuple(inp[0].shape[1:]), kernel="-", stride="-",
                 padding="-", filters="-", output_shape=tuple(out.shape[1:]),
                 params=sum(p.numel() for p in m.parameters()))
        if isinstance(m, nn.Conv2d):
            r.update(kernel=first(m.kernel_size), stride=first(m.stride), padding=first(m.padding), filters=m.out_channels)
        elif isinstance(m, (nn.MaxPool2d, nn.AvgPool2d)):
            r.update(kernel=first(m.kernel_size), stride=first(m.stride), padding=first(m.padding))
        rows.append(r)

    for m in model.modules():
        if len(list(m.children())) == 0 and not isinstance(m, nn.ReLU):
            hooks.append(m.register_forward_hook(hook))
    was = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 1, in_size, in_size, device=next(model.parameters()).device))
    model.train(was)
    for h in hooks:
        h.remove()
    return rows


def receptive_field(cfg):
    """RF (input pixels) of one unit in the LAST conv layer, via r += (k-1)*jump, jump *= stride."""
    r, j, last, steps = 1, 1, 1, []
    for i in range(cfg["n_blocks"]):
        if cfg["pool"] == "none":
            r += (3 - 1) * j; j *= 2; last = r
            steps.append(f"conv{i+1} (k3,s2): r={r}, jump={j}")
        else:
            r += (3 - 1) * j; last = r
            steps.append(f"conv{i+1} (k3,s1): r={r}, jump={j}")
            r += (2 - 1) * j; j *= 2
            steps.append(f"pool{i+1} (k2,s2): r={r}, jump={j}")
    return last, steps


# ---- Task 4: ResNet-18 ----------------------------------------------------------------
def make_resnet18(strategy="feature_extraction", pretrained=True):
    """strategy: 'feature_extraction' (fc only) | 'gradual' (layer4 + fc) | 'differential' (all)."""
    m = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, 10)
    for p in m.parameters():
        p.requires_grad = False
    train_mods = {"feature_extraction": [m.fc], "gradual": [m.layer4, m.fc], "differential": [m]}[strategy]
    for mod in train_mods:
        for p in mod.parameters():
            p.requires_grad = True
    return m
