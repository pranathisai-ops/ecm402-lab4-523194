"""Task 6: Grad-CAM from scratch with forward hook + tensor gradient hook.
Target layer used for ResNet-18: model.layer4[-1] (block OUTPUT, i.e. after the final ReLU)."""
import numpy as np
import torch
import torch.nn.functional as F
from data import IN_MEAN, IN_STD


class GradCAM:
    def __init__(self, model, target_layer):
        self.model, self.acts, self.grads = model, None, None
        self._h = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.acts = output.detach()
        output.register_hook(lambda g: setattr(self, "grads", g.detach()))

    def __call__(self, x, class_idx=None):
        """x: (B,3,H,W). Returns cams (B,H,W) in [0,1], explained class idx, predicted class idx."""
        self.model.eval()
        x = x.clone().requires_grad_(True)               # guarantees a graph even with a frozen backbone
        with torch.enable_grad():
            logits = self.model(x)
            pred = logits.argmax(1)
            idx = pred if class_idx is None else torch.full_like(pred, class_idx)
            self.model.zero_grad()
            logits.gather(1, idx[:, None]).sum().backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)                 # alpha_k = GAP of dy/dA_k
        cam = F.relu((w * self.acts).sum(dim=1, keepdim=True))        # ReLU(sum_k alpha_k A_k)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        mn, mx = cam.amin(dim=(2, 3), keepdim=True), cam.amax(dim=(2, 3), keepdim=True)
        cam = (cam - mn) / (mx - mn + 1e-8)
        return cam.squeeze(1).cpu().numpy(), idx.cpu().numpy(), pred.cpu().numpy()

    def remove(self):
        self._h.remove()


def denorm_gray(x):
    """(3,H,W) ImageNet-normalised tensor -> (H,W) image in [0,1] (all channels identical)."""
    m = torch.tensor(IN_MEAN).view(3, 1, 1); s = torch.tensor(IN_STD).view(3, 1, 1)
    return (x.cpu() * s + m)[0].clamp(0, 1).numpy()


def overlay(ax, img, cam, title):
    ax.imshow(img, cmap="gray"); ax.imshow(cam, cmap="jet", alpha=0.45); ax.set_title(title, fontsize=8); ax.axis("off")


def cam_on_object(img, cam, thr=0.15):
    """Fraction of CAM mass falling on garment pixels (img > thr): a crude but quantitative check."""
    mask = img > thr
    return float(cam[mask].sum() / (cam.sum() + 1e-8))
