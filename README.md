# ecm402-lab4-523194
# ECM402 Lab 4 — CNNs for Fashion-MNIST

**Roll number:** 523194
**GitHub:** https://github.com/<you>/ecm402-lab4-523194
## Hardware
- CPU: <your CPU>, 2 threads used
- No GPU assumed
- Per-epoch times:
  - Custom CNN (28×28, batch 64): ~42 s
  - ResNet-18 (64×64, batch 32, 10k subset): ~120–140 s

## Seeds
- Global seed: 42 (torch, numpy, random)
- Train/val split: `random_split` with `torch.Generator().manual_seed(42)`
- ResNet subset: stratified, same seed

## Reproduce
```bash
pip install -r requirements.txt
python -m src.main
