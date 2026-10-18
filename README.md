# ECM-402 Lab 4 - CNNs for Multi-Dimensional Signals (Roll No. 523194)

GitHub: https://github.com/<your-username>/ecm402-lab4-523194   <!-- fill in before submission -->

Fashion-MNIST: custom CNN, ablations, ResNet-18 transfer learning (64x64), Grad-CAM, optional denoising bonus.

## Setup and run
    pip install -r requirements.txt
    python src/main.py            # reproduces every figure/table in results/
    python src/main.py --bonus    # also runs the Task 8 denoising bonus
    python src/main.py --quick    # tiny smoke test of the pipeline (numbers NOT for the report)

The dataset downloads automatically into `data/` (git-ignored). No checkpoints are committed.

## Random seeds
`SEED = 42` (src/data.py) controls: the 80/20 train/validation split (`random_split` with a seeded
generator), the stratified 10,000 / 2,000 ResNet subsets (seeded NumPy RNG), data-loader shuffling,
and weight initialisation (`set_seed` before every model is built, so all variants share initial seeds).

## Hardware and runtimes
Fill in after running `python src/main.py` (the script writes `results/tables/hardware_and_runtimes.md`):

| item | value |
|---|---|
| Hardware (CPU-only / GPU model) | ... |
| MLP baseline, sec/epoch | ... |
| Custom CNN, sec/epoch | ... |
| ResNet-18 best strategy (64x64, 10k subset), sec/epoch | ... |

## Layout
    src/data.py       transforms, splits, loaders, filtering demo (Task 1)
    src/models.py     custom CNN, MLP baseline, ResNet-18 setup, layer table / RF utilities (Tasks 2, 4)
    src/train.py      training/validation loops, optimizers, table helpers
    src/ablations.py  Task 3 controlled experiments
    src/gradcam.py    Grad-CAM (hooks on model.layer4[-1], the post-ReLU block output)
    src/denoise.py    Task 8 bonus
    src/main.py       reproduces all results
    results/          figures/ and tables/ (generated)

## Protocol
All selection (architecture, augmentation, fine-tuning strategy, best epoch) uses the validation split only.
The official test set is loaded only in Task 5/6, after all models are frozen.
