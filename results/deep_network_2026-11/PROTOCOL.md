# A network with twice the depth of `big`: how much does the bound grow? (protocol)

Written 2026-11-06, before training or measuring. Notebook: `colab/03_deep_network.ipynb`.
Scripts: `experiments/train_deep_network.py`, `experiments/measure_deep_network.py`.

## Network and training

`deep8`: 784 - 8 x (1024, sigmoid) - 10, i.e. 8 sigmoid layers against 4 of `big`, same width.
Training copies `Section_5_4/train_mnist.py` exactly (Adam, lr 1e-4, StepLR 25 / 0.75, 100
epochs, batch 256, RandomCrop(28, padding=4), cross entropy, no input normalisation), seed 0,
on the Colab GPU in float32. Weights and a checkpoint per epoch go to Drive (resumable); the
repository's `Section_5_4/trained/` and `Section_5_4/results/` are not touched. Reported: test
accuracy, and the list of correctly classified test images (as `indices_<net>.pth`).
A network that does not reach 90% test accuracy is reported as such and not used for bounds
(sigmoid networks of this depth can train poorly); no retuning without a new protocol.

## Measurement

- Images: the first 30 test images classified correctly by both `big` and `deep8`.
- eps indices 0, 4, 8, 12, 15; n_splits 1 and 8; float64; real cubic; seed 7; batched path with
  the automatic plane-fit choice; on the Colab GPU.
- Per (network, image, eps, n): the Pasado bound, the sampled autograd maximum of |dF/dt|
  (4001 points), their ratio (looseness), and the time per pass (median of 3).
- Reported per eps: mean bound and mean looseness of `big` and `deep8`, the ratio
  deep8 / big of both, and the time ratio. Soundness: every bound >= its sampled maximum.

Descriptive; no decision rule. It extends the depth axis of the LiRPA comparison (2, 3, 4
sigmoid layers so far) with an 8-layer point for Pasado alone.
