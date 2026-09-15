# Known Issues (from a full local re-run, 2026-07-06/07)

Ran on Windows via the project's `.venv` (`Python 3.13.3`). Full log: `run_all_2.log`
(note: this log contains some duplicated step output because a background
process wasn't cleanly killed before it was resumed — harmless, but ignore
repeated blocks; the final state per step is what matters).

## Installed vs. required versions (README table)

| Library | Required | Installed (.venv) | Notes |
|---|---|---|---|
| Python | 3.10.6 (3.9+) | 3.13.3 | much newer than tested |
| Numpy | 1.25.1 (1.19.5+) | 2.5.1 | **breaking** — see bugs below |
| Affapy | 0.1 | 0.1 | matches |
| Scikit-learn | 1.3.0 (0.24.2+) | 1.9.0 | pickle format skew, see below |
| Seaborn | 0.12.2 (0.11.2+) | 0.13.2 | no issues observed |
| Matplotlib | 3.7.2 (3.7.1+) | 3.11.0 | no issues observed |
| tabulate | 0.9.0 | 0.10.0 | no issues observed |
| tqdm | 4.65.0 | 4.68.3 | no issues observed |
| PyTorch | 2.0.1+cpu (1.9.0+) | 2.12.1+cpu | no issues observed |
| Torchvision | 0.15.2+cpu (0.14.1+) | 0.27.1+cpu | no issues observed |
| Torchaudio | 2.0.2+cpu (0.13.1+) | 2.11.0+cpu | no issues observed |
| IPython | 8.14.0 (5.8.0+) | 9.15.0 | not exercised (no notebook run) |
| Jupyter Notebook | 6.5.4 (5.7.11+) | 7.6.0 | not exercised (no notebook run) |

## Results

Pass (exit 0): all 7 source-level test scripts (`forward_mode_non_tensorized_src/tests/*.py`,
`reverse_mode_non_tensorized_src/tests/*.py`), Section 5.2 (`climate.sh`, `chemical.sh`),
Section 5.3 (`black_scholes.sh`), Section 5.4 FFNN Lipschitz (`lipschitz.sh`), Section 5.5
`adult_script.py`.

Fail (exit 1):

1. **Section 5.4 CNN Lipschitz** — `lipschitz_cnn.sh` → `get_lipschitz_cnn.py` →
   `forward_mode_tensorized_src/conv.py:38`:
   ```
   AttributeError: module 'numpy' has no attribute 'product'
   ```
   `np.product` was deprecated in NumPy 1.25 and removed in NumPy 2.0. Fix: replace with
   `np.prod` in `forward_mode_tensorized_src/conv.py`.

2. **Section 5.5 plotting** — `adult_plot.py:21`:
   ```
   ValueError: invalid literal for int() with base 10: ' np.int64(25'
   ```
   NumPy ≥2.0 changed scalar `repr()`/`str()` (NEP 51): `str(np.int64(25))` now prints
   `np.int64(25)` instead of `25`. `adult_plot.py` naively parses a stringified tuple
   assuming the old format. Fix: either cast to plain `int` before stringifying upstream,
   or parse with a regex (e.g. `re.search(r'-?\d+', t.split(",")[2])`).

Both failures reproduce the exact same root cause seen in the previous run (same day,
earlier log) — confirmed deterministic, not flaky.

## Non-fatal but worth knowing

- **sklearn pickle version skew**: every step that loads a pretrained/pickled sklearn
  model (`MLPRegressor`, `MLPClassifier`, `LabelBinarizer`) prints
  `InconsistentVersionWarning` (models pickled with 1.3.0, loaded with 1.9.0). Didn't
  crash anything in this run, but sklearn's own docs warn this "might lead to breaking
  code or invalid results" — worth re-pickling the trained models with the current
  sklearn if exact numeric reproducibility matters.
- **`Section_5_2/clear.sh` exits 1** — harmless (just clears output dirs before a rerun),
  but technically a nonzero exit; doesn't block anything downstream.
- **Two venvs in the repo**: `.venv` (fully populated, used for this run) and `venv`
  (only has `pip` installed, effectively empty/abandoned). Worth deleting the unused one
  to avoid confusion about which to activate.
- **Not run**: `Section_5_4/train_mnist.py`, `train_cnn.py`, `test_mnist.py` and the two
  Jupyter notebooks (`Plot.ipynb`, `Plot_cnn.ipynb`) — README marks retraining as
  optional (pretrained models are already provided), and the notebooks need interactive
  Jupyter rather than a batch run.
