"""Executes the code cells of a checkpoint notebook locally in test mode (CPU, tiny grids,
no Drive, no clone, no pip): python colab/run_local_test.py colab/01_postfix_rerun.ipynb OUT_DIR"""
import json
import os
import sys

nb_path, out_dir = sys.argv[1], sys.argv[2]
os.environ.update(PASADO_NB_LOCAL_TEST="1", PASADO_NB_OUT=os.path.abspath(out_dir),
                  PASADO_NB_REPO=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ns = {"__name__": "__main__"}
for i, cell in enumerate(json.load(open(nb_path))["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    print(f"\n===== cell {i}: {src.splitlines()[0]}", flush=True)
    exec(compile(src, f"<cell {i}>", "exec"), ns)
print("\nLOCAL TEST: all cells executed")
