import subprocess
import yaml
from pathlib import Path

EXPERIMENTS_FILE = Path("conf/experiments/experiments.yaml")
TRAIN_SCRIPT = "-m scripts.train"


def run_experiments():
    if not EXPERIMENTS_FILE.exists():
        raise FileNotFoundError(f"{EXPERIMENTS_FILE} not found!")

    with open(EXPERIMENTS_FILE, "r") as f:
        experiments = yaml.safe_load(f)

    print(f"[INFO] Found {len(experiments)} experiments to run.\n")

    for idx, exp in enumerate(experiments, 1):
        print(f"[RUN {idx}] {exp.get('name', 'Unnamed experiment')}")
        overrides = []

        # Construir overrides de Hydra
        for k, v in exp.items():
            if k in ["name"]:
                continue
            if isinstance(v, bool):
                v = str(v).lower()
            overrides.append(f"{k}={v}")

        cmd = ["python", TRAIN_SCRIPT] + overrides
        print(f"[CMD] {' '.join(cmd)}\n")
        subprocess.run(" ".join(cmd), shell=True, check=True)
        print("\n" + "-" * 60 + "\n")

    print("[INFO] All experiments finished.")


if __name__ == "__main__":
    run_experiments()
