from __future__ import annotations
import argparse
from pathlib import Path
import yaml
from autokolpak_ai.validate import validate_dataset

parser = argparse.ArgumentParser(description="Проверка датасета перед обучением")
parser.add_argument("--config", default="configs/train.yaml")
args = parser.parse_args()
config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
report = validate_dataset(config["data_dir"], Path(config["output_dir"]) / "dataset_report.json")
if not report["ok"]:
    raise SystemExit("Датасет не прошёл валидацию; обучение остановлено")
print({"status": "validated", "message": "Датасет готов к обучению", "models": report["unique_models"]})