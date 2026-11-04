from __future__ import annotations
import argparse
import json
from autokolpak_ai.validate import validate_dataset

parser = argparse.ArgumentParser()
parser.add_argument("data_dir")
parser.add_argument("--report")
args = parser.parse_args()
print(json.dumps(validate_dataset(args.data_dir, args.report), ensure_ascii=False, indent=2))