from __future__ import annotations
import argparse
from autokolpak_ai.validate import validate_dataset

parser = argparse.ArgumentParser()
parser.add_argument("data_dir")
parser.add_argument("--report")
args = parser.parse_args()
print(validate_dataset(args.data_dir, args.report))