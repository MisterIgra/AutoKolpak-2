from __future__ import annotations
import argparse
import json
from autokolpak_ai.inference import load_checkpoint, predict_image

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("image")
parser.add_argument("--diameter", type=float)
args = parser.parse_args()
model, _ = load_checkpoint(args.checkpoint)
print(json.dumps(predict_image(model, args.image, args.diameter), ensure_ascii=False, indent=2))