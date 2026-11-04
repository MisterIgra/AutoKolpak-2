from __future__ import annotations
import importlib.util
import sys

for package in ("PIL", "yaml", "torch", "torchvision", "fastapi", "trimesh"):
    print(f"{package}: {'ok' if importlib.util.find_spec(package) else 'missing'}")
print(f"python: {sys.version.split()[0]}")