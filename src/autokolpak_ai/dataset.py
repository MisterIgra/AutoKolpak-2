from __future__ import annotations
import json, random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from .constants import *
from .transforms import inference_transform

def load_annotations(data_dir):
    path = Path(data_dir).resolve() / "metadata" / "annotations.jsonl"
    if not path.is_file(): raise FileNotFoundError(f"Не найден файл аннотаций: {path}")
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try: rows.append(json.loads(line))
        except json.JSONDecodeError as exc: raise ValueError(f"Некорректный JSON в строке {number}: {exc}") from exc
    return rows

def split_by_model(rows, validation_fraction=.15, test_fraction=.15, seed=42):
    groups = {}
    for row in rows: groups.setdefault(row["model_id"], []).append(row)
    if len(groups) < 3: raise ValueError("Для split необходимо минимум 3 уникальные модели")
    ids = list(groups); random.Random(seed).shuffle(ids)
    if not 0 <= validation_fraction < 1 or not 0 <= test_fraction < 1 or validation_fraction + test_fraction >= 1:
        raise ValueError("Доли validation/test должны быть неотрицательными и оставлять train")
    test_n = max(1, round(len(ids) * test_fraction)); val_n = max(1, round(len(ids) * validation_fraction))
    if val_n + test_n >= len(ids): raise ValueError("Слишком маленький датасет для train/validation/test")
    test = ids[:test_n]; val = ids[test_n:test_n + val_n]; train = ids[test_n + val_n:]
    return {"train": [r for i in train for r in groups[i]], "validation": [r for i in val for r in groups[i]], "test": [r for i in test for r in groups[i]]}, {"train": train, "validation": val, "test": test}

class HubcapDataset(Dataset):
    def __init__(self, rows, data_dir, transform=None):
        self.rows, self.root, self.transform = rows, Path(data_dir).resolve(), transform or inference_transform()
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]; path = (self.root / row["image_path"]).resolve()
        if self.root not in path.parents: raise ValueError("Путь изображения выходит за пределы датасета")
        if not path.is_file(): raise FileNotFoundError(f"Изображение не найдено: {path}")
        with Image.open(path) as source:
            image = source.convert("RGB")
        tensor = self.transform(image)
        ratios = ratios_from_row(row); ratios["center_hole_ratio"] = float("nan") if row.get("center_type") == "cap" else ratios["center_hole_ratio"]
        return {"image": tensor, "family": FAMILIES.index(row["family"]), "spokes": int(row["spoke_count"]) - 5, "center": CENTER_TYPES.index(row["center_type"]), "finish": FINISHES.index(row["finish"]), "regression": torch.tensor([ratios[x] for x in REGRESSION_FIELDS], dtype=torch.float32), "regression_ratios": ratios, "outer_diameter_mm": float(row["outer_diameter_mm"]), "model_id": row["model_id"]}