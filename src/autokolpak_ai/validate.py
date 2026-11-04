from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from .constants import (
    CENTER_TYPES,
    DESIGN_VARIANTS,
    FAMILIES,
    FINISHES,
    GENERATOR_VERSION,
    RAW_FIELD_RANGES,
    SCHEMA_VERSION,
    SPOKE_COUNTS,
)


def _message(items: list[str], value: str) -> None:
    items.append(value)


def validate_dataset(data_dir: str | Path, report_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(data_dir).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    rows: list[tuple[int, dict[str, Any]]] = []
    annotation = root / "metadata" / "annotations.jsonl"
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generator_version_expected": GENERATOR_VERSION,
        "data_dir": str(root),
        "annotation_rows": 0,
        "valid_unique_images": 0,
        "unique_models": 0,
        "family_models": {},
        "spoke_count_models": {},
        "finish_models": {},
        "duplicate_geometry_signatures": [],
        "errors_count": 0,
        "errors_preview": [],
        "warnings_count": 0,
        "warnings_preview": [],
        "ok": False,
        "notes": ["Проверка синтетического датасета не заменяет проверку качества на реальных фотографиях."],
    }
    if not annotation.is_file():
        _message(errors, f"Отсутствует {annotation}")
    else:
        try:
            lines = annotation.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            _message(errors, f"Невозможно прочитать annotations.jsonl: {exc}")
            lines = []
        for number, line in enumerate(lines, 1):
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("строка должна содержать JSON object")
                rows.append((number, value))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                _message(errors, f"Строка {number}: некорректный JSON ({exc})")

    seen_paths: set[str] = set()
    valid_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    signatures: Counter[str] = Counter()
    for number, row in rows:
        try:
            required = ("model_id", "image_path", "family", "design_variant", "center_type", "finish", "spoke_count", "spoke_roots_count", *RAW_FIELD_RANGES)
            missing = [field for field in required if field not in row]
            if missing:
                raise ValueError("отсутствуют поля: " + ", ".join(missing))
            image_path = str(row["image_path"])
            resolved = (root / image_path).resolve()
            if resolved == root or root not in resolved.parents:
                raise ValueError("image_path выходит за пределы dataset root")
            if image_path in seen_paths:
                raise ValueError("повтор image_path")
            seen_paths.add(image_path)
            if not resolved.is_file():
                raise ValueError("изображение отсутствует")
            with Image.open(resolved) as image:
                image.verify()
            if row.get("generator_version") != GENERATOR_VERSION:
                raise ValueError(f"generator_version должен быть {GENERATOR_VERSION}")
            if row["family"] not in FAMILIES or row["design_variant"] not in DESIGN_VARIANTS or row["center_type"] not in CENTER_TYPES or row["finish"] not in FINISHES:
                raise ValueError("неизвестная категориальная метка")
            count = int(row["spoke_count"])
            roots = int(row["spoke_roots_count"])
            if count not in SPOKE_COUNTS:
                raise ValueError("spoke_count вне диапазона 5–24")
            expected_roots = count if row["family"] in ("classic", "mesh") else count // 2
            if roots != expected_roots or (row["family"] not in ("classic", "mesh") and count % 2):
                raise ValueError("несогласованные spoke_count/spoke_roots_count")
            for field, (low, high) in RAW_FIELD_RANGES.items():
                value = float(row[field])
                if not math.isfinite(value) or not low <= value <= high:
                    raise ValueError(f"{field} вне диапазона")
            model_id = str(row["model_id"])
            valid_by_model[model_id].append(row)
            signature = str(row.get("geometry_signature", ""))
            if signature:
                signatures[signature] += 1
        except (KeyError, TypeError, ValueError, OSError) as exc:
            _message(errors, f"Строка {number}: {exc}")

    for model_id, model_rows in valid_by_model.items():
        reference = {key: value for key, value in model_rows[0].items() if key not in {"image_id", "image_path", "camera_azimuth_deg", "camera_elevation_deg", "resolution", "seed"}}
        for row in model_rows[1:]:
            current = {key: value for key, value in row.items() if key not in {"image_id", "image_path", "camera_azimuth_deg", "camera_elevation_deg", "resolution", "seed"}}
            if current != reference:
                _message(errors, f"model_id={model_id}: параметры между views не согласованы")
                break

    duplicate_signatures = sorted(signature for signature, count in signatures.items() if count > 1)
    if duplicate_signatures:
        _message(errors, "обнаружены дублирующиеся geometry_signature")
    if len(valid_by_model) < 3:
        _message(errors, "требуется минимум 3 уникальные модели")
    for family in FAMILIES:
        if not any(rows_for_model[0].get("family") == family for rows_for_model in valid_by_model.values()):
            _message(warnings, f"слабое покрытие family={family}")

    report.update(
        annotation_rows=len(rows),
        valid_unique_images=len(seen_paths),
        unique_models=len(valid_by_model),
        family_models=dict(Counter(model_rows[0]["family"] for model_rows in valid_by_model.values())),
        spoke_count_models=dict(Counter(str(model_rows[0]["spoke_count"]) for model_rows in valid_by_model.values())),
        finish_models=dict(Counter(model_rows[0]["finish"] for model_rows in valid_by_model.values())),
        duplicate_geometry_signatures=duplicate_signatures,
        errors_count=len(errors),
        errors_preview=errors[:20],
        warnings_count=len(warnings),
        warnings_preview=warnings[:20],
        ok=not errors,
    )
    if report_path is not None:
        output = Path(report_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report