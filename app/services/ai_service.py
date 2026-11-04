from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from io import BytesIO
from typing import Any

from PIL import Image

from autokolpak_ai.constants import CENTER_TYPES, FAMILIES, FINISHES, REGRESSION_FIELDS, REGRESSION_RANGES


class ModelNotConfigured(RuntimeError):
    pass


class AIService:
    def __init__(self) -> None:
        self.base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com").strip().rstrip("/") or "https://api.anthropic.com"
        self.token = os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("ANTHROPIC_API_KEY")
        self.model = os.getenv("AUTOKOLPAK_MODEL", "claude-sonnet-5")
        try:
            self.timeout = max(1.0, min(300.0, float(os.getenv("AUTOKOLPAK_API_TIMEOUT", "90"))))
        except (TypeError, ValueError):
            self.timeout = 90.0

    @property
    def ready(self) -> bool:
        return bool(self.token)

    def _prompt(self) -> str:
        return ("Проанализируй фото автомобильного колпака. Верни ровно один JSON object. "
                "Считай visible outer spoke ends отдельно от spoke roots; выполни два независимых подсчёта. "
                f"Допустимые family: {FAMILIES}; center: {CENTER_TYPES}; finishes: {FINISHES}. "
                f"Допустимые ratios и диапазоны: {REGRESSION_RANGES}. Миллиметры без reference diameter недостоверны. "
                "Разрешены поля family, spoke_count, spoke_roots_count, center_type, finish, geometry_ratios, top3 и error.")

    @staticmethod
    def _image_payload(image: Image.Image) -> str:
        image = image.convert("RGB")
        image.thumbnail((1568, 1568), Image.Resampling.LANCZOS)
        output = BytesIO()
        image.save(output, format="JPEG", quality=92)
        return base64.b64encode(output.getvalue()).decode("ascii")

    @staticmethod
    def _parse(text: str) -> dict[str, Any]:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)
        value = json.loads(cleaned)
        if not isinstance(value, dict):
            raise ValueError("Ответ модели не является JSON object")
        if value.get("error"):
            raise ValueError(str(value["error"]))
        if value.get("family") not in FAMILIES:
            raise ValueError("В ответе нет допустимого family")
        if value.get("center_type") not in CENTER_TYPES or value.get("finish") not in FINISHES:
            raise ValueError("Ответ содержит недопустимые labels")
        for name in REGRESSION_FIELDS:
            if name in value.get("geometry_ratios", {}):
                low, high = REGRESSION_RANGES[name]
                value["geometry_ratios"][name] = max(low, min(high, float(value["geometry_ratios"][name])))
        return value

    def analyze(self, image: Image.Image) -> dict[str, Any]:
        if not self.ready:
            raise ModelNotConfigured("ANTHROPIC_AUTH_TOKEN или ANTHROPIC_API_KEY не настроен")
        payload = {"model": self.model, "max_tokens": 1200, "messages": [{"role": "user", "content": [{"type": "text", "text": self._prompt()}, {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": self._image_payload(image)}}]}]}
        request = urllib.request.Request(self.base_url + "/v1/messages", data=json.dumps(payload).encode(), headers={"content-type": "application/json", "x-api-key": self.token, "anthropic-version": "2023-06-01"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise RuntimeError("Anthropic API отклонил авторизацию") from exc
            raise RuntimeError(f"Anthropic API вернул HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError("Не удалось связаться с Anthropic API") from exc
        content = body.get("content", [])
        text = "".join(item.get("text", "") for item in content if isinstance(item, dict))
        return self._parse(text)