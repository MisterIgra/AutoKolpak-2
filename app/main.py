from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from app.api.routes import router

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="AutoKolpak AI", version="2.0.0")
app.include_router(router)
vendor = ROOT / "frontend" / "vendor"
if vendor.is_dir():
    app.mount("/vendor", StaticFiles(directory=vendor), name="vendor")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(ROOT / "frontend" / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/whoami")
def whoami() -> dict[str, str]:
    return {"project": "AutoKolpak AI", "version": "2.0.0", "generator_version": "4.0.0"}