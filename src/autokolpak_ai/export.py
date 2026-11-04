import json
from pathlib import Path
from .geometry import create_parametric_mesh

EXPORT_NOTE="Концептуальная параметрическая геометрия; не является производственной CAD-моделью и требует проверки специалистом."
def export_model(params, output_dir, formats=("stl","glb")):
    mesh=create_parametric_mesh(params); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); exported=[]
    for fmt in formats:
        fmt=fmt.lower()
        if fmt not in ("stl","glb"): raise ValueError(f"Неподдерживаемый формат: {fmt}")
        path=out/f"concept.{fmt}"; mesh.export(path); exported.append(str(path))
    (out/"metadata.json").write_text(json.dumps({"parameters":params,"units":"mm","note":EXPORT_NOTE},ensure_ascii=False,indent=2),encoding="utf-8")
    return exported