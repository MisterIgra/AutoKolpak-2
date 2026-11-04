from __future__ import annotations
import math
from .constants import RAW_FIELD_RANGES, FAMILIES, CENTER_TYPES

def sanitize_parameters(params):
    result=dict(params)
    if "outer_diameter_mm" not in result or result["outer_diameter_mm"] is None: raise ValueError("Для физического экспорта нужен измеренный наружный диаметр")
    for key,(lo,hi) in RAW_FIELD_RANGES.items():
        if key in result and result[key] is not None:
            value = float(result[key])
            if not math.isfinite(value): raise ValueError(f"{key} должен быть конечным числом")
            result[key]=max(lo,min(hi,value))
    if result.get("family","classic") not in FAMILIES: raise ValueError("Неизвестное семейство")
    if result.get("center_type","open") not in CENTER_TYPES: raise ValueError("Неизвестный тип центра")
    result["spoke_count"]=max(5,min(24,int(result.get("spoke_count",5))))
    result.setdefault("center_hole_diameter_mm", 60.0)
    result.setdefault("bulge_depth_mm", 16.0)
    result.setdefault("face_thickness_mm", 7.0)
    result.setdefault("rim_depth_mm", 120.0)
    return result

def create_parametric_mesh(params, sections=96):
    import trimesh
    p=sanitize_parameters(params); radius=p["outer_diameter_mm"]/2; depth=p.get("rim_depth_mm",100); thickness=p.get("face_thickness_mm",7)
    hole = 0.0 if p["center_type"] == "cap" else p["center_hole_diameter_mm"] / 2
    parts=[trimesh.creation.annulus(r_min=max(1,hole),r_max=radius, height=max(1,thickness), sections=sections),trimesh.creation.cylinder(radius=max(2, radius*.22), height=max(1,depth), sections=sections)]
    n=p["spoke_count"]
    for i in range(n):
        angle=2*math.pi*i/n; length=radius*.72; width=radius*.07
        box=trimesh.creation.box(extents=(length,width,max(1,thickness))); box.apply_translation((radius*.52,0,depth*.45)); box.apply_transform(trimesh.transformations.rotation_matrix(angle,(0,0,1))); parts.append(box)
    return trimesh.util.concatenate(parts)