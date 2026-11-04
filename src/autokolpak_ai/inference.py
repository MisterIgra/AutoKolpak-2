from pathlib import Path
import torch
from PIL import Image
from .constants import *
from .model import HubcapNet
from .transforms import inference_transform

ARCHITECTURE="resnet18_multitask_v2"
def load_checkpoint(path, device="cpu"):
    p=Path(path)
    if not p.is_file(): raise FileNotFoundError(f"Checkpoint не найден: {p}")
    ck=torch.load(p,map_location=device,weights_only=True)
    if not isinstance(ck, dict) or "model_state" not in ck: raise ValueError("Некорректный checkpoint; требуется переобучение")
    checks={"schema_version":SCHEMA_VERSION,"architecture":ARCHITECTURE,"families":FAMILIES,"spoke_counts":SPOKE_COUNTS,"center_types":CENTER_TYPES,"finishes":FINISHES,"regression_fields":REGRESSION_FIELDS}
    for key,expected in checks.items():
        if ck.get(key)!=expected: raise ValueError(f"Несовместимый checkpoint ({key}); требуется переобучение")
    model=HubcapNet(pretrained=False); model.load_state_dict(ck["model_state"],strict=True); model.to(device).eval(); return model,ck

def predict_pil(model,image,reference_outer_diameter_mm=None,device="cpu"):
    if reference_outer_diameter_mm is not None and not 330<=float(reference_outer_diameter_mm)<=460: raise ValueError("reference_outer_diameter_mm должен быть в диапазоне 330–460 мм")
    x=inference_transform()(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad(): out=model(x)
    probabilities={k:torch.softmax(out[k],dim=1)[0] for k in ("family","spokes","center","finish")}
    family=FAMILIES[int(probabilities["family"].argmax())]; count=int(probabilities["spokes"].argmax())+5; center=CENTER_TYPES[int(probabilities["center"].argmax())]; finish=FINISHES[int(probabilities["finish"].argmax())]
    ratios=denormalize_ratios(dict(zip(REGRESSION_FIELDS,out["regression"][0].cpu().tolist())))
    if center=="cap": ratios["center_hole_ratio"]=None
    params={"family":family,"spoke_count":count,"center_type":center,"finish":finish,"center_hole_ratio":ratios["center_hole_ratio"],"bulge_depth_ratio":ratios["bulge_depth_ratio"],"face_thickness_ratio":ratios["face_thickness_ratio"],"rim_depth_ratio":ratios["rim_depth_ratio"]}
    if reference_outer_diameter_mm is not None:
        for field,ratio in ratios.items(): params[field.replace("_ratio","_mm")]=None if ratio is None else round(ratio*reference_outer_diameter_mm,2)
    else:
        for field in REGRESSION_SOURCE_FIELDS: params[field]=None
    confidence={k:float(v.max()) for k,v in probabilities.items()}
    top3={k:[{"label": (FAMILIES if k=="family" else SPOKE_COUNTS if k=="spokes" else CENTER_TYPES if k=="center" else FINISHES)[int(i)], "score":float(probabilities[k][i])} for i in torch.topk(probabilities[k], min(3, probabilities[k].numel())).indices] for k in probabilities}
    return {"model_version":"2.0.0","parameters":params,"geometry_ratios":ratios,"confidence":confidence,"top3":top3,"reference_outer_diameter_mm":reference_outer_diameter_mm,"units":"mm only with measured reference diameter","confidence_interpretation":"Ranking/self-reported score, не гарантированная вероятность.","warning":"Синтетическое обучение не подтверждает качество на реальных фотографиях."+(" Центральное отверстие скрыто крышкой; диаметр отверстия не определяется." if center=="cap" else "")}

def predict_image(model,image_path,reference_outer_diameter_mm=None,device="cpu"):
    with Image.open(image_path) as im: return predict_pil(model,im,reference_outer_diameter_mm,device)