import torch
import torch.nn.functional as F
from .constants import REGRESSION_FIELDS

def multitask_loss(outputs, targets, label_smoothing=0.0, loss_weights=None):
    weights={"family":1.,"spokes":1.,"center":1.,"finish":1.,"regression":1.}; weights.update(loss_weights or {})
    losses={}
    for key in ("family","spokes","center","finish"): losses[key]=F.cross_entropy(outputs[key],targets[key].long(),label_smoothing=label_smoothing)
    pred=outputs["regression"]; truth=targets["regression"].to(pred.dtype); field_losses=[]
    for idx,_ in enumerate(REGRESSION_FIELDS):
        mask=torch.isfinite(truth[:,idx]); field_losses.append(F.smooth_l1_loss(pred[mask,idx],truth[mask,idx]) if mask.any() else pred[:,idx].sum()*0)
    losses["regression"]=torch.stack(field_losses).mean(); total=sum(losses[k]*weights[k] for k in losses)
    return total, {**losses,"total":total}