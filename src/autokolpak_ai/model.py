import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

class HubcapNet(nn.Module):
    architecture = "resnet18_multitask_v2"
    def __init__(self, pretrained=True, dropout=.25):
        super().__init__()
        try: backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        except Exception as exc: raise RuntimeError("Не удалось загрузить pretrained ResNet-18; повторите с pretrained=False") from exc
        features=backbone.fc.in_features; backbone.fc=nn.Identity(); self.backbone=backbone; self.dropout=nn.Dropout(dropout)
        self.family_head=nn.Linear(features,5); self.spokes_head=nn.Linear(features,20); self.center_head=nn.Linear(features,2); self.finish_head=nn.Linear(features,3); self.regression_head=nn.Sequential(nn.Linear(features,4),nn.Tanh())
    def forward(self,x):
        z=self.dropout(self.backbone(x)); return {"family":self.family_head(z),"spokes":self.spokes_head(z),"center":self.center_head(z),"finish":self.finish_head(z),"regression":self.regression_head(z)}