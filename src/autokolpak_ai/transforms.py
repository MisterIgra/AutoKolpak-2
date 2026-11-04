from PIL import Image, ImageOps
import torch
from torchvision import transforms
from .constants import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

class PadToSquare:
    def __init__(self, fill=(32, 32, 32)): self.fill = fill
    def __call__(self, image):
        image = image.convert("RGB")
        width, height = image.size
        size = max(width, height)
        canvas = Image.new("RGB", (size, size), self.fill)
        canvas.paste(image, ((size - width) // 2, (size - height) // 2))
        return canvas

def training_transform(size=IMAGE_SIZE):
    return transforms.Compose([PadToSquare(), transforms.Resize((size, size)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(180), transforms.RandomAffine(10, translate=(.05, .05), scale=(.9, 1.1), shear=5), transforms.ColorJitter(.15, .15, .15, .05), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

def inference_transform(size=IMAGE_SIZE):
    return transforms.Compose([PadToSquare(), transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])