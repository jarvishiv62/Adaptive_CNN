# mini_check.py — run this first to isolate the cause
import torch
print("1. CUDA:", torch.cuda.is_available())

print("2. Loading VGG...")
import torchvision.models as m
vgg = m.vgg16(weights='IMAGENET1K_V1')
print("   VGG OK")

print("3. VRAM test...")
x = torch.rand(4, 3, 256, 256).cuda()
print("   VRAM OK")

print("4. DataLoader test...")
from data.dataset import get_dataloaders
train_loader, _, _ = get_dataloaders()
batch = next(iter(train_loader))
print("   DataLoader OK:", batch['low'].shape)