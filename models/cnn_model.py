import torch
import torch.nn as nn
import torchvision.models as models

class PneumoniaCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(PneumoniaCNN, self).__init__()
        # استفاده از یک مدل پیش‌آموزش دیده (ResNet) - بسیار حرفه‌ای برای دکتری
        self.network = models.resnet18(pretrained=True)
        
        # تغییر لایه آخر برای انطباق با تعداد کلاس‌های ما (سالم و بیمار)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, xb):
        return self.network(xb)

print("✅ Model architecture (ResNet18) initialized.")