import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import roots_legendre

class GaussianQuadratureLoss(nn.Module):
    def __init__(self, degree=5, sigma=1.0):
        super(GaussianQuadratureLoss, self).__init__()
        # استخراج نقاط و وزن‌های کوادراتور لژاندر
        nodes, weights = roots_legendre(degree)
        self.nodes = torch.tensor(nodes, dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.sigma = sigma

    def forward(self, logits, targets):
        # ۱. محاسبه خطای پایه (Cross Entropy)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # ۲. تبدیل پیش‌بینی‌ها به احتمال (Softmax)
        probs = F.softmax(logits, dim=1)
        conf = torch.gather(probs, 1, targets.unsqueeze(1)).squeeze()

        # ۳. بخش خلاقانه ریاضی: وزن‌دهی بر اساس کوادراتور
        # ما فرض می‌کنیم توزیع خطا یک انتگرال است که با کوادراتور تقریب می‌زنیم
        quad_weight = torch.zeros_like(conf)
        for i in range(len(self.nodes)):
            # استفاده از گره‌ها و وزن‌های لژاندر برای اصلاح سطح زیان
            quad_weight += self.weights[i] * torch.exp(-((conf - self.nodes[i])**2) / (2 * self.sigma**2))
        
        # ۴. ترکیب خطای نهایی
        weighted_loss = ce_loss * quad_weight
        return weighted_loss.mean()

print("🧠 Gaussian Quadrature Loss Function (Thesis-Informed) initialized.")