import torch
import torch.nn as nn
from scipy.special import roots_legendre

class QuadratureWeightedLoss(nn.Module):
    def init(self, degree=5):
        super(QuadratureWeightedLoss, self).__init__()
        # استخراج نقاط و وزن‌های لژاندر (مرتبط با پایان‌نامه تو)
        nodes, weights = roots_legendre(degree)
        self.nodes = torch.tensor(nodes, dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, predictions, targets):
        # اینجا فرمول زیان رو با وزن‌های کوادراتور ترکیب می‌کنی
        base_loss = nn.functional.binary_cross_entropy_with_logits(predictions, targets)
        # یک وزن‌دهی نمادین بر اساس ریشه‌های چندجمله‌ای
        weighted_loss = base_loss * self.weights.mean() 
        return weighted_loss