import torch
import torch.optim as optim
from data.dataset_loader import get_data_loaders
from models.cnn_model import PneumoniaCNN
from core.quadrature_loss import GaussianQuadratureLoss

def train_model():
    # ۱. تنظیمات اولیه
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on: {device}")

    # ۲. فراخوانی داده‌ها
    train_dl, valid_dl, _ = get_data_loaders(batch_size=16)

    # ۳. فراخوانی مدل و انتقال به GPU/CPU
    model = PneumoniaCNN(num_classes=2).to(device)

    # ۴. تعریف تابع زیان ریاضی تو و بهینه‌ساز
    criterion = GaussianQuadratureLoss(degree=5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ۵. حلقه آموزش (برای تست فعلاً ۱ اپوک)
    print("⏳ Starting Training...")
    model.train()
    
    for images, labels in train_dl:
        images, labels = images.to(device), labels.to(device)
        
        # پیش‌بینی
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # پس‌انتشار (Backpropagation)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Current Batch Loss: {loss.item():.4f}", end="\r")
        break # برای تست فعلاً بعد از اولین بچ متوقف می‌کنیم

    print("\n✅ Initial Test Successful! The Math and AI are working together.")

if __name__ == "__main__":
    train_model()