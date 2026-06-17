import sys
import os

# اضافه کردن مسیر فعلی پروژه به لیست مسیرهای پایتون
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.optim as optim
import os

# این همان خطی است که "پل" می‌زند بین فایل‌ها
from dataset_loader import get_data_loaders 
from cnn_model import PneumoniaCNN
from quadrature_loss import GaussianQuadratureLoss

def start_training():
    # ۱. انتخاب سخت‌افزار
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ System is using: {device}")

    # ۲. فراخوانی داده‌ها
    # ما آدرس پوشه data را به تابع می‌دهیم
    try:
        train_dl, valid_dl, _ = get_data_loaders(data_dir='./data', batch_size=8)
        print("📁 Data successfully linked to the Trainer.")
    except Exception as e:
        print(f"❌ Error linking data: {e}")
        return

    # ۳. آماده‌سازی مدل و توابع ریاضی
    model = PneumoniaCNN(num_classes=2).to(device)
    criterion = GaussianQuadratureLoss(degree=5) # تابع زیان ریاضی تو
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # ۴. شروع آموزش (فقط برای تست)
    print("🚀 Training process started...")
    model.train()
    
    # گرفتن یک "بچ" از داده‌ها برای تست
    images, labels = next(iter(train_dl))
    images, labels = images.to(device), labels.to(device)
    
    # اجرای مدل
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # آپدیت وزن‌ها
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"✅ Success! First batch processed. Loss: {loss.item():.4f}")
    print("💾 Everything is working! Ready for full training.")

if __name__ == "__main__":
    start_training()