import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(data_dir='./data', batch_size=16): # بچ‌سایز ۱۶ برای رم ۸ گیگ ایمن‌تر است
    # مسیر دقیق پوشه‌هایی که دستی دانلود و کپی کردی
    base_path = os.path.join(data_dir, 'raw')
    
    train_dir = os.path.join(base_path, 'train')
    val_dir = os.path.join(base_path, 'val')
    test_dir = os.path.join(base_path, 'test')

    # تعریف تبدیل‌ها (Transforms)
    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    
    train_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    
    valid_tfms = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize(*stats)
    ])

    # لود کردن از فولدرها
    train_ds = datasets.ImageFolder(train_dir, train_tfms)
    valid_ds = datasets.ImageFolder(val_dir, valid_tfms)
    test_ds = datasets.ImageFolder(test_dir, valid_tfms)

    # ساخت لاودرها (num_workers=0 برای جلوگیری از ارور در ویندوز)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0)
    valid_dl = DataLoader(valid_ds, batch_size*2, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size*2, num_workers=0)

    print(f"✅ DataLoaders Ready: {len(train_ds)} train images found.")
    return train_dl, valid_dl, test_dl

if __name__ == "__main__":
    try:
        train_dl, _, _ = get_data_loaders()
        print("Success! Data is loaded correctly.")
    except Exception as e:
        print(f"❌ Error: {e}. Check if 'train' folder exists in data/raw/")