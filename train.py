import os
import sys
import torch
import torch.optim as optim

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from core.dataset_loader import get_data_loaders
    from models.cnn_model import PneumoniaCNN
    from core.quadrature_loss import GaussianQuadratureLoss
    print(" All research modules linked successfully.")
except ImportError as e:
    print(f" Import Error: {e}")
    sys.exit()

def run_research_training():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Training Environment: {device}")

    
    try:
        train_dl, valid_dl, _ = get_data_loaders(data_dir='./data', batch_size=8)
    except Exception as e:
        print(f" Data Loading Error: {e}")
        return

    
    model = PneumoniaCNN(num_classes=2).to(device)
    
    criterion = GaussianQuadratureLoss(degree=5) 
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print(" Starting Controlled Training Session...")
    model.train()
    
    running_loss = 0.0
    total_batches = 200 

    for i, (images, labels) in enumerate(train_dl):
        images, labels = images.to(device), labels.to(device)
        
    
        outputs = model(images)
        
        
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        if i % 5 == 0:
            print(f" Batch [{i}/{total_batches}] | Loss: {loss.item():.4f}")

        if i >= total_batches:
            break

    print(f"\n Training Step Completed. Average Loss: {running_loss/total_batches:.4f}")


    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    
    save_path = 'checkpoints/model_v1.pth'
    torch.save(model.state_dict(), save_path)
    print(f" Model weights saved to: {save_path}")
    print(" Ready for the next phase: Visualization and Deployment.")

if __name__ == "__main__":
    run_research_training()