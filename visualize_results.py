import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from models.cnn_model import PneumoniaCNN

def run_advanced_visualization(image_path):
    device = torch.device("cpu")
    model = PneumoniaCNN(num_classes=2)
    model.load_state_dict(torch.load('checkpoints/model_v1.pth', map_location=device))
    model.eval()

    img = Image.open(image_path).convert('RGB')
    rgb_img = np.float32(img.resize((224, 224))) / 255
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0)

    target_layers = [model.network.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    
    targets = [ClassifierOutputTarget(1)] 
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output).item()
        confidence = torch.nn.functional.softmax(output, dim=1)[0][prediction].item()


    classes = ['NORMAL', 'PNEUMONIA']
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    
    ax[0].imshow(img)
    ax[0].set_title(f"Original X-Ray\nPrediction: {classes[prediction]} ({confidence*100:.1f}%)")
    ax[0].axis('off')
    
    ax[1].imshow(visualization)
    ax[1].set_title("Grad-CAM Interpretability\n(Heatmap showing infected areas)")
    ax[1].axis('off')

    plt.tight_layout()
    if not os.path.exists('results'): os.makedirs('results')
    plt.savefig('results/gradcam_analysis.png', dpi=300)
    plt.show()
    print(f" Visualization saved in results/gradcam_analysis.png")

if __name__ == "__main__":
    
    test_img = 'data/raw/test/PNEUMONIA/person100_bacteria_475.jpeg'
    run_advanced_visualization(test_img)