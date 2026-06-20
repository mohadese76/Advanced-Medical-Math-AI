from flask import Flask, render_template, request, jsonify
import torch
import os
from torchvision import transforms
from PIL import Image
from models.cnn_model import PneumoniaCNN # مدل خودت

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#1. Loading the model and weights
device = torch.device("cpu")
model = PneumoniaCNN(num_classes=2)
checkpoint_path = 'checkpoints/model_v1.pth'
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# 2. Prediction function with uncertainty (MC Dropout)
def predict_image(image_path, num_samples=10):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0)
    
    model.train() # Enable Dropout to calculate Uncertainty
    preds = []
    with torch.no_grad():
        for _ in range(num_samples):
            output = model(input_tensor)
            preds.append(torch.nn.functional.softmax(output, dim=1))
    
    preds = torch.stack(preds)
    mean_pred = preds.mean(0)
    uncertainty = preds.std(0).mean().item()
    
    prediction = torch.argmax(mean_pred).item()
    confidence = mean_pred[0][prediction].item()
    
    return prediction, confidence, uncertainty

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        pred, conf, unc = predict_image(file_path)
        
        result = {
            'diagnosis': 'PNEUMONIA (Infected)' if pred == 1 else 'NORMAL (Healthy)',
            'confidence': f"{conf*100:.2f}%",
            'uncertainty': f"{unc:.4f}",
            'image_url': file_path
        }
        return jsonify(result)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER): os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)