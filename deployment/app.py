from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # اینجا عکس رو می‌گیری و به مدل هوش مصنوعی می‌دی
    return jsonify({"status": "success", "prediction": "Malignant/Benign"})

if name == '__main__':
    app.run(debug=True)