Advanced Medical AI: Numerical Optimization meets Deep Learning

Pneumonia Detection using Gaussian Quadrature-Based Loss and Uncertainty Estimation

Python PyTorch Flask

📌 Project Overview

This research-oriented project bridges the gap between Applied Mathematics and
Deep Learning. Leveraging my background in Electrical Engineering and Applied
Mathematics, I developed a framework that replaces standard Cross-Entropy loss
with a custom Gaussian Quadrature-based Loss Function to improve numerical
stability and sensitivity in medical diagnostics.

🔬 Mathematical Innovation: Quadrature-Weighted Loss

The core contribution is the integration of Legendre Polynomial nodes into the
neural network's optimization phase. By treating the loss surface as a
continuous integral and approximating it via Gaussian Quadrature, the model
achieves superior convergence in high-entropy regions.

Research Achievement: My proposed method reached a 93.1% Recall (Sensitivity),
significantly reducing the rate of false negatives in pneumonia detection—a
critical factor in clinical safety.

📸 Visual Results & Explainable AI (XAI)

1. Model Interpretability (Grad-CAM)

To ensure the model is making decisions based on pathological features rather
than noise, I implemented Grad-CAM. This highlights the specific lung regions
the model focuses on for diagnosis.

2. Clinical Decision Support System (Web App)

A production-ready Flask Web Interface where clinicians can upload X-rays and
receive an instant diagnosis, confidence score, and Uncertainty Rating.

3. Scientific Benchmark

| Metric                   | Standard CNN (Baseline) | **Our Quadrature-Enhanced Model** |
| :----------------------- | :---------------------- | :-------------------------------- |
| **Accuracy**             | 91.2%                   | **86.0%** (Controlled Training)   |
| **Recall (Sensitivity)** | 88.2%                   | **93.1%** 🚀                       |
| **Uncertainty Score**    | N/A                     | **0.02** (Low)                    |

🛠️ Technical Stack

  - AI Core: PyTorch, Torchvision, Grad-CAM (Interpretability).
  - Math Engine: SciPy (Roots Legendre), NumPy, Optimization.
  - Full-Stack: Flask (Backend), Streamlit (Research Dashboard), Chart.js.
  - Analytics: Scikit-learn (Ablation Studies), Pandas, Plotly.

📂 Project Structure

├── core/               # Mathematical Loss Functions & Data Loaders
├── models/             # CNN Architectures & Saved Weights (.pth)
├── templates/          # Clinical Web Frontend
├── static/             # Assets, CSS, and Uploaded X-rays
├── results/            # Grad-CAM outputs and CSV Benchmarks
├── Technical_Report.pdf # Comprehensive 40-page Research Summary
└── run_pipeline.py     # Automated Research Execution

👩‍🔬 About the Author

Mohadeseh Mokhtari Esfidvajani

  - M.Sc. in Applied Mathematics (K.N. Toosi University of Technology)
  - B.Sc. in Electrical Engineering (Electronics)
  - 8+ Years of Math Instruction | 6+ Years in AI Research.
  - Focus: Numerical Optimization, Reliable AI, Medical Imaging.