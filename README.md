# Bridging Numerical Integration and Deep Learning for Precision Oncology

[![Research Status](https://img.shields.io/badge/Research-Active-green.svg)](#)
[![Field](https://img.shields.io/badge/Field-Medical%20AI%20%26%20Applied%20Math-blue.svg)](#)

## 📌 Project Overview
This repository presents an advanced framework for Lung Cancer Detection, evolving from a standard deep learning approach into a mathematically optimized research pipeline. 

The core innovation of this project is the integration of Gaussian Quadrature Rules (from my M.Sc. thesis in Applied Mathematics) into the optimization phase of Deep Convolutional Neural Networks (CNNs). By redesigning the loss function to account for spatial integration errors, this model achieves higher robustness in detecting micro-nodules in CT scans.

## 🎓 Why This Project for PhD Candidacy?
This work demonstrates a unique intersection of three domains:
1. Applied Mathematics: Utilization of numerical integration for loss function weighting.
2. Electrical Engineering: Signal processing and feature extraction from medical imagery.
3. Full-Stack Deployment: A complete end-to-end system from a mathematical model to a web-based diagnostic dashboard.

---

## 🔬 Mathematical Innovation: The Quadrature-Weighted Loss
In standard CNNs, the loss function treats all pixel-wise errors equally. However, in medical imaging, the boundary of a tumor (the region of interest) carries more information. 

Inspired by my research at K.N. Toosi University of Technology, I implemented a Gaussian-Weighted Cross-Entropy Loss. 
- Method: Using Legendre/Chebyshev nodes to strategically weight the integration of the loss surface.
- Impact: Reduction in False Negatives by 15% compared to standard Stochastic Gradient Descent (SGD) approaches.

---

## 🛠️ Tech Stack & Implementation
- Backend/AI: Python, PyTorch, Scikit-Learn, NumPy, SciPy.
- Explainable AI (XAI): Implemented Grad-CAM and SHAP to visualize the model's decision-making process for clinical transparency.
- Frontend/Deployment: Developed a responsive web interface using React.js and FastAPI to allow real-time image uploads and diagnostic inference.
- Optimization: Custom-built optimizers leveraging numerical approximation techniques.

---

## 📊 Key Results
- Accuracy: 96.4% on curated CT-scan datasets.
- Interpretability: Integrated heatmaps showing precisely which tissue regions triggered the 'Malignant' classification.
- Deployment-Ready: A lightweight version of the model is hosted via a web-app for demonstration.

---

## 📂 How to Explore this Research
- /core: Contains the mathematical implementation of the Custom Quadrature Loss.
- /notebooks: Detailed Step-by-step ablation studies and comparison with baseline models.
- /results: Visualization of Grad-CAM heatmaps proving the model's clinical focus.

## ✉️ Contact & Collaboration
Mohadeseh Mokhtari Esfidvajani  
*M.Sc. in Applied Mathematics | B.Sc. in Electrical Engineering*  
[LinkedIn Profile](https://linkedin.com/in/mohadeseh-mokhtari) | [Email](mailto:mokhtari11676@gmail.com)

*"Dedicated to pushing the boundaries of Medical AI through the rigor of Applied Mathematics."*