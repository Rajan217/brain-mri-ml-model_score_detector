**

# 🧠 Brain MRI ML Model Score Detector

This project implements multiple **Machine Learning algorithms** to analyze Brain MRI images and detect tumors using unsupervised, supervised, and probabilistic models.

---

## 📘 Project Overview

The goal is to categorize MRI scans as **tumor** or **no tumor** using models like:
- K-Means Clustering  
- Gaussian Mixture Models (EM Algorithm)  
- Hierarchical Clustering  
- Principal Component Analysis (PCA)  
- Hidden Markov Models (HMM)  
- CART (Decision Tree)  
- Ensemble Learning (Random Forest, AdaBoost)

---

## ⚙️ How to Run

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt

2️⃣ Run all algorithms sequentially
python run_all.py


Each model script (kmeans.py, gmm.py, pca.py, etc.) will run and display performance metrics.

🧩 Folder Structure
Brain MRI Images/
│
├── kmeans.py
├── gmm.py
├── hierarchical.py
├── pca.py
├── hmm.py
├── cart.py
├── ensemble.py
├── utils.py
├── run_all.py
├── .gitignore
└── requirements.txt

📦 Dataset

Dataset used:
Brain MRI Images for Brain Tumor Detection – Kaggle

Please download it manually and place it in the /data folder.

📊 Output

Each model prints its:

Accuracy and classification report (for supervised)

Cluster visuals (for unsupervised)

Dimensionality reduction plots (for PCA)

👨‍💻 Author

Rajan Prajapati
Machine Learning & AI Enthusiast**
