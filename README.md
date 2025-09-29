# Anomaly Detection for Financial Transactions with Dynamic Neural ODEs and Isolation Forest

## Overview

In the realm of financial security, detecting fraudulent transactions in real-time is essential to prevent losses and maintain trust in banking systems. This project presents an innovative anomaly detection system that integrates Neural Ordinary Differential Equations (Neural ODEs) with Isolation Forest, tailored for identifying fraud in credit card transactions. By modeling transaction data as dynamic trajectories through Neural ODEs and isolating anomalies via ensemble-based forestry, the tool achieves high accuracy on imbalanced datasets, where frauds are rare (less than 0.2% of transactions).

Focused on unsupervised learning, this approach trains on normal transactions to learn typical patterns, flagging deviations as potential fraud. The system is efficient, GPU-accelerated, and includes an interactive UI for inference, making it suitable for financial institutions, fintech companies, and researchers. Built for scalability, it demonstrates ROC-AUC scores of ~0.98 and PR-AUC of ~0.81, highlighting its effectiveness in reducing false positives while catching anomalies.

This repository contains a Jupyter Notebook implementation, optimized for Google Colab with GPU support for training and real-time predictions.

## Features

- Data Preprocessing with Standardization and Normalization: Applies StandardScaler to PCA features and Amount, while normalizing Time to integrate temporal dynamics.
- Dynamic Embeddings via Neural ODEs: Models transaction evolution over artificial time using ODE solvers (Runge-Kutta 4) to generate robust representations for anomaly separation.
- Unsupervised Anomaly Detection with Isolation Forest: Fits on normal embeddings to compute anomaly scores, with a heuristic threshold for classification.
- Comprehensive Evaluation: Includes ROC-AUC, PR-AUC metrics, score distribution histograms, and ROC curve visualizations for model assessment.
- Professional Interactive UI: Built with ipywidgets, featuring input textarea, sample loading buttons, and color-coded results for user-friendly inference.
- Fraud-Focused Design: Handles imbalanced data by subsampling normals, ensuring focus on rare events like fraud without labeled training.
- Extensible Pipeline: Easily adaptable for larger datasets or integration with streaming systems (e.g., Kafka for real-time transaction feeds).

This tool not only detects anomalies but also provides interpretable scores, aiding compliance with financial regulations like PCI DSS.

## Data Source

* Kaggle Dataset: Utilizes the "Credit Card Fraud Detection" dataset from Kaggle (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), containing ~284,807 anonymized transactions with 492 frauds.
* Usage in Project: Processes features V1-V28 (PCA-transformed), Amount, and Time; subsamples 10,000 normals for efficiency while retaining all anomalies.
* Focus: Emphasizes PCA anonymity for privacy, with key indicators like transaction amount skewness and time-based patterns for fraud detection.
* Limitation: The dataset uses anonymized PCA-transformed features (V1-V28), which are unitless and not directly interpretable. Users must input these specific values for inference, which may be impractical for real-world use without access to the PCA transformation. For more user-friendly inputs, consider adapting to datasets with original features (e.g., simulated fraud data with fields like transaction type, merchant, etc.).

## Technology Stack

* Core ML: PyTorch for Neural ODEs, torchdiffeq for ODE solving, scikit-learn for Isolation Forest and scaling.
* Data Handling: pandas and NumPy for preprocessing; matplotlib and seaborn for visualizations.
* UI & Interactivity: ipywidgets and IPython for the professional inference interface.
* Environment: Python 3.x with dependencies like torch, scikit-learn, and kaggle managed via pip or Colab; GPU acceleration via CUDA.
* Additional: Kaggle API for dataset download, ensuring reproducibility.

This stack enables fast training (50 epochs in minutes on GPU) and inference under 1 second per transaction.

## Workflow

1. Dataset Loading and Preprocessing: Download via Kaggle, standardize features, normalize Time, and split into normal/anomalous subsets.
2. Neural ODE Training: Define ODE function as MLP, train with MSE reconstruction loss on normal data using Adam optimizer.
3. Embedding Extraction: Evolve inputs through ODE from t=0 to t=1 to get dynamic embeddings.
4. Isolation Forest Fitting: Train on normal embeddings, compute scores for all data.
5. Evaluation: Calculate ROC-AUC/PR-AUC, visualize distributions and curves, test sample inferences.
6. UI Inference: Parse user input, preprocess, extract embedding, score with Isolation Forest, and display flagged results with explanations.

## Outputs

### Anomaly Score Distribution
![Anomaly Score Distribution](https://github.com/AashishSaini16/Anomaly_Detection_for_Financial_Transactions_with_Dynamic_Neural_ODEs_and_Isolation_Forest/blob/main/Anomaly_Score_Distribution.PNG)

### ROC Curve
![Receiver Operating Characteristic (ROC)](https://github.com/AashishSaini16/Anomaly_Detection_for_Financial_Transactions_with_Dynamic_Neural_ODEs_and_Isolation_Forest/blob/main/ROC.PNG)

### User Interface
![Transaction Based Anomaly Detection Interface](https://github.com/AashishSaini16/Anomaly_Detection_for_Financial_Transactions_with_Dynamic_Neural_ODEs_and_Isolation_Forest/blob/main/Output.PNG)
