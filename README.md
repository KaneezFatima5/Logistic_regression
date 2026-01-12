🎵 Music Genre Classification using Classical Machine Learning

From raw audio → feature engineering → models from scratch → deep analysis

🔍 Overview

This project focuses on music genre classification using classical machine learning techniques.
The goal was to deeply understand the full ML pipeline, starting from raw audio processing to model implementation from scratch and thorough performance analysis.

🎯 Objective

Given an audio file, predict its music genre by extracting meaningful features and training machine learning models.

🔄 End-to-End Pipeline
Audio Files
   ↓
Feature Extraction
   ↓
Data Standardization & PCA
   ↓
Model Training (from scratch & sklearn)
   ↓
Evaluation & Visualization

🎼 Feature Extraction

Audio files were converted into structured numerical representations using:

MFCCs (Mel-Frequency Cepstral Coefficients)

Chroma features

Spectral Centroid

Spectral Bandwidth

Zero-Crossing Rate

Tempo-related features

These features capture both frequency and temporal characteristics of music.

🧠 Models Implemented
🔹 Logistic Regression (From Scratch)

Implemented using NumPy

Manual:

Loss computation

Gradient descent

Weight updates

Served as a baseline to understand optimization mechanics

🔹 Scikit-learn Models

Logistic Regression

Random Forest

Support Vector Machine (SVM)

Gaussian Naive Bayes

Used to compare learning behavior, bias–variance tradeoff, and performance.

⚙️ Data Preprocessing

Train–test split

Feature standardization

Dimensionality reduction using PCA

Reduced noise and improved generalization

📊 Evaluation Metrics

Models were evaluated using:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

📈 Visualization & Analysis

Training loss curves

Confusion matrices (heatmaps)

t-SNE visualization for feature separability

PCA variance analysis

These visualizations helped interpret model decision boundaries and data structure.

🔍 Key Learnings

Feature quality strongly impacts classical ML performance

PCA improves training stability and efficiency

Ensemble models (Random Forest) outperform linear models

SVM shows strong performance on high-dimensional data

Visualization is critical for interpretability

🛠️ Tech Stack

Python

NumPy

Scikit-learn

Librosa

Matplotlib

Seaborn

📁 Repository Structure
├── data/
├── feature_extraction/
├── models/
│   ├── logistic_regression_from_scratch.py
│   ├── random_forest.py
│   ├── svm.py
│   └── naive_bayes.py
├── evaluation/
├── visualization/
└── README.md