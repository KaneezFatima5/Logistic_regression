🎵 Music Genre Classification using Classical Machine Learning
============================================
From raw audio → feature engineering → models from scratch → data analysis

🔍 Overview
------------
This project focuses on music genre classification using classical machine learning techniques.
The goal was to deeply understand the full ML pipeline, starting from raw audio processing to model implementation from scratch and thorough performance analysis.

🔄 End-to-End Pipeline
----------------------
Audio Files
   ↓
Feature Extraction
   ↓
Data Standardization & PCA
   ↓
Model Training (from scratch & sklearn)
   ↓
Evaluation & Visualization

### 1\. Feature Extraction
-----------------------
Audio files were converted into structured numerical representations using:

-   MFCCs (Mel-Frequency Cepstral Coefficients)
-   Chroma features
-   Spectral Centroid
-   Spectral Bandwidth
-   Zero-Crossing Rate
-   Tempo-related features

These features capture both frequency and temporal characteristics of music.

### 2\. Models Implemented
🔹 Logistic Regression (From Scratch)
-------------------------------------
Implemented using NumPy

-   Loss computation
-   Gradient descent
-   Weight updates

Served as a baseline to understand optimization mechanics

### 3\. Classification and Regression Model Comparison

-   Logistic Regression
-   Random Forest
-   Support Vector Machine (SVM)
-   Gaussian Naive Bayes

Used to compare learning behavior, bias–variance tradeoff, and performance.

### 4\. Data Preprocessing

-   Train–test split
-   Feature standardization
-   Dimensionality reduction using PCA
-   Reduced noise and improved generalization

### 5\. Evaluation Metrics

Models were evaluated using:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   Confusion Matrix

### 6\. Visualization & Analysis

-   Training loss curves
-   Confusion matrices (heatmaps)
-   t-SNE visualization for feature separability
-   PCA variance analysis

These visualizations helped interpret model decision boundaries and data structure.

🔍 Key Learnings
------------------
-   Feature quality strongly impacts classical ML performance
-   PCA improves training stability and efficiency
-   Ensemble models (Random Forest) outperform linear models
-   SVM shows strong performance on high-dimensional data
-   Visualization is critical for interpretability

🛠️ Tech Stack
---------------
-   Python
-   NumPy
-   Scikit-learn
-   Librosa
-   Matplotlib
-   Seaborn

📁 Repository Structure
### Running Code

There exist 3 different options for running this program, which are all detailed below.
Common Steps

First, verify that the paths in ./src/config.py are all valid, and that the hyperparameters are set to what you desire.
You must be in the src folder when running anything.
Training and Evaluating model.

To train and evaluate a model, simply run main.py after doing the common steps. The model will be saved at the location specified. You will still need to upload the submission dataset to kaggle manually.

python3 main.py

### Training a model.

To simply train a model, run train.py after doing the common steps. The model will be saved at the location specified.

python3 train.py

### Evaluating a model.

To evaluate a model, run evaluate.py after doing the common steps. You will still need to upload the submission dataset to kaggle manually.

python3 evaluate.py


### Contributions

Sathvik Quadros - Wrote code for project structure, features extraction functions for tempo and beates, gradient descent, unit test cases, visualization for misclassified samples, and worked on the report.
Fatima - Wrote code for rest of the feature extraction functions, logistic regression, random forest, svm and gaussian naive bayes, and worked on the readme file

### Kaggle Score, Accuracy, and Date Run
Kaggle Score - 0.69
Average accuracy on validation set - 0.68
Date run - November 8th, 2025