🎵 Music Genre Classification using Machine Learning
📌 Project Overview

This project focuses on predicting the genre of music from raw audio files using classical machine learning techniques. The goal was to explore the full ML pipeline, starting from audio feature extraction to model training, evaluation, and comparative analysis of different algorithms.

The project emphasizes understanding model behavior, feature engineering, and interpretability rather than treating models as black boxes.

🧠 Problem Statement

Given a dataset of audio music files, predict the genre of each track by extracting meaningful features and training machine learning models capable of learning discriminative patterns from audio signals.

🔄 Pipeline Overview
Audio Files → Feature Extraction → Preprocessing → Dimensionality Reduction
            → Model Training → Evaluation → Visualization & Analysis

🎼 Feature Extraction

Raw audio signals were transformed into structured numerical representations using signal-processing techniques.

Converted audio files into spectrogram-based features

Extracted time–frequency domain characteristics suitable for ML models

Ensured consistent sampling and feature dimensionality across files

This step bridges the gap between raw audio and classical ML models.

⚙️ Data Preprocessing

To improve model performance and stability:

Standardization was applied to normalize feature distributions

Principal Component Analysis (PCA) was used to:

Reduce dimensionality

Remove feature redundancy

Improve computational efficiency

Dataset was split into training and testing sets to ensure unbiased evaluation

🧪 Models Implemented
🔹 Logistic Regression (From Scratch & scikit-learn)

Implemented Logistic Regression from scratch using NumPy

Compared results with scikit-learn implementation

Trained on reduced PCA feature space

Served as a baseline linear model

🔹 Random Forest Classifier

Captured non-linear relationships between features

Analyzed feature interactions and ensemble behavior

🔹 Gaussian Naive Bayes

Evaluated probabilistic assumptions on audio features

Used as a lightweight baseline for comparison

🔹 Support Vector Machine (SVM)

Tested margin-based classification on transformed feature space

Compared kernel behavior and generalization performance

📊 Model Evaluation

Models were evaluated using multiple performance metrics to ensure robust analysis:

Accuracy

Precision

Recall

F1-score

Confusion Matrix for class-wise performance analysis

This multi-metric approach helped identify strengths and weaknesses across genres.

📈 Visualization & Analysis

To better understand data and model behavior:

Loss curves to analyze convergence during training

Heatmaps for correlation and confusion matrix visualization

t-SNE visualization to observe class separability in reduced feature space

Comparative analysis of model decision boundaries and performance trends

🔍 Key Insights

Feature preprocessing and dimensionality reduction significantly impact model performance

Linear models benefit from PCA, while ensemble methods handle raw feature interactions better

Different models exhibit distinct bias–variance tradeoffs on audio data

Visualization tools (t-SNE, heatmaps) provide valuable interpretability beyond metrics

🛠️ Tech Stack

Python

NumPy

scikit-learn

Matplotlib / Seaborn

Librosa (audio processing)

📁 Repository Structure
├── data/
│   ├── raw_audio/
│   └── processed_features/
├── feature_extraction/
├── models/
│   ├── logistic_regression_from_scratch.py
│   ├── random_forest.py
│   ├── svm.py
│   └── naive_bayes.py
├── evaluation/
├── visualization/
├── notebooks/
├── utils/
└── README.md

🚀 How to Run

Clone the repository

git clone https://github.com/your-username/music-genre-classification.git
cd music-genre-classification


Install dependencies

pip install -r requirements.txt


Run feature extraction

Train and evaluate models using provided scripts or notebooks

📌 Future Work

Extend to deep learning–based models (CNNs on spectrograms)

Explore advanced audio features (MFCC deltas, chroma features)

Perform hyperparameter tuning and cross-validation

Compare classical ML vs deep learning performance

### Repo Structure
The repository is organized into 6 main folders:
* checkpoints - Contains current best model from NN and Transfer learning.
* data - Contains all the datasets and submission data.
* models - Contains the current best model, and is where trained models are stored using pickle.
* report - Contains the tex files for the report.
* src - Contains source files for the project, also contains config.py.
* tests - Contains unit tests for various parts of code in src.

## File Manifest

### ./models
* best_model_epoch=x_val_loss=y.ckpt - The current best model, saved using pytorch checkpoints.

### ./data
This folder may be empty as the dataset must be removed when submitting.
* music folder - Contains test and train folders containing audio music files respectively 
* spreadsheets folder - Contains following 
    * test.csv - The test dataset containing all extracted features.
    * train.csv - The train dataset containing all extracted features.
    * sub.csv - The file storing the submission to kaggle.
* list_test.csv - ids for test dataset.
* potential_outliers.txt - Analyzed data of outliers.

### ./models
* selected.pkl - The current best model, saved using pickle.

### ./report
* report.pdf - The compiled report.
* report.tex - The uncompiled tex file of the report.
* /report_archive - Old report format that we planned on using before swapping over to a simpler one due to it not being part of the rubric.

### ./src
* config.py - Contains all configurable hyperparameters.
* extract.py - extract various features from the audio files.
* train.py - Trains a models (logistic regression, random forest, support vector machines, gaussian naive bayes) using configuration information from config.py, saves the model at the path given by config.py
* evaluate.py - Uses the model stored at the given path in config.py, evaluates all the test data and stores the output in the path given by config.py
* main.py - Simply calls the functions from evaluate.py and train.py, while passing configuration information from config.py
* logistic_regression.py - Contains the code for the logistic regression model, standardizaion and PCA.
* gaussian_naive_bayes.py - Contains the code gaussian naive bayes model.
* random_forest.py - Contains the code for random forest model.
* svm.py - Contains the code for support vector machines model.
* visualization.py - Visualizing the misclassified samples.
* visualize.py - Simply calls the functions from visualization.py.
* utils.py - Contains some utility functions such as load spreadsheets, get feature matrix, get class matrix etc.

### ./tests
* test_gradient_descent.py - Contains unit tests for some of the functions in gradient_descent.py
* test_logistic_regression.py - Contains unit tests for the functions in logistic_regression.py

## Running Code
There exist 3 different options for running this program, which are all detailed below.

### Common Steps
First, verify that the paths in `./src/config.py` are all valid, and that the hyperparameters are set to what you desire. 

You must be in the src folder when running anything.

### Training and Evaluating model.
To train and evaluate a model, simply run main.py after doing the common steps. The model will be saved at the location specified. You will still need to upload the submission dataset to kaggle manually.
```
python3 main.py
```

### Training a model.
To simply train a model, run train.py after doing the common steps. The model will be saved at the location specified.
```
python3 train.py
```

### Evaluating a model.
To evaluate a model, run evaluate.py after doing the common steps. You will still need to upload the submission dataset to kaggle manually.

```
python3 evaluate.py
```

## Contributions
* Sathvik Quadros - Wrote code for project structure, features extraction functions for tempo and beates, gradient descent, unit test cases, visualization for misclassified samples, and worked on the report.
* Fatima - Wrote code for rest of the feature extraction functions, logistic regression, random forest, svm and gaussian naive bayes, and worked on the readme file 

## Kaggle Score, Accuracy, and Date Run
* Kaggle Score - 0.69
* Average accuracy on validation set - 0.68
* Date run - November 8th, 2025