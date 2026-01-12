# CS529-Neural Network 
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