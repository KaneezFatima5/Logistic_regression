from config import *
from utils import *
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


def feature_scaling(x_data):
    """Perform standardization using standard scalar"""
    scalar=StandardScaler()
    return scalar.fit_transform(x_data), scalar
class SupportVectorMachine:
    def __init__(self):
        """Get the model from library"""
        self.model=SVC(kernel=KERNEL, random_state=42)
        self.scalar=None
    def train(self, train_df):
        """get feature matrix"""
        x=get_features(train_df)
        y=get_class_col(train_df)
        """save the scaler performed using training dataset to use it for standardizing test dataset"""
        x_scaled, self.scalar=feature_scaling(x)
        # Define the hyperparameter grid for GridSearchCV
        """ 'C' controls regularization strength — smaller C => stronger regularization"""
        """gamma controls the influence of a single training example — smaller gamma => wider influence"""
        param_grid= { "C": C, "gamma": GAMMA}
        """Create a GridSearchCV object to systematically test different combinations of C and gamma"""
        """# cv=5 -> use 5-fold cross-validation to evaluate each combination"""
        """# verbose=2 -> print progress and results for each parameter combination"""
        grid_search=GridSearchCV(self.model, param_grid, cv=5, verbose=2, n_jobs=-1)
        """train multiple svm models with different combinations of c and gamma"""
        grid_search.fit(x_scaled, y)
        """Get the best performing model for testing (based on cross-validation score)"""
        self.model=grid_search.best_estimator_

    def evaluate(self, test_df):
        y_id=get_index_col(test_df)
        x_test=get_features(test_df)
        x_test_scaled=self.scalar.transform(x_test)
        """predict values on trained model"""
        y_predict=self.model.predict(x_test_scaled)
        """Convert values of idx and predicted classes into a pd dataframe"""
        return pd.DataFrame({"id":y_id, "class":y_predict})
    
