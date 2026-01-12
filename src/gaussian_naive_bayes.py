from utils import *
from sklearn.naive_bayes import GaussianNB

class GaussianNaiveBayes:
    def __init__(self):
        """Get the model from library"""
        self.model=GaussianNB()
    def train(self, train_df):
        """get feature matrix"""
        x=get_features(train_df)
        y=get_class_col(train_df)
        """Train model on the train dataset"""
        self.model.fit(x, y)

    def evaluate(self, test_df):
        x=get_features(test_df)
        idx=get_index_col(test_df)
        """predict values on trained model"""
        y_pred=self.model.predict(x)
        """Convert values of idx and predicted classes into a pd dataframe"""
        return pd.DataFrame({"id": idx, "class": y_pred})


