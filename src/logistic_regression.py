from utils import *
def softmax(z):
    """Normalize the array values"""
    z = z - np.max(z, axis=1, keepdims=True)
    exp_s = np.exp(z)
    return exp_s / np.sum(exp_s, axis=1, keepdims=True)


def add_intercept(x):
    """X=features Matrix"""
    n = x.shape[0]
    bias_vector = np.ones((n, 1))
    """Adding bias column to the feature matrix"""
    return np.hstack([bias_vector, x])


def standardize(train_dataset, dataset):
    """Calculating mean and standard daviation of train dataset and using those values to standardize both train and test dataset"""
    mean = train_dataset.mean(axis=0)
    std = train_dataset.std(axis=0)
    return (dataset - mean) / std

class LogisticRegression:
    def __init__(self, train_df, class_column, drop_columns, fit_intercept=True):
        self.classes = train_df[class_column].unique()
        self.num_classes = len(self.classes)
        self.fit_intercept = fit_intercept
        self.drop_columns = drop_columns
        self.train_df = get_features(train_df)
        m = self.train_df.shape[1]
        if self.fit_intercept:
            m=m+1
        """Initialize Weights (features X classes)"""
        self.w = np.zeros((m, self.num_classes))
        self.losses = []

    def train(self, train_df, class_column, lr=0.1, reg=0.1, epochs=5000):
        x = get_features(train_df)
        x = standardize(x, x)
        """Adding bias term"""
        if self.fit_intercept:
            x = add_intercept(x)
        n, m = x.shape
        y = get_class_col(train_df)

        """convert target array into indexes"""
        y_idx = np.array([np.where(self.classes == yi)[0][0] for yi in y])
        """y_onehot dimension=no.of instance x classes"""
        y_onehot = np.zeros((n, self.num_classes))
        y_onehot[np.arange(n), y_idx] = 1

        for epoch in range(epochs):
            """nxk"""
            logit = x.dot(self.w)
            prob = softmax(logit)
            """Calculate Loss function -- excluding the bias term in the regularization function """
            loss = -np.sum(y_onehot * np.log(prob + 1e-12)) / n
            loss += reg * np.sum(self.w[1:] ** 2) / (2 * n)
            self.losses.append(loss)
            reg_grad = reg * self.w
            """Setting the regularization =0 for bias term"""
            reg_grad[0, :] = 0
            """Calculate gradient descent"""
            gradient = x.T.dot(y_onehot - prob) - reg_grad  # n*k
            """update weights"""
            self.w += lr * gradient / n

    def predict_probabilities(self, x):
        if self.fit_intercept:
            x = add_intercept(x)
        return softmax(x.dot(self.w))

    def evaluate(self, test_df):
        """Test feature matrix"""
        test_features_df = get_features(test_df)
        """standardize test data """
        test_features_df = standardize(self.train_df, test_features_df)
        y_id = get_index_col(test_df)
        """convert feature values into probabilities"""
        p = self.predict_probabilities(test_features_df)
        """Get the max argument out of the row of probabilities of different classes"""
        idx = np.argmax(p, axis=1)
        """Get the class name at that index"""
        y_evaluations = self.classes[idx]
        """Convert it into a pandas dataframe"""
        return pd.DataFrame({"id": y_id, "class": y_evaluations})

