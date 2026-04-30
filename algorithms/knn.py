import numpy as np

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        # k-NN is a lazy learner - just store the training data
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        # Make predictions for each test instance
        return np.array([self._predict_single(x) for x in X])
    
    def _predict_single(self, x):
        # Calculate Euclidean distance to all training points
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        
        # Find indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Get labels of k nearest neighbors
        k_labels = self.y_train[k_indices]
        
        # Return majority class (most common label)
        return np.bincount(k_labels.astype(int)).argmax()


def normalize_features(X_train, X_test):
    """Z-score normalization using training set statistics."""
    # Calculate mean and std from training data only
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    
    # Avoid division by zero for constant features
    std[std == 0] = 1
    
    # Apply same transformation to both sets
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    
    return X_train_norm, X_test_norm