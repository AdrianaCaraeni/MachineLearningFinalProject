import numpy as np
from collections import Counter
from decision_tree import DecisionTree


class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, min_samples_split=2,
                 min_gain=0.0, random_state=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.random_state = random_state
        self.trees = []
        self.numerical_features = set()

    def fit(self, X, y, numerical_features=None):
        X = np.array(X)
        y = np.array(y)
        self.numerical_features = set(numerical_features) if numerical_features else set()

        n_features = X.shape[1]
        m = max(1, int(np.round(np.sqrt(n_features))))

        rng = np.random.RandomState(self.random_state)

        self.trees = []
        for i in range(self.n_trees):
            seed = rng.randint(0, 2**31)

            X_boot, y_boot = self._bootstrap_sample(X, y, seed)

            feature_indices = self._random_feature_subset(n_features, m,
                                                           np.random.RandomState(seed))

            tree = DecisionTree(
                criterion='information_gain',
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_gain=self.min_gain,
                feature_indices=feature_indices
            )
            tree.fit(X_boot, y_boot, numerical_features=self.numerical_features)
            self.trees.append(tree)

    def predict(self, X):
        X = np.array(X)
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.array([self._majority_vote(all_preds[:, i])
                         for i in range(X.shape[0])])

    def _bootstrap_sample(self, X, y, seed):
        rng = np.random.RandomState(seed)
        n = X.shape[0]
        indices = rng.choice(n, size=n, replace=True)
        return X[indices], y[indices]

    def _random_feature_subset(self, n_features, m, rng):
        return list(rng.choice(n_features, size=m, replace=False))

    def _majority_vote(self, predictions):
        return Counter(predictions).most_common(1)[0][0]