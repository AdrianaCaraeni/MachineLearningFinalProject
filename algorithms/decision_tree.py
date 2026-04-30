import numpy as np
from collections import Counter


class Node:
    def __init__(self, feature=None, threshold=None, value=None, is_leaf=False, is_numerical=False):
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.children = {}
        self.left = None
        self.right = None
        self.is_leaf = is_leaf
        self.is_numerical = is_numerical


class DecisionTree:
    def __init__(self, criterion='information_gain', max_depth=None,
                 min_samples_split=2, min_gain=0.0, feature_indices=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.feature_indices = feature_indices
        self.root = None
        self.numerical_features = set()

    def fit(self, X, y, numerical_features=None):
        X = np.array(X)
        y = np.array(y)
        if numerical_features is not None:
            self.numerical_features = set(numerical_features)
        self.root = self._build_tree(X, y, depth=0)

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_single(x, self.root) for x in X])

    def _build_tree(self, X, y, depth):
        n_samples = len(y)

        if len(np.unique(y)) == 1:
            return Node(value=y[0], is_leaf=True)

        if self.max_depth is not None and depth >= self.max_depth:
            return Node(value=self._most_common_label(y), is_leaf=True)

        if n_samples < self.min_samples_split:
            return Node(value=self._most_common_label(y), is_leaf=True)

        feature_indices = self.feature_indices if self.feature_indices is not None \
            else list(range(X.shape[1]))

        best_feature, best_threshold, best_gain = self._best_split(X, y, feature_indices)

        if best_feature is None or best_gain <= self.min_gain:
            return Node(value=self._most_common_label(y), is_leaf=True)

        is_num = best_feature in self.numerical_features

        if is_num:
            node = Node(feature=best_feature, threshold=best_threshold,
                        is_numerical=True, is_leaf=False)
            left_mask = X[:, best_feature].astype(float) <= best_threshold
            node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
            node.right = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        else:
            node = Node(feature=best_feature, is_numerical=False, is_leaf=False)
            for val in np.unique(X[:, best_feature]):
                mask = X[:, best_feature] == val
                node.children[val] = self._build_tree(X[mask], y[mask], depth + 1)

        return node

    def _best_split(self, X, y, feature_indices):
        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        if self.criterion == 'information_gain':
            parent_impurity = self._entropy(y)
        else:
            parent_impurity = self._gini_index(y)

        for fi in feature_indices:
            if fi in self.numerical_features:
                gain, threshold = self._best_numerical_gain(X, y, fi, parent_impurity)
            else:
                threshold = None
                if self.criterion == 'information_gain':
                    gain = self._information_gain(X, y, fi, parent_impurity)
                else:
                    gain = self._gini_gain(X, y, fi, parent_impurity)

            if gain > best_gain:
                best_gain = gain
                best_feature = fi
                best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _best_numerical_gain(self, X, y, feature_idx, parent_impurity):
        col = X[:, feature_idx].astype(float)
        sorted_vals = np.sort(np.unique(col))

        if len(sorted_vals) == 1:
            return 0.0, sorted_vals[0]

        thresholds = (sorted_vals[:-1] + sorted_vals[1:]) / 2.0

        best_gain = -np.inf
        best_thresh = thresholds[0]

        n = len(y)
        for t in thresholds:
            left_mask = col <= t
            right_mask = ~left_mask
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue

            if self.criterion == 'information_gain':
                weighted = (left_mask.sum() / n) * self._entropy(y[left_mask]) + \
                           (right_mask.sum() / n) * self._entropy(y[right_mask])
                gain = parent_impurity - weighted
            else:
                weighted = (left_mask.sum() / n) * self._gini_index(y[left_mask]) + \
                           (right_mask.sum() / n) * self._gini_index(y[right_mask])
                gain = parent_impurity - weighted

            if gain > best_gain:
                best_gain = gain
                best_thresh = t

        return best_gain, best_thresh

    def _information_gain(self, X, y, feature_idx, parent_entropy):
        n = len(y)
        weighted = 0.0
        for val in np.unique(X[:, feature_idx]):
            subset = y[X[:, feature_idx] == val]
            weighted += (len(subset) / n) * self._entropy(subset)
        return parent_entropy - weighted

    def _gini_gain(self, X, y, feature_idx, parent_gini):
        n = len(y)
        weighted = 0.0
        for val in np.unique(X[:, feature_idx]):
            subset = y[X[:, feature_idx] == val]
            weighted += (len(subset) / n) * self._gini_index(subset)
        return parent_gini - weighted

    def _entropy(self, y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return -np.sum(p * np.log2(p + 1e-12))

    def _gini_index(self, y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1.0 - np.sum(p ** 2)

    def _predict_single(self, x, node):
        if node.is_leaf:
            return node.value

        if node.is_numerical:
            val = float(x[node.feature])
            if val <= node.threshold:
                return self._predict_single(x, node.left)
            else:
                return self._predict_single(x, node.right)
        else:
            val = x[node.feature]
            if val in node.children:
                return self._predict_single(x, node.children[val])
            else:
                return self._fallback(node)

    def _fallback(self, node):
        if node.is_leaf:
            return node.value
        for child in node.children.values():
            result = self._fallback(child)
            if result is not None:
                return result
        if node.left is not None:
            return self._fallback(node.left)
        if node.right is not None:
            return self._fallback(node.right)
        return None

    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]