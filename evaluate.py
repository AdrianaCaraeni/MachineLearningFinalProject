import numpy as np

def stratified_kfold_indices(y, k=10, seed=42):
    np.random.seed(seed)
    classes = np.unique(y)
    folds = [[] for _ in range(k)]
    for c in classes:
        idx = np.where(y == c)[0]
        np.random.shuffle(idx)
        splits = np.array_split(idx, k)
        for i, s in enumerate(splits):
            folds[i].extend(s)
    return folds

def cross_validate(model_fn, X, y, k=10):
    folds = stratified_kfold_indices(y, k)
    accuracies, f1s = [], []
    for i in range(k):
        test_idx = np.array(folds[i])
        train_idx = np.array([x for j, f in enumerate(folds) if j != i for x in f])
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model = model_fn()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracies.append(accuracy(y_test, preds))
        f1s.append(f1_score(y_test, preds))
    return np.mean(accuracies), np.mean(f1s)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def f1_score(y_true, y_pred):
    # Macro F1 for multiclass, binary F1 for 2 classes
    classes = np.unique(y_true)
    f1s = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1s.append(2 * prec * rec / (prec + rec + 1e-8))
    return np.mean(f1s)