import numpy as np


def compute_accuracy(y_true, y_pred):
    return float(np.mean(np.all(y_true == y_pred, axis=1)))


def compute_f1(y_true, y_pred):
    true_col = y_true[:, 0]
    pred_col = y_pred[:, 0]

    tp = np.sum((pred_col == 1) & (true_col == 1))
    fp = np.sum((pred_col == 1) & (true_col == 0))
    fn = np.sum((pred_col == 0) & (true_col == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def make_stratified_folds(labels, k=5, seed=42):
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]

    for class_label in np.unique(labels):
        indices = np.where(labels == class_label)[0]
        rng.shuffle(indices)
        for i, idx in enumerate(indices):
            folds[i % k].append(idx)

    return [np.array(fold) for fold in folds]


def cross_validate(ModelClass, model_kwargs, X, Y, labels, k=5, seed=42):
    folds = make_stratified_folds(labels, k=k, seed=seed)

    accuracies = []
    f1_scores = []

    for held_out in range(k):
        test_idx = folds[held_out]
        train_idx = np.concatenate([folds[f] for f in range(k) if f != held_out])

        model = ModelClass(**model_kwargs)
        model.fit(X[train_idx], Y[train_idx])
        preds = model.predict(X[test_idx])

        accuracies.append(compute_accuracy(Y[test_idx], preds))
        f1_scores.append(compute_f1(Y[test_idx], preds))

    return {
        "mean_accuracy": float(np.mean(accuracies)),
        "mean_f1": float(np.mean(f1_scores)),
    }