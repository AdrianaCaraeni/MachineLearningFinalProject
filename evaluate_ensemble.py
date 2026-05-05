"""
EC2: Heterogeneous ensemble — each base learner trains on its own bootstrap sample
     of the training fold; final prediction is majority vote across learners.
     (See also per-dataset comparison of each base learner vs. the full ensemble.)

EC3: 10-fold stratified CV of the ensemble on the four official datasets:
     Digits, Parkinson's, Rice, and Credit.
"""

import sys
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, 'algorithms')

from knn import KNN
from decision_tree import DecisionTree
from random_forest import RandomForest
from gaussian_naive_bayes import GaussianNaiveBayes
from cross_validation import make_stratified_folds, compute_accuracy, compute_f1
from cv_multiclass import macro_f1


# ── Bootstrap sampling ────────────────────────────────────────────────────────

def bootstrap_sample(X, Y, seed):
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    idx = rng.choice(n, size=n, replace=True)
    return X[idx], Y[idx]


# ── Base learner wrappers ─────────────────────────────────────────────────────

class KNNBase:
    def __init__(self, k=21):
        self.k = k

    def fit(self, X, y):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1
        self.model = KNN(k=self.k)
        self.model.fit((X - self.mean) / self.std, y)

    def predict(self, X):
        return self.model.predict((X - self.mean) / self.std)


class DecisionTreeBase:
    def __init__(self, max_depth=5, numerical_cols=None):
        self.max_depth = max_depth
        self.numerical_cols = numerical_cols or []

    def fit(self, X, y):
        self.model = DecisionTree(criterion='information_gain', max_depth=self.max_depth)
        self.model.fit(X, y, numerical_features=self.numerical_cols)

    def predict(self, X):
        return self.model.predict(X)


class RandomForestBase:
    def __init__(self, n_trees=10, max_depth=5, numerical_cols=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.numerical_cols = numerical_cols or []

    def fit(self, X, y):
        self.model = RandomForest(n_trees=self.n_trees, max_depth=self.max_depth, random_state=42)
        self.model.fit(X, y, numerical_features=self.numerical_cols)

    def predict(self, X):
        return self.model.predict(X)


class GaussianNBBase:
    """Multiclass-capable Gaussian Naive Bayes (continuous features)."""

    def __init__(self, var_smoothing=1e-9, numerical_cols=None):
        self.var_smoothing = var_smoothing
        self.numerical_cols = numerical_cols

    def fit(self, X, y):
        self.model = GaussianNaiveBayes(var_smoothing=self.var_smoothing)
        self.model.fit(np.asarray(X, dtype=float), y)

    def predict(self, X):
        return self.model.predict(np.asarray(X, dtype=float))


# ── Ensemble ──────────────────────────────────────────────────────────────────

class HeterogeneousEnsemble:
    """
    Each base learner is trained on its own bootstrap sample of the training data.
    Final prediction is determined by majority vote across all base learners.
    """
    def __init__(self, base_learner_configs, seed=42):
        self.configs = base_learner_configs
        self.seed = seed
        self.learners = []

    def fit(self, X, y):
        self.learners = []
        rng = np.random.RandomState(self.seed)
        for ClassType, kwargs in self.configs:
            boot_seed = rng.randint(0, 2**31)
            Y2d = y.reshape(-1, 1)
            X_boot, Y_boot = bootstrap_sample(X, Y2d, boot_seed)
            y_boot = Y_boot[:, 0]
            learner = ClassType(**kwargs)
            learner.fit(X_boot, y_boot)
            self.learners.append(learner)

    def predict(self, X):
        all_preds = np.array([learner.predict(X) for learner in self.learners])
        return np.array([np.bincount(all_preds[:, i].astype(int)).argmax()
                         for i in range(X.shape[0])])


# ── Metrics ───────────────────────────────────────────────────────────────────

def f1_for_labels(y_true_2d, y_pred_2d):
    y_true = y_true_2d.reshape(-1)
    if len(np.unique(y_true)) <= 2:
        return compute_f1(y_true_2d, y_pred_2d)
    y_pred = y_pred_2d.reshape(-1)
    return macro_f1(y_true, y_pred)


# ── Cross-validation ──────────────────────────────────────────────────────────

def cv_ensemble(configs, X, y, k=10, seed=42):
    Y = y.reshape(-1, 1)
    folds = make_stratified_folds(y, k=k, seed=seed)
    accs, f1s = [], []
    for held_out in range(k):
        test_idx = folds[held_out]
        train_idx = np.concatenate([folds[f] for f in range(k) if f != held_out])
        Xtr, ytr = X[train_idx], y[train_idx]
        Xte, yte = X[test_idx], y[test_idx]
        ensemble = HeterogeneousEnsemble(configs, seed=seed)
        ensemble.fit(Xtr, ytr)
        preds = ensemble.predict(Xte).reshape(-1, 1)
        yte2d = yte.reshape(-1, 1)
        accs.append(compute_accuracy(yte2d, preds))
        f1s.append(f1_for_labels(yte2d, preds))
        print(f"  fold {held_out+1}/{k}  Acc={accs[-1]:.4f}  F1={f1s[-1]:.4f}", flush=True)
    return np.mean(accs), np.mean(f1s)


def cv_individual_base(ClassType, kwargs, X, y, k=10, seed=42):
    """One base learner on full training fold each split (no bootstrap). EC2 comparison."""
    Y = y.reshape(-1, 1)
    folds = make_stratified_folds(y, k=k, seed=seed)
    accs, f1s = [], []
    for held_out in range(k):
        test_idx = folds[held_out]
        train_idx = np.concatenate([folds[f] for f in range(k) if f != held_out])
        Xtr, ytr = X[train_idx], y[train_idx]
        Xte, yte = X[test_idx], y[test_idx]
        model = ClassType(**kwargs)
        model.fit(Xtr, ytr)
        preds = model.predict(Xte).reshape(-1, 1)
        yte2d = yte.reshape(-1, 1)
        accs.append(compute_accuracy(yte2d, preds))
        f1s.append(f1_for_labels(yte2d, preds))
    return np.mean(accs), np.mean(f1s)


def build_configs(p, num_cols):
    return [
        (KNNBase, {'k': p['knn_k']}),
        (DecisionTreeBase, {'max_depth': p['dt_depth'], 'numerical_cols': num_cols}),
        (RandomForestBase, {'n_trees': p['rf_trees'], 'max_depth': p['rf_depth'],
                            'numerical_cols': num_cols}),
        (GaussianNBBase, {'var_smoothing': p['gnb_var_smoothing'], 'numerical_cols': num_cols}),
    ]


# ── Data loaders ────────────────────────────────────────────────────────────────

def load_rice(filepath):
    X, y, class_map = [], [], {}
    with open(filepath, newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            features = [float(v) for v in row[:-1]]
            label = row[-1].strip()
            if label not in class_map:
                class_map[label] = len(class_map)
            X.append(features)
            y.append(class_map[label])
    print(f"Rice: loaded {len(X)} instances.")
    return np.array(X), np.array(y), list(range(7)), []


def load_credit(filepath):
    X, y = [], []
    numerical_cols, categorical_cols = [], []
    with open(filepath, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        for i, col in enumerate(header[:-1]):
            (numerical_cols if 'num' in col else categorical_cols).append(i)
        rows = [row for row in reader if row]
    cat_encoders = {}
    for col in categorical_cols:
        vals = list({row[col] for row in rows})
        cat_encoders[col] = {v: i for i, v in enumerate(vals)}
    for row in rows:
        features = []
        for i, val in enumerate(row[:-1]):
            features.append(float(val) if i in numerical_cols
                            else float(cat_encoders[i][val]))
        X.append(features)
        y.append(int(row[-1]))
    print(f"Credit: loaded {len(X)} instances.")
    return np.array(X), np.array(y), numerical_cols, categorical_cols


def load_digits_data():
    from sklearn import datasets
    digits = datasets.load_digits(return_X_y=True)
    X = np.asarray(digits[0], dtype=float)
    y = np.asarray(digits[1])
    print(f"Digits: loaded {len(X)} instances, {len(np.unique(y))} classes.")
    return np.array(X), np.array(y), list(range(X.shape[1])), []


def _resolve_parkinsons_path():
    for p in ('data/parkinsons.csv', 'parkinsons.csv'):
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "Parkinson's CSV not found. Expected data/parkinsons.csv or parkinsons.csv in the project root."
    )


def load_parkinsons(filepath=None):
    path = filepath or _resolve_parkinsons_path()
    X, y = [], []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        label_idx = len(header) - 1
        for row in reader:
            if not row:
                continue
            features = []
            j = 0
            while j < len(row):
                if j != label_idx:
                    features.append(float(row[j]))
                j = j + 1
            X.append(features)
            y.append(int(row[label_idx]))
    X = np.array(X)
    y = np.array(y)
    print(f"Parkinson's: loaded {len(X)} instances from {path}.")
    return X, y, list(range(X.shape[1])), []


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    datasets = {
        'Digits': load_digits_data(),
        "Parkinson's": load_parkinsons(),
        'Rice': load_rice('rice.csv'),
        'Credit': load_credit('credit_approval.csv'),
    }

    best_params = {
        'Digits': {
            'knn_k': 5, 'dt_depth': 15, 'rf_trees': 20, 'rf_depth': 10,
            'gnb_var_smoothing': 1e-9,
        },
        "Parkinson's": {
            'knn_k': 7, 'dt_depth': 10, 'rf_trees': 50, 'rf_depth': 10,
            'gnb_var_smoothing': 1e-9,
        },
        'Rice': {
            'knn_k': 21, 'dt_depth': 3, 'rf_trees': 10, 'rf_depth': 5,
            'gnb_var_smoothing': 1e-9,
        },
        'Credit': {
            'knn_k': 7, 'dt_depth': 1, 'rf_trees': 10, 'rf_depth': 3,
            'gnb_var_smoothing': 1e-9,
        },
    }

    ensemble_results = {}
    ec2_individual_results = {}

    for name, (X, y, num_cols, _cat_cols) in datasets.items():
        p = best_params[name]
        print(f"\n{'='*50}")
        print(f"EC3 — Ensemble CV — {name}")
        print(f"{'='*50}")

        configs = build_configs(p, num_cols)

        acc, f1 = cv_ensemble(configs, X, y, k=10)
        ensemble_results[name] = (acc, f1)
        print(f"  >> Ensemble  Acc={acc:.4f}  F1={f1:.4f}")

        print(f"\nEC2 — Individual base learners (same folds, no bootstrap) — {name}")
        labels = ['KNN', 'DecisionTree', 'RandomForest', 'GaussianNB']
        ec2_individual_results[name] = {}
        for label, (ClassType, kwargs) in zip(labels, configs):
            ia, if1 = cv_individual_base(ClassType, kwargs, X, y, k=10)
            ec2_individual_results[name][label] = (ia, if1)
            print(f"  {label:<14} Acc={ia:.4f}  F1={if1:.4f}")

    names = list(ensemble_results.keys())
    accs = [ensemble_results[n][0] for n in names]
    f1s = [ensemble_results[n][1] for n in names]
    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width/2, accs, width, label='Accuracy', color='steelblue')
    ax.bar(x + width/2, f1s, width, label='F1-Score', color='darkorange')
    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(i - width/2, a + 0.01, f'{a:.4f}', ha='center', fontsize=8)
        ax.text(i + width/2, f + 0.01, f'{f:.4f}', ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel('Score')
    ax.set_title('EC3: Heterogeneous Ensemble (four official datasets)')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('fig_ensemble_summary.png', dpi=150)
    plt.close()
    print("\nSaved fig_ensemble_summary.png")

    print("\n" + "=" * 50)
    print("EC3 — FINAL ENSEMBLE RESULTS (per dataset)")
    print("=" * 50)
    print(f"{'Dataset':<14} {'Accuracy':>10} {'F1':>10}")
    for name in names:
        a, f = ensemble_results[name]
        print(f"{name:<14} {a:>10.4f} {f:>10.4f}")

