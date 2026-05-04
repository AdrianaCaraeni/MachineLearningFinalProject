import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, 'algorithms')

from knn import KNN
from decision_tree import DecisionTree
from random_forest import RandomForest
from naive_baiyes import MultinomialNaiveBayes
from cross_validation import make_stratified_folds, compute_accuracy, compute_f1


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
        self.std  = X.std(axis=0)
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


class NaiveBayesBase:
    def __init__(self, alpha=1.0, n_bins=6, numerical_cols=None):
        self.alpha = alpha
        self.n_bins = n_bins
        self.numerical_cols = numerical_cols or []
        self.bin_edges = {}

    def _tokenize(self, X):
        docs = []
        for row in X:
            tokens = []
            for i, val in enumerate(row):
                if i in self.bin_edges:
                    bin_idx = int(np.digitize(val, self.bin_edges[i]))
                    tokens.append(f"f{i}_b{bin_idx}")
                else:
                    tokens.append(f"f{i}_v{int(val)}")
            docs.append(tokens)
        return docs

    def fit(self, X, y):
        self.bin_edges = {}
        for col in self.numerical_cols:
            self.bin_edges[col] = np.percentile(
                X[:, col], np.linspace(0, 100, self.n_bins + 1)[1:-1])
        docs = self._tokenize(X)
        vocab = set(t for doc in docs for t in doc)
        pos_docs = [docs[i] for i in range(len(y)) if y[i] == 1]
        neg_docs = [docs[i] for i in range(len(y)) if y[i] == 0]
        self.model = MultinomialNaiveBayes(alpha=self.alpha)
        self.model.train(pos_docs, neg_docs, vocab)

    def predict(self, X):
        docs = self._tokenize(X)
        return np.array([1 if self.model.classify(d) == 'positive' else 0 for d in docs])


# ── Ensemble ──────────────────────────────────────────────────────────────────

class HeterogeneousEnsemble:
    """
    Each base learner is trained on its own bootstrap sample of the training data.
    Final prediction is determined by majority vote across all base learners.
    """
    def __init__(self, base_learner_configs, seed=42):
        self.configs = base_learner_configs  # list of (ClassType, kwargs, bootstrap_seed)
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
        # majority vote per instance
        return np.array([np.bincount(all_preds[:, i].astype(int)).argmax()
                         for i in range(X.shape[0])])


# ── Cross-validation for ensemble ─────────────────────────────────────────────

def cv_ensemble(configs, X, y, k=10, seed=42):
    Y = y.reshape(-1, 1)
    folds = make_stratified_folds(y, k=k, seed=seed)
    accs, f1s = [], []
    for held_out in range(k):
        test_idx  = folds[held_out]
        train_idx = np.concatenate([folds[f] for f in range(k) if f != held_out])
        Xtr, ytr = X[train_idx], y[train_idx]
        Xte, yte = X[test_idx],  y[test_idx]
        ensemble = HeterogeneousEnsemble(configs, seed=seed)
        ensemble.fit(Xtr, ytr)
        preds = ensemble.predict(Xte).reshape(-1, 1)
        yte2d = yte.reshape(-1, 1)
        accs.append(compute_accuracy(yte2d, preds))
        f1s.append(compute_f1(yte2d, preds))
        print(f"  fold {held_out+1}/10  Acc={accs[-1]:.4f}  F1={f1s[-1]:.4f}")
    return np.mean(accs), np.mean(f1s)


# ── Data loaders ──────────────────────────────────────────────────────────────

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


def load_titanic(filepath):
    X, y = [], []
    sex_map = {'male': 0, 'female': 1}
    with open(filepath, newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row:
                continue
            y.append(int(row[0]))
            X.append([
                float(row[1]),
                float(sex_map[row[2].strip()]),
                float(row[3]) if row[3].strip() else 29.7,
                float(row[4]),
                float(row[5]),
                float(row[6]) if row[6].strip() else 14.45,
            ])
    print(f"Titanic: loaded {len(X)} instances.")
    return np.array(X), np.array(y), [2, 3, 4, 5], [0, 1]


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    datasets = {
        'Rice':    load_rice('rice.csv'),
        'Credit':  load_credit('credit_approval.csv'),
        'Titanic': load_titanic('titanic.csv'),
    }

    # Best hyperparams from individual runs for each dataset
    best_params = {
        'Rice': {
            'knn_k': 21, 'dt_depth': 3, 'rf_trees': 10, 'rf_depth': 5,
            'nb_alpha': 1.0, 'nb_bins': 6,
        },
        'Credit': {
            'knn_k': 7,  'dt_depth': 1, 'rf_trees': 10, 'rf_depth': 3,
            'nb_alpha': 2.0, 'nb_bins': 6,
        },
        'Titanic': {
            'knn_k': 7,  'dt_depth': 5, 'rf_trees': 15, 'rf_depth': 5,
            'nb_alpha': 1.0, 'nb_bins': 6,
        },
    }

    ensemble_results = {}

    for name, (X, y, num_cols, cat_cols) in datasets.items():
        p = best_params[name]
        print(f"\n{'='*50}")
        print(f"Ensemble CV — {name} Dataset")
        print(f"{'='*50}")

        configs = [
            (KNNBase,          {'k': p['knn_k']}),
            (DecisionTreeBase, {'max_depth': p['dt_depth'],  'numerical_cols': num_cols}),
            (RandomForestBase, {'n_trees':   p['rf_trees'],  'max_depth': p['rf_depth'],
                                'numerical_cols': num_cols}),
            (NaiveBayesBase,   {'alpha': p['nb_alpha'], 'n_bins': p['nb_bins'],
                                'numerical_cols': num_cols}),
        ]

        acc, f1 = cv_ensemble(configs, X, y, k=10)
        ensemble_results[name] = (acc, f1)
        print(f"  >> {name} Ensemble  Acc={acc:.4f}  F1={f1:.4f}")

    # Summary bar chart
    names  = list(ensemble_results.keys())
    accs   = [ensemble_results[n][0] for n in names]
    f1s    = [ensemble_results[n][1] for n in names]
    x      = np.arange(len(names))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width/2, accs, width, label='Accuracy',  color='steelblue')
    ax.bar(x + width/2, f1s,  width, label='F1-Score',  color='darkorange')
    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(i - width/2, a + 0.005, f'{a:.4f}', ha='center', fontsize=8)
        ax.text(i + width/2, f + 0.005, f'{f:.4f}', ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel('Score')
    ax.set_title('Heterogeneous Ensemble Performance across Datasets')
    ax.set_ylim(0.5, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('fig_ensemble_summary.png', dpi=150)
    plt.close()
    print("\nSaved fig_ensemble_summary.png")

    print("\nFINAL ENSEMBLE RESULTS")
    print(f"{'Dataset':<12} {'Accuracy':>10} {'F1-Score':>10}")
    for name in names:
        a, f = ensemble_results[name]
        print(f"{name:<12} {a:>10.4f} {f:>10.4f}")