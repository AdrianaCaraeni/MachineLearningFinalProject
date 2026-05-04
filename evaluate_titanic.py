import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, 'algorithms')

from decision_tree import DecisionTree
from random_forest import RandomForest
from naive_baiyes import MultinomialNaiveBayes
from cross_validation import cross_validate

# Dataset columns: label, attr1_cat (pclass), attr2_cat (sex), attr3_num (age),
#                  attr4_num (sibsp), attr5_num (parch), attr6_num (fare)
# Categorical feature indices (in feature array, 0-indexed): 0 (pclass), 1 (sex)
# Numerical feature indices: 2 (age), 3 (sibsp), 4 (parch), 5 (fare)
NUMERICAL_COLS   = [2, 3, 4, 5]
CATEGORICAL_COLS = [0, 1]


def load_titanic(filepath):
    X, y = [], []
    sex_map = {'male': 0, 'female': 1}

    with open(filepath, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if not row:
                continue
            label     = int(row[0])
            pclass    = float(row[1])
            sex       = float(sex_map[row[2].strip()])
            age       = float(row[3]) if row[3].strip() else 29.7  # median fallback
            sibsp     = float(row[4])
            parch     = float(row[5])
            fare      = float(row[6]) if row[6].strip() else 14.45
            X.append([pclass, sex, age, sibsp, parch, fare])
            y.append(label)

    print(f"Loaded {len(X)} instances.")
    return np.array(X), np.array(y)


class DecisionTreeWrapper:
    def __init__(self, max_depth=None, numerical_cols=None):
        self.max_depth = max_depth
        self.numerical_cols = numerical_cols or []

    def fit(self, X, Y):
        self.model = DecisionTree(criterion='information_gain', max_depth=self.max_depth)
        self.model.fit(X, Y[:, 0], numerical_features=self.numerical_cols)

    def predict(self, X):
        return self.model.predict(X).reshape(-1, 1)


class RandomForestWrapper:
    def __init__(self, n_trees=10, max_depth=None, numerical_cols=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.numerical_cols = numerical_cols or []

    def fit(self, X, Y):
        self.model = RandomForest(n_trees=self.n_trees, max_depth=self.max_depth, random_state=42)
        self.model.fit(X, Y[:, 0], numerical_features=self.numerical_cols)

    def predict(self, X):
        return self.model.predict(X).reshape(-1, 1)


class NaiveBayesWrapper:
    def __init__(self, alpha=1.0, n_bins=4, numerical_cols=None):
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

    def fit(self, X, Y):
        y = Y[:, 0].astype(int)
        self.bin_edges = {}
        for col in self.numerical_cols:
            vals = X[:, col]
            self.bin_edges[col] = np.percentile(vals, np.linspace(0, 100, self.n_bins + 1)[1:-1])
        docs = self._tokenize(X)
        vocab = set(t for doc in docs for t in doc)
        pos_docs = [docs[i] for i in range(len(y)) if y[i] == 1]
        neg_docs = [docs[i] for i in range(len(y)) if y[i] == 0]
        self.model = MultinomialNaiveBayes(alpha=self.alpha)
        self.model.train(pos_docs, neg_docs, vocab)

    def predict(self, X):
        docs = self._tokenize(X)
        preds = [1 if self.model.classify(doc) == 'positive' else 0 for doc in docs]
        return np.array(preds).reshape(-1, 1)


def run_search(name, ModelClass, configs, X, Y, y):
    print(f"\n{name} Hyperparameter Search")
    results = {}
    for kwargs in configs:
        result = cross_validate(ModelClass, kwargs, X, Y, y, k=10)
        results[str(kwargs)] = (kwargs, result['mean_accuracy'], result['mean_f1'])
        print(f"  {kwargs}  Acc={result['mean_accuracy']:.4f}  F1={result['mean_f1']:.4f}")
    best_key = max(results, key=lambda k: results[k][2])
    best = results[best_key]
    print(f"  Best: {best[0]}  Acc={best[1]:.4f}  F1={best[2]:.4f}")
    return results, best


if __name__ == '__main__':
    X, y = load_titanic('titanic.csv')
    Y = y.reshape(-1, 1)

    dt_configs = [{'max_depth': d, 'numerical_cols': NUMERICAL_COLS}
                  for d in [1, 3, 5, 7, 10, 15, 20, None]]

    rf_configs = [{'n_trees': n, 'max_depth': d, 'numerical_cols': NUMERICAL_COLS}
                  for n, d in [(5, 3), (5, 5), (10, 3), (10, 5), (15, 5), (20, 5)]]

    nb_configs = [{'alpha': a, 'n_bins': b, 'numerical_cols': NUMERICAL_COLS}
                  for a, b in [(0.1, 4), (0.5, 4), (1.0, 4), (1.0, 6), (2.0, 4), (2.0, 6)]]

    dt_results, best_dt = run_search("Decision Tree", DecisionTreeWrapper, dt_configs, X, Y, y)
    rf_results, best_rf = run_search("Random Forest", RandomForestWrapper, rf_configs, X, Y, y)
    nb_results, best_nb = run_search("Naive Bayes",   NaiveBayesWrapper,   nb_configs, X, Y, y)

    # Figure 1: Decision Tree accuracy and F1 vs max_depth
    depths      = [c['max_depth'] for c in dt_configs]
    depth_labels= [str(d) if d is not None else 'unlimited' for d in depths]
    accs        = [dt_results[str(c)][1] for c in dt_configs]
    f1s         = [dt_results[str(c)][2] for c in dt_configs]
    best_d_lbl  = str(best_dt[0]['max_depth']) if best_dt[0]['max_depth'] is not None else 'unlimited'
    x_pos       = list(range(len(depth_labels)))

    plt.figure(figsize=(7, 4))
    plt.plot(x_pos, accs, 'o-', label='Accuracy', color='steelblue')
    plt.plot(x_pos, f1s,  's--', label='F1-Score', color='darkorange')
    plt.axvline(depth_labels.index(best_d_lbl), color='gray', linestyle=':', label=f'Best depth={best_d_lbl}')
    plt.xticks(x_pos, depth_labels, rotation=15)
    plt.xlabel('max_depth'); plt.ylabel('Score')
    plt.title('Decision Tree Performance vs max_depth — Titanic')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig('fig1_titanic_dt_vs_depth.png', dpi=150); plt.close()
    print("Saved fig1_titanic_dt_vs_depth.png")

    # Figure 2: Random Forest F1 vs n_trees grouped by max_depth
    rf_depths = sorted(set(c['max_depth'] for c in rf_configs if c['max_depth'] is not None))
    colors = ['steelblue', 'darkorange']
    plt.figure(figsize=(7, 4))
    for color, d in zip(colors, rf_depths):
        pts = [(c['n_trees'], rf_results[str(c)][2])
               for c in rf_configs if c['max_depth'] == d]
        pts.sort()
        ns, f1s_d = zip(*pts)
        plt.plot(ns, f1s_d, 'o-', label=f'max_depth={d}', color=color)
    plt.xlabel('Number of Trees'); plt.ylabel('F1-Score')
    plt.title('Random Forest: n_trees vs F1-Score — Titanic')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig('fig2_titanic_rf_vs_ntrees.png', dpi=150); plt.close()
    print("Saved fig2_titanic_rf_vs_ntrees.png")

    # Figure 3: Naive Bayes F1 vs alpha grouped by n_bins
    nb_bins = sorted(set(c['n_bins'] for c in nb_configs))
    plt.figure(figsize=(7, 4))
    for color, b in zip(colors, nb_bins):
        pts = [(c['alpha'], nb_results[str(c)][2])
               for c in nb_configs if c['n_bins'] == b]
        pts.sort()
        alphas_b, f1s_b = zip(*pts)
        plt.plot(alphas_b, f1s_b, 'o-', label=f'n_bins={b}', color=color)
    plt.xscale('log')
    plt.xlabel('alpha (Laplace smoothing)'); plt.ylabel('F1-Score')
    plt.title('Naive Bayes: alpha vs F1-Score — Titanic')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig('fig3_titanic_nb_vs_alpha.png', dpi=150); plt.close()
    print("Saved fig3_titanic_nb_vs_alpha.png")

    print("\nFINAL RESULTS — Titanic Dataset")
    print(f"  Decision Tree Acc={best_dt[1]:.4f}  F1={best_dt[2]:.4f}  params={best_dt[0]}")
    print(f"  Random Forest Acc={best_rf[1]:.4f}  F1={best_rf[2]:.4f}  params={best_rf[0]}")
    print(f"  Naive Bayes   Acc={best_nb[1]:.4f}  F1={best_nb[2]:.4f}  params={best_nb[0]}")