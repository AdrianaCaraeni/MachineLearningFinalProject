import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, 'algorithms')

from knn import KNN, normalize_features
from decision_tree import DecisionTree
from random_forest import RandomForest
from cross_validation import cross_validate


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
    print(f"Loaded {len(X)} instances. Classes: {class_map}")
    return np.array(X), np.array(y)


# Wrappers so each algorithm matches the interface cross_validate expects:
# fit(X, Y) and predict(X) where Y is 2D

class KNNWrapper:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, Y):
        y = Y[:, 0]
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1
        self.model = KNN(k=self.k)
        self.model.fit((X - self.mean) / self.std, y)

    def predict(self, X):
        return self.model.predict((X - self.mean) / self.std).reshape(-1, 1)


class DecisionTreeWrapper:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, Y):
        self.model = DecisionTree(criterion='information_gain', max_depth=self.max_depth)
        self.model.fit(X, Y[:, 0], numerical_features=list(range(X.shape[1])))

    def predict(self, X):
        return self.model.predict(X).reshape(-1, 1)


class RandomForestWrapper:
    def __init__(self, n_trees=10, max_depth=None):
        self.n_trees = n_trees
        self.max_depth = max_depth

    def fit(self, X, Y):
        self.model = RandomForest(n_trees=self.n_trees, max_depth=self.max_depth, random_state=42)
        self.model.fit(X, Y[:, 0], numerical_features=list(range(X.shape[1])))

    def predict(self, X):
        return self.model.predict(X).reshape(-1, 1)


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
    X, y = load_rice('rice.csv')
    Y = y.reshape(-1, 1)

    knn_configs = [{'k': k} for k in [1, 3, 5, 7, 11, 15, 21, 31]]
    dt_configs  = [{'max_depth': d} for d in [1, 3, 5, 7, 10, 15, 20, None]]
    rf_configs  = [{'n_trees': n, 'max_depth': d} for n, d in
                   [(10, 5), (10, 10), (20, 5), (20, 10), (50, 10), (50, None), (100, 10), (100, None)]]

    knn_results, best_knn = run_search("KNN",           KNNWrapper,          knn_configs, X, Y, y)
    dt_results,  best_dt  = run_search("Decision Tree", DecisionTreeWrapper, dt_configs,  X, Y, y)
    rf_results,  best_rf  = run_search("Random Forest", RandomForestWrapper, rf_configs,  X, Y, y)

    # Figure 1: KNN accuracy and F1 vs k
    ks      = [c['k'] for c in knn_configs]
    accs    = [knn_results[str(c)][1] for c in knn_configs]
    f1s     = [knn_results[str(c)][2] for c in knn_configs]
    best_k  = best_knn[0]['k']

    plt.figure(figsize=(7, 4))
    plt.plot(ks, accs, 'o-', label='Accuracy', color='steelblue')
    plt.plot(ks, f1s,  's--', label='F1-Score', color='darkorange')
    plt.axvline(best_k, color='gray', linestyle=':', label=f'Best k={best_k}')
    plt.xlabel('k'); plt.ylabel('Score')
    plt.title('KNN Performance vs k — Rice Grains')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig('fig1_knn_vs_k.png', dpi=150); plt.close()

    # Figure 2: Decision Tree accuracy and F1 vs max_depth
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
    plt.title('Decision Tree Performance vs max_depth — Rice Grains')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig('fig2_dt_vs_depth.png', dpi=150); plt.close()

    # Figure 3: Random Forest accuracy and F1 vs n_trees (at best max_depth)
    best_md = best_rf[0]['max_depth']
    sweep   = [5, 10, 20, 30, 50, 75, 100]
    print("\nRandom Forest n_trees sweep...")
    sweep_accs, sweep_f1s = [], []
    for nt in sweep:
        r = cross_validate(RandomForestWrapper, {'n_trees': nt, 'max_depth': best_md}, X, Y, y, k=10)
        sweep_accs.append(r['mean_accuracy']); sweep_f1s.append(r['mean_f1'])
        print(f"  n_trees={nt}  Acc={r['mean_accuracy']:.4f}  F1={r['mean_f1']:.4f}")

    md_label = str(best_md) if best_md is not None else 'unlimited'
    plt.figure(figsize=(7, 4))
    plt.plot(sweep, sweep_accs, 'o-', label='Accuracy', color='steelblue')
    plt.plot(sweep, sweep_f1s,  's--', label='F1-Score', color='darkorange')
    plt.xlabel('Number of Trees'); plt.ylabel('Score')
    plt.title(f'Random Forest: n_trees vs Performance (max_depth={md_label}) — Rice Grains')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig('fig3_rf_vs_ntrees.png', dpi=150); plt.close()

    print("\nFINAL RESULTS — Rice Grains Dataset")
    print(f"  KNN           Acc={best_knn[1]:.4f}  F1={best_knn[2]:.4f}  params={best_knn[0]}")
    print(f"  Decision Tree Acc={best_dt[1]:.4f}  F1={best_dt[2]:.4f}  params={best_dt[0]}")
    print(f"  Random Forest Acc={best_rf[1]:.4f}  F1={best_rf[2]:.4f}  params={best_rf[0]}")