import sys
import csv
import numpy as np
import matplotlib.pyplot as plt



sys.path.insert(0, 'algorithms')
from knn import KNN
from decision_tree import DecisionTree
from random_forest import RandomForest
from cross_validation import cross_validate


#This file loads parkinsons.csv, 22 numerical voice measurements per row and label is the Diagnosis column (last col): 0 healthy, 1 parkinson's
def load_parkinsons(path):
    X, y = [], []

    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)

        # diagnosis is last column
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

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    print(f"Loaded {len(X)} instances. {n_pos} parkinson's / {n_neg} healthy.")
    print(f"Features per instance: {X.shape[1]}")

    return X, y



# wrapper funcs which match the fit(X, Y2D) / predict -> 2D interface used in cross_validate


class KNNWrapper:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, Y):
        self.mu  = X.mean(axis=0)
        self.sd  = X.std(axis=0)
        self.sd[self.sd == 0] = 1.0      

        self.model = KNN(k=self.k)
        self.model.fit((X - self.mu) / self.sd, Y[:, 0])

    def predict(self, X):
        Xn = (X - self.mu) / self.sd
        return self.model.predict(Xn).reshape(-1, 1)



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
        self.n_trees   = n_trees
        self.max_depth = max_depth

    def fit(self, X, Y):
        self.model = RandomForest(n_trees=self.n_trees, max_depth=self.max_depth, random_state=42)
        self.model.fit(X, Y[:, 0], numerical_features=list(range(X.shape[1])))

    def predict(self, X):
        return self.model.predict(X).reshape(-1, 1)



def run_search(name, ModelClass, configs, X, Y, y):
    print(f"\n{name} -- hyperparameter search")
    results = {}

    i = 0
    while i < len(configs):
        kwargs = configs[i]
        r = cross_validate(ModelClass, kwargs, X, Y, y, k=10)
        results[str(kwargs)] = (kwargs, r['mean_accuracy'], r['mean_f1'])
        print(f"  {kwargs}  Acc={r['mean_accuracy']:.4f}  F1={r['mean_f1']:.4f}")
        i = i + 1
    best_key = max(results, key=lambda k: results[k][2])
    best     = results[best_key]
    
    print(f"  Best -> {best[0]}  Acc={best[1]:.4f}  F1={best[2]:.4f}")
    
    return results, best




if __name__ == '__main__':

    X, y = load_parkinsons('parkinsons.csv')
    Y    = y.reshape(-1, 1)
    knn_configs = [{'k': k} for k in [1, 3, 5, 7, 11, 15, 21, 31]]

    dt_configs  = [{'max_depth': d} for d in [1, 3, 5, 7, 10, 15, 20, None]]

    rf_configs  = [{'n_trees': n, 'max_depth': d} for n, d in [
        (10, 5),  (10, 10),
        (20, 5),  (20, 10),
        (50, 5),  (50, 10),
        (50, None), (100, 10),
    ]]


    knn_results, best_knn = run_search("KNN",           KNNWrapper,          knn_configs, X, Y, y)
    dt_results,  best_dt  = run_search("Decision Tree", DecisionTreeWrapper, dt_configs,  X, Y, y)
    rf_results,  best_rf  = run_search("Random Forest", RandomForestWrapper, rf_configs,  X, Y, y)



#fig 1 KNN vs k 

    ks      = [c['k'] for c in knn_configs]
    knn_acc = [knn_results[str(c)][1] for c in knn_configs]
    knn_f1  = [knn_results[str(c)][2] for c in knn_configs]
    best_k  = best_knn[0]['k']

    plt.figure(figsize=(7, 4))
    plt.plot(ks, knn_acc, 'o-',  label='Accuracy', color='steelblue')
    plt.plot(ks, knn_f1,  's--', label='F1-Score', color='darkorange')
    plt.axvline(best_k, color='gray', linestyle=':', label=f'Best k={best_k}')
    plt.xlabel('k')
    plt.ylabel('Score')
    plt.title("KNN Performance vs k -- Parkinson's")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig1_parkinsons_knn_vs_k.png', dpi=150)
    plt.close()


#fig 2: decision tree vs max_graph 

    depths       = [c['max_depth'] for c in dt_configs]
    depth_labels = [str(d) if d is not None else 'unlimited' for d in depths]
    dt_acc       = [dt_results[str(c)][1] for c in dt_configs]
    dt_f1        = [dt_results[str(c)][2] for c in dt_configs]
    best_d_lbl   = str(best_dt[0]['max_depth']) if best_dt[0]['max_depth'] is not None else 'unlimited'
    x_pos        = list(range(len(depth_labels)))

    plt.figure(figsize=(7, 4))
    plt.plot(x_pos, dt_acc, 'o-',  label='Accuracy', color='steelblue')
    plt.plot(x_pos, dt_f1,  's--', label='F1-Score', color='darkorange')
    plt.axvline(depth_labels.index(best_d_lbl), color='gray', linestyle=':',
                label=f'Best depth={best_d_lbl}')
    plt.xticks(x_pos, depth_labels, rotation=15)
    plt.xlabel('max_depth')
    plt.ylabel('Score')
    plt.title("Decision Tree Performance vs max_depth -- Parkinson's")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig2_parkinsons_dt_vs_depth.png', dpi=150)
    plt.close()


#fig 3: random forest vs n_trees for each max_depth
    rf_depths = sorted(set(c['max_depth'] for c in rf_configs if c['max_depth'] is not None))
    colors    = ['steelblue', 'darkorange', 'green']

    plt.figure(figsize=(7, 4))
    j = 0
    while j < len(rf_depths):
        d = rf_depths[j]
        pts = []
        for c in rf_configs:
            if c['max_depth'] == d:
                pts.append((c['n_trees'], rf_results[str(c)][2]))
        pts.sort()
        ns, f1s_d = zip(*pts)
        plt.plot(ns, f1s_d, 'o-', label=f'max_depth={d}', color=colors[j % len(colors)])
        j += 1

    plt.xlabel('Number of Trees')
    plt.ylabel('F1-Score')
    plt.title("Random Forest: n_trees vs F1-Score -- Parkinson's")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig3_parkinsons_rf_vs_ntrees.png', dpi=150)
    plt.close()


    print("\nFINAL RESULTS og Parkinson's Dataset")
    print(f"  KNN           Acc={best_knn[1]:.4f}  F1={best_knn[2]:.4f}  params={best_knn[0]}")
    print(f"  Decision Tree Acc={best_dt[1]:.4f}  F1={best_dt[2]:.4f}  params={best_dt[0]}")
    print(f"  Random Forest Acc={best_rf[1]:.4f}  F1={best_rf[2]:.4f}  params={best_rf[0]}")