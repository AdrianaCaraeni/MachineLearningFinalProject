
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, 'algorithms')

from gaussian_naive_bayes import GaussianNaiveBayes
from cross_validation       import cross_validate
from cv_multiclass          import cross_validate_multiclass







#EC4: Gaussian Naive Bayes which is a variant of MNB that handles continuous features and multiclass
#been ran on all four datasets


#loading datsets below 


def load_rice(path):
    X, y, class_map = [], [], {}
    with open(path, newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            feats = [float(v) for v in row[:-1]]
            label = row[-1].strip()
            if label not in class_map:
                class_map[label] = len(class_map)
            X.append(feats)
            y.append(class_map[label])
    print(f"Rice -> {len(X)} instances, {len(X[0])} numerical features.")
    return np.array(X), np.array(y), 'binary'



def load_parkinsons(path):
    X, y = [], []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        label_idx = len(header) - 1
        for row in reader:
            if not row:
                continue
            feats = []
            j = 0
            while j < len(row):
                if j != label_idx:
                    feats.append(float(row[j]))
                j = j + 1
            X.append(feats)
            y.append(int(row[label_idx]))
    print(f"Parkinson's -> {len(X)} instances, {len(X[0])} numerical features.")
    return np.array(X), np.array(y), 'binary'



def load_credit(path):
    X, y = [], []
    numerical_cols, categorical_cols = [], []

    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        i = 0
        while i < len(header) - 1:
            if 'num' in header[i]:
                numerical_cols.append(i)
            else:
                categorical_cols.append(i)
            i = i + 1
        rows = [r for r in reader if r]

    cat_encoders = {}
    for col in categorical_cols:
        vals = list({row[col] for row in rows})
        cat_encoders[col] = {v: idx for idx, v in enumerate(vals)}

    for row in rows:
        feats = []
        i = 0
        while i < len(row) - 1:
            if i in numerical_cols:
                feats.append(float(row[i]))
            else:
                feats.append(float(cat_encoders[i][row[i]]))
            i  = i + 1
        X.append(feats)
        y.append(int(row[-1]))

    print(f"Credit -> {len(X)} instances ({len(numerical_cols)} num + {len(categorical_cols)} cat encoded).")
    return np.array(X), np.array(y), 'binary'



def load_digits_data():
    from sklearn import datasets #again loading only 
    digits = datasets.load_digits(return_X_y=True)
    X = np.asarray(digits[0], dtype=float)
    y = np.asarray(digits[1])
    print(f"Digits -> {len(X)} images, {X.shape[1]} pixel features, {len(np.unique(y))} classes.")
    return X, y, 'multiclass'



#wrappers gaussian nb below 


class GNBWrapper:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, Y):
        self.model = GaussianNaiveBayes(var_smoothing=self.var_smoothing)
        self.model.fit(X, Y[:, 0])

    def predict(self, X):
        return self.model.predict(X).reshape(-1, 1)



def run_gnb_search(name, X, y, problem_type, smoothings):
    print(f"\n{name}")
    results = []

    i = 0
    while i < len(smoothings):
        vs = smoothings[i]
        kwargs = {'var_smoothing': vs}

        if problem_type == 'binary':
            r = cross_validate(GNBWrapper, kwargs, X, y.reshape(-1, 1), y, k=10)
        else:
            r = cross_validate_multiclass(GNBWrapper, kwargs, X, y, k=10)

        acc = r['mean_accuracy']
        f1  = r['mean_f1']
        results.append((vs, acc, f1))
        print(f"  var_smoothing={vs:>8.0e}  Acc={acc:.4f}  F1={f1:.4f}")
        i = i +1


    best = max(results, key=lambda t: t[2])
    print(f"  Best -> var_smoothing={best[0]:.0e}  Acc={best[1]:.4f}  F1={best[2]:.4f}")
    return results, best




if __name__ == '__main__':
    smoothings = [1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1.0]


    datasets = {
        'Rice':        load_rice('rice.csv'),
        "Parkinson's": load_parkinsons('parkinsons.csv'),
        'Credit':      load_credit('credit_approval.csv'),
        'Digits':      load_digits_data(),
    }


    all_results = {}     # name -> (sweep_results, best)

    for name, (X, y, ptype) in datasets.items():
        sweep, best = run_gnb_search(name, X, y, ptype, smoothings)
        all_results[name] = (sweep, best)


    #per-dataset figures
    # F1 vs var_smoothing

    for name in datasets.keys():
        sweep, best = all_results[name]
        vss  = [t[0] for t in sweep]
        accs = [t[1] for t in sweep]
        f1s  = [t[2] for t in sweep]

        plt.figure(figsize=(7, 4))
        plt.plot(vss, accs, 'o-',  label='Accuracy', color='steelblue')
        plt.plot(vss, f1s,  's--', label='F1-Score', color='darkorange')
        plt.axvline(best[0], color='gray', linestyle=':',
                    label=f'Best vs={best[0]:.0e}')
        plt.xscale('log')
        plt.xlabel('var_smoothing  (log scale)')
        plt.ylabel('Score')
        plt.title(f'Gaussian NB: var_smoothing vs Performance -- {name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        safe = name.lower().replace(" ", "_").replace("'", "")
        fname = f'fig_q4_gnb_{safe}.png'
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved {fname}")


    #summary 

        names = list(all_results.keys())
    accs  = [all_results[n][1][1] for n in names]
    f1s   = [all_results[n][1][2] for n in names]
    x     = np.arange(len(names))
    w     = 0.35
 
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w/2, accs, w, label='Accuracy', color='steelblue')
    ax.bar(x + w/2, f1s,  w, label='F1-Score', color='darkorange')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.05)
    ax.set_title('EC4: Gaussian Naive Bayes Across All Four Datasets (best var_smoothing per dataset)')
    ax.legend()
    plt.savefig('fig_q4_gnb_summary.png', dpi=150)
    plt.close()
 
 
    print("\n EC4 RESULTS (Gaussian NB)")
    print(f"{'Dataset':<14} {'var_smooth':>11} {'Accuracy':>10} {'F1':>10}")
    for n in names:
        sweep, best = all_results[n]
        print(f"{n:<14} {best[0]:>11.0e} {best[1]:>10.4f} {best[2]:>10.4f}")