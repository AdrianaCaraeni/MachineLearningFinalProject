#multiclass cross-validation helper funcs 
#macro-F1 = unweighted avg of per-class F1 and we kept this separate from cross_validation.py so the binary code there stays as-is since 2 different people are working on this project, I didnt wanna step on anybody's toes 




import numpy as np


def macro_f1(y_true, y_pred):
    
    classes = np.unique(np.concatenate([y_true, y_pred]))
    f1s = []
    i = 0
    
    while i < len(classes):
        c  = classes[i]
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))

        if tp + fp == 0:
            prec = 0.0
        else:
            prec = tp / (tp + fp)

        if tp + fn == 0:
            rec = 0.0
        else:
            rec = tp / (tp + fn)

        if prec + rec == 0:
            f1s.append(0.0)
            
        else:
            f1s.append(2.0 * prec * rec / (prec + rec))
        i = i + 1

    return float(np.mean(f1s))






def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))



def stratified_folds(labels, k=10, seed=42):
    
    
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]

    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        i = 0
        while i < len(idx):
            folds[i % k].append(idx[i])
            i = i + 1

    return [np.array(f) for f in folds]



def cross_validate_multiclass(ModelClass, model_kwargs, X, y, k=10, seed=42):
    
    folds  = stratified_folds(y, k=k, seed=seed)
    accs   = []
    f1s    = []
    held = 0
    
    
    while held < k:
        test_idx  = folds[held]
        train_idx = np.concatenate([folds[f] for f in range(k) if f != held])

        model = ModelClass(**model_kwargs)
        model.fit(X[train_idx], y[train_idx].reshape(-1, 1))
        preds_2d = model.predict(X[test_idx])
        preds = preds_2d.ravel() if preds_2d.ndim > 1 else preds_2d

        accs.append(accuracy(y[test_idx], preds))
        f1s .append(macro_f1(y[test_idx], preds))
        held = held + 1



    return {
        'mean_accuracy': float(np.mean(accs)),
        'mean_f1':       float(np.mean(f1s)),
    }