import matplotlib.pyplot as plt

# All results from 10-fold stratified cross-validation

knn_results = {
    1:  (0.8890, 0.9023),
    3:  (0.9079, 0.9196),
    5:  (0.9155, 0.9263),
    7:  (0.9210, 0.9312),
    11: (0.9244, 0.9343),
    15: (0.9257, 0.9354),
    21: (0.9273, 0.9367),
    31: (0.9268, 0.9364),
}

dt_results = {
    1:           (0.9255, 0.9346),
    3:           (0.9255, 0.9347),
    5:           (0.9202, 0.9305),
    7:           (0.9165, 0.9272),
    10:          (0.9068, 0.9188),
    15:          (0.8942, 0.9075),
    20:          (0.8898, 0.9033),
    'unlimited': (0.8879, 0.9018),
}

rf_results = {
    (10, 5):        (0.9276, 0.9370),
    (10, 10):       (0.9215, 0.9318),
    (20, 5):        (0.9268, 0.9364),
    (20, 10):       (0.9241, 0.9340),
    (50, 10):       (0.9262, 0.9360),
    (50, 'unlimited'): (0.9223, 0.9326),
}

# Figure 1: KNN accuracy and F1 vs k
ks   = sorted(knn_results.keys())
accs = [knn_results[k][0] for k in ks]
f1s  = [knn_results[k][1] for k in ks]

plt.figure(figsize=(7, 4))
plt.plot(ks, accs, 'o-', label='Accuracy', color='steelblue')
plt.plot(ks, f1s,  's--', label='F1-Score', color='darkorange')
plt.axvline(21, color='gray', linestyle=':', label='Best k=21')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Score')
plt.title('KNN Performance vs k — Rice Grains Dataset')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig1_knn_vs_k.png', dpi=150)
plt.close()
print("Saved fig1_knn_vs_k.png")

# Figure 2: Decision Tree accuracy and F1 vs max_depth
depth_keys   = [1, 3, 5, 7, 10, 15, 20, 'unlimited']
depth_labels = [str(d) for d in depth_keys]
accs = [dt_results[d][0] for d in depth_keys]
f1s  = [dt_results[d][1] for d in depth_keys]
x_pos = list(range(len(depth_labels)))

plt.figure(figsize=(7, 4))
plt.plot(x_pos, accs, 'o-', label='Accuracy', color='steelblue')
plt.plot(x_pos, f1s,  's--', label='F1-Score', color='darkorange')
plt.axvline(depth_labels.index('3'), color='gray', linestyle=':', label='Best depth=3')
plt.xticks(x_pos, depth_labels)
plt.xlabel('max_depth')
plt.ylabel('Score')
plt.title('Decision Tree Performance vs max_depth — Rice Grains Dataset')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig2_dt_vs_depth.png', dpi=150)
plt.close()
print("Saved fig2_dt_vs_depth.png")

# Figure 3: Random Forest accuracy and F1 vs n_trees (grouping by max_depth)
configs_d5  = [(n, d) for n, d in rf_results if d == 5]
configs_d10 = [(n, d) for n, d in rf_results if d == 10]

nt_d5  = sorted([n for n, d in configs_d5])
nt_d10 = sorted([n for n, d in configs_d10])
f1_d5  = [rf_results[(n, 5)][1]  for n in nt_d5]
f1_d10 = [rf_results[(n, 10)][1] for n in nt_d10]

plt.figure(figsize=(7, 4))
plt.plot(nt_d5,  f1_d5,  'o-', label='max_depth=5',  color='steelblue')
plt.plot(nt_d10, f1_d10, 's--', label='max_depth=10', color='darkorange')
plt.xlabel('Number of Trees')
plt.ylabel('F1-Score')
plt.title('Random Forest: n_trees vs F1-Score — Rice Grains Dataset')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig3_rf_vs_ntrees.png', dpi=150)
plt.close()
print("Saved fig3_rf_vs_ntrees.png")

print("\nFINAL RESULTS — Rice Grains Dataset")
print(f"  KNN (k=21)                        Acc=0.9273  F1=0.9367")
print(f"  Decision Tree (max_depth=3)        Acc=0.9255  F1=0.9347")
print(f"  Random Forest (n=10, max_depth=5)  Acc=0.9276  F1=0.9370")