import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import PolynomialFeatures

# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_DIR = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..")
)

benchmark_dir = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "phi-tiny layers + metrics + proxy",
        "metric data",
        "metrics"
    )
)

# ============================================================
# TARGETS
# ============================================================

ablation_drops = [
    -65.5,  # 0
    -10.4,  # 1
    -7.2,   # 2
    -42.8,  # 3
    -7.2,   # 4
    -6.8,   # 5
    -7.5,   # 6
    -4.6,   # 7
    -6.1,   # 8
    -7.5,   # 9
    -6.6,   # 10
    -6.3,   # 11
    -4.4,   # 12
    -7.9,   # 13
    -13.1,  # 14
    -1.2,   # 15
    -2.3,   # 16
    -3.4,   # 17
    -1.3,   # 18
    -2.4,   # 19
    -0.6,   # 20
    -0.4,   # 21
    -0.7,   # 22
    -2.9,   # 23
    -0.2,   # 24
    -0.7,   # 25
    -0.6,   # 26
    +0.6,   # 27
    -0.7,   # 28
     0.0,   # 29
    -0.3,   # 30
    -1.0    # 31
]

# ============================================================
# FEATURES
# ============================================================

features_2d = [
    ("metric_01_MSE.csv", "MSE"),
    ("metric_02_Cosine_Distance.csv", "Cosine_Dist"),
    ("metric_03_Residual_Contribution.csv", "Res_Contrib"),
    ("metric_04_CKA.csv", "CKA"),
    ("metric_05_L1_Distance.csv", "L1_Dist"),
    ("metric_06_L_Infinity.csv", "L_Inf"),
    ("metric_07_Variance_Ratio.csv", "Var_Ratio"),
    ("metric_08_Pearson_Correlation.csv", "Pearson")
]

num_layers = 32
num_features = len(features_2d) + 1

X_numpy = np.zeros((num_layers, num_features))

feature_names = []

print("Loading features...")

for col_idx, (filename, feat_name) in enumerate(features_2d):

    feature_names.append(feat_name)

    filepath = os.path.join(benchmark_dir, filename)

    df = pd.read_csv(filepath, index_col=0)

    matrix = df.values

    for i in range(num_layers - 1):
        X_numpy[i, col_idx] = matrix[i, i + 1]

    X_numpy[23, col_idx] = matrix[22, 23]

# Router entropy
feature_names.append("Router_Entropy")

df_ent = pd.read_csv(
    os.path.join(
        benchmark_dir,
        "metric_09_Router_Entropy.csv"
    ),
    index_col=0
)

X_numpy[:, 8] = df_ent["Avg_Router_Entropy"].values

# ============================================================
# POLYNOMIAL FEATURES
# ============================================================

poly = PolynomialFeatures(
    degree=2,
    include_bias=False
)

X_poly = poly.fit_transform(X_numpy)

poly_feature_names = poly.get_feature_names_out(
    feature_names
)

print(f"\nOriginal features   : {X_numpy.shape[1]}")
print(f"Polynomial features : {X_poly.shape[1]}")

# ============================================================
# DATA
# ============================================================

X_all = X_poly

y_all = np.array(ablation_drops)

# ============================================================
# LOOCV
# ============================================================

print("\nStarting Leave-One-Out Cross Validation...\n")

loocv_predictions = []

for test_idx in range(num_layers):

    # ========================================================
    # SPLIT
    # ========================================================

    X_train = np.delete(X_all, test_idx, axis=0)
    y_train = np.delete(y_all, test_idx)

    X_test = X_all[test_idx:test_idx+1]
    y_test = y_all[test_idx]

    # ========================================================
    # TORCH TENSORS
    # ========================================================

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    X_test = torch.tensor(X_test, dtype=torch.float32)

    # ========================================================
    # NORMALIZATION
    # IMPORTANT:
    # fit ONLY on train
    # ========================================================

    X_mean = X_train.mean(dim=0)
    X_std = X_train.std(dim=0) + 1e-8

    X_train_norm = (X_train - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std

    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-8

    y_train_norm = (y_train - y_mean) / y_std

    # ========================================================
    # MODEL
    # ========================================================

    model = nn.Linear(
        X_train_norm.shape[1],
        1
    )

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.005,
        weight_decay=0.5
    )

    # ========================================================
    # TRAIN
    # ========================================================

    epochs = 5000

    for epoch in range(epochs):

        optimizer.zero_grad()

        predictions = model(X_train_norm)

        loss = criterion(
            predictions,
            y_train_norm
        )

        loss.backward()

        optimizer.step()

    # ========================================================
    # PREDICT
    # ========================================================

    model.eval()

    with torch.no_grad():

        pred_norm = model(X_test_norm)

        pred = (pred_norm * y_std) + y_mean

    loocv_predictions.append(
        pred.item()
    )

    print(
        f"Layer {test_idx:2d} | "
        f"True: {y_test:7.2f} | "
        f"Pred: {pred.item():7.2f}"
    )

# ============================================================
# FINAL METRICS
# ============================================================

loocv_predictions = np.array(loocv_predictions)

pearson_corr, _ = pearsonr(
    loocv_predictions,
    y_all
)

spearman_corr, _ = spearmanr(
    loocv_predictions,
    y_all
)

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)

print("LOOCV RESULTS")

print("=" * 70)

print(f"\nPEARSON  : {pearson_corr:.4f}")
print(f"SPEARMAN : {spearman_corr:.4f}")

print("\n" + "=" * 70)

# ============================================================
# OPTIONAL:
# TRAIN ON FULL DATA FOR INTERPRETATION
# ============================================================

print("\nTraining final model on ALL layers...\n")

X = torch.tensor(X_all, dtype=torch.float32)

y = torch.tensor(
    y_all,
    dtype=torch.float32
).view(-1, 1)

X_mean = X.mean(dim=0)
X_std = X.std(dim=0) + 1e-8

X_norm = (X - X_mean) / X_std

y_mean = y.mean()
y_std = y.std() + 1e-8

y_norm = (y - y_mean) / y_std

model = nn.Linear(X_norm.shape[1], 1)

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.005,
    weight_decay=0.5
)

epochs = 5000

for epoch in range(epochs):

    optimizer.zero_grad()

    predictions = model(X_norm)

    loss = criterion(predictions, y_norm)

    loss.backward()

    optimizer.step()

# ============================================================
# TRAIN PERFORMANCE (FULL DATA)
# ============================================================

model.eval()

with torch.no_grad():

    final_preds_norm = model(X_norm)

    final_preds = (
        final_preds_norm * y_std
    ) + y_mean

final_preds = final_preds.numpy().flatten()

train_pearson, _ = pearsonr(
    final_preds,
    y_all
)

train_spearman, _ = spearmanr(
    final_preds,
    y_all
)

print("\n" + "=" * 70)

print("FULL TRAIN RESULTS")

print("=" * 70)

print(f"\nPEARSON  : {train_pearson:.4f}")
print(f"SPEARMAN : {train_spearman:.4f}")

print("=" * 70)
# ============================================================
# FEATURE IMPORTANCE
# ============================================================

weights = model.weight.data.numpy().flatten()

importance = sorted(
    zip(poly_feature_names, weights),
    key=lambda x: abs(x[1]),
    reverse=True
)

print("\nFEATURES")
print("-" * 70)

for name, weight in importance:

    sign = "+" if weight > 0 else "-"

    print(
        f"{name:40s} "
        f"{sign} {abs(weight):.4f}"
    )