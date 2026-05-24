import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from scipy.stats import pearsonr, spearmanr

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
        "Qwen1.5 metrics + layers + proxy",
        "metric data",
        "metrics"
    )
)

# ============================================================
# TARGETS
# ============================================================

ablation_drops = [
    -70.7, -70.1,  -9.9, -58.0, -37.0, -13.0,
    -66.9,  -6.7, -23.5, -39.6, -41.5, -11.6,
    -61.4, -46.5, -19.5,  -1.2,   1.7, -11.9,
      0.1,   1.2,  -4.5,   1.3,   1.1,  -1.1
]

# ============================================================
# FEATURES
# ============================================================

print("Сборка датасета из CSV-файлов...")

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

num_layers = 24
num_features = len(features_2d) + 1

X_numpy = np.zeros((num_layers, num_features))

feature_names = []

for col_idx, (filename, feat_name) in enumerate(features_2d):

    feature_names.append(feat_name)

    filepath = os.path.join(
        benchmark_dir,
        filename
    )

    df = pd.read_csv(filepath, index_col=0)

    matrix = df.values

    for i in range(num_layers - 1):
        X_numpy[i, col_idx] = matrix[i, i + 1]

    X_numpy[23, col_idx] = matrix[22, 23]

# ============================================================
# ROUTER ENTROPY
# ============================================================

feature_names.append("Router_Entropy")

df_ent = pd.read_csv(
    os.path.join(
        benchmark_dir,
        "metric_09_Router_Entropy.csv"
    ),
    index_col=0
)

X_numpy[:, 8] = df_ent["Avg_Router_Entropy"].values

print(
    f"Успешно загружено "
    f"{num_layers} слоев "
    f"и {num_features} признаков.\n"
)

# ============================================================
# DATA
# ============================================================

X_all = X_numpy

y_all = np.array(ablation_drops)

# ============================================================
# LOOCV
# ============================================================

print("Starting Leave-One-Out Cross Validation...\n")

loocv_predictions = []

for test_idx in range(num_layers):

    # ========================================================
    # SPLIT
    # ========================================================

    X_train = np.delete(X_all, test_idx, axis=0)
    y_train = np.delete(y_all, test_idx)

    X_test = X_all[test_idx:test_idx + 1]
    y_test = y_all[test_idx]

    # ========================================================
    # TORCH
    # ========================================================

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_train = torch.tensor(
        y_train,
        dtype=torch.float32
    ).view(-1, 1)

    X_test = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    # ========================================================
    # NORMALIZATION
    # IMPORTANT:
    # FIT ONLY ON TRAIN
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

    model = nn.Linear(num_features, 1)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.01,
        weight_decay=0.1
    )

    # ========================================================
    # TRAIN
    # ========================================================

    epochs = 3000

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
# LOOCV RESULTS
# ============================================================

loocv_predictions = np.array(loocv_predictions)

loocv_pearson, _ = pearsonr(
    loocv_predictions,
    y_all
)

loocv_spearman, _ = spearmanr(
    loocv_predictions,
    y_all
)

print("\n" + "=" * 70)

print("LOOCV RESULTS")

print("=" * 70)

print(f"\nPEARSON  : {loocv_pearson:.4f}")
print(f"SPEARMAN : {loocv_spearman:.4f}")

print("=" * 70)

# ============================================================
# TRAIN FINAL MODEL ON ALL DATA
# ============================================================

print("\nTraining final model on ALL layers...\n")

X = torch.tensor(
    X_all,
    dtype=torch.float32
)

y = torch.tensor(
    y_all,
    dtype=torch.float32
).view(-1, 1)

# normalization
X_mean = X.mean(dim=0)
X_std = X.std(dim=0) + 1e-8

X_norm = (X - X_mean) / X_std

y_mean = y.mean()
y_std = y.std() + 1e-8

y_norm = (y - y_mean) / y_std

# ============================================================
# FINAL MODEL
# ============================================================

model = nn.Linear(num_features, 1)

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.01,
    weight_decay=0.1
)

epochs = 3000

for epoch in range(epochs):

    optimizer.zero_grad()

    predictions = model(X_norm)

    loss = criterion(
        predictions,
        y_norm
    )

    loss.backward()

    optimizer.step()

# ============================================================
# FULL TRAIN RESULTS
# ============================================================

model.eval()

with torch.no_grad():

    final_preds_norm = model(X_norm)

    final_preds = (
        final_preds_norm * y_std
    ) + y_mean

preds_flat = final_preds.numpy().flatten()

train_pearson, _ = pearsonr(
    preds_flat,
    y_all
)

train_spearman, _ = spearmanr(
    preds_flat,
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

bias = model.bias.data.item()

print("\nFEATURE IMPORTANCE")
print("-" * 70)

feature_importance = sorted(
    zip(feature_names, weights),
    key=lambda x: abs(x[1]),
    reverse=True
)

for name, weight in feature_importance:

    sign = "+" if weight > 0 else "-"

    print(
        f"{name:20s} "
        f"{sign} {abs(weight):.4f}"
    )

# ============================================================
# PROXY FORMULA
# ============================================================

formula = "Proxy = "

for name, weight in feature_importance:

    sign = "+" if weight > 0 else "-"

    formula += (
        f"\n        "
        f"{sign} {abs(weight):.4f} * ({name})"
    )

formula += (
    f"\n        "
    f"{'+' if bias > 0 else '-'} "
    f"{abs(bias):.4f} (Bias)"
)

print("\n" + "=" * 70)

print("PROXY METRIC FORMULA")

print("=" * 70)

print(formula)