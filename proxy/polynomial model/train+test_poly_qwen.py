import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import spearmanr
from sklearn.preprocessing import PolynomialFeatures
import random

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

def load_dataset(dataset_path, num_layers, drops):
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
    x_numpy = np.zeros((num_layers, len(features_2d) + 1))
    feature_names = []
    for col_idx, (filename, feat_name) in enumerate(features_2d):
        feature_names.append(feat_name)
        df = pd.read_csv(os.path.join(dataset_path, filename), index_col=0)
        matrix = df.values
        for i in range(num_layers - 1):
            x_numpy[i, col_idx] = matrix[i, i + 1]
        x_numpy[num_layers - 1, col_idx] = matrix[num_layers - 2, num_layers - 1]
    feature_names.append("Router_Entropy")
    df_ent = pd.read_csv(
        os.path.join(dataset_path, "metric_09_Router_Entropy.csv"),
        index_col=0
    )
    x_numpy[:, 8] = df_ent["Avg_Router_Entropy"].values
    
    poly = PolynomialFeatures(degree=2, include_bias=False)
    x_poly = poly.fit_transform(x_numpy)
    poly_feature_names = poly.get_feature_names_out(feature_names)
    
    x_tensor = torch.tensor(x_poly, dtype=torch.float32)
    y_tensor = torch.tensor(drops, dtype=torch.float32).view(-1, 1)
    return x_tensor, y_tensor, poly_feature_names

datasets_info = {
    "Qwen + Commonsense": {
        "path": os.path.join(BASE_DIR, "commonsense", "Qwen1.5 metrics + layers", "metric data", "metrics"),
        "layers": 24,
        "drops": [
            -70.7, -70.1, -9.9, -58.0, -37.0, -13.0, -66.9, -6.7, -23.5, -39.6, -41.5, -11.6,
            -61.4, -46.5, -19.5, -1.2, 1.7, -11.9, 0.1, 1.2, -4.5, 1.3, 1.1, -1.1
        ]
    },
    "Phi-Tiny + Commonsense": {
        "path": os.path.join(BASE_DIR, "commonsense", "phi-tiny metrics + layers", "metric data", "metrics"),
        "layers": 32,
        "drops": [
            -65.5, -10.4, -7.2, -42.8, -7.2, -6.8, -7.5, -4.6, -6.1, -7.5, -6.6, -6.3,
            -4.4, -7.9, -13.1, -1.2, -2.3, -3.4, -1.3, -2.4, -0.6, -0.4, -0.7, -2.9,
            -0.2, -0.7, -0.6, 0.6, -0.7, 0.0, -0.3, -1.0
        ]
    },
    "Qwen + SIQA": {
        "path": os.path.join(BASE_DIR, "siqa", "Qwen1.5 metrics + layers", "metric data", "metrics"),
        "layers": 24,
        "drops": [
            -67.0, -65.2, -14.2, -31.0, -19.0, -6.2, -53.6, -7.0, -11.4, -20.4, -4.0, -7.6,
            -18.8, -24.6, -9.4, -4.4, -2.4, -0.8, -1.0, 0.6, -1.4, -0.2, -1.0, 0.4
        ]
    },
    "Phi-Tiny + SIQA": {
        "path": os.path.join(BASE_DIR, "siqa", "phi-tiny metrics + layers", "metric data", "metrics"),
        "layers": 32,
        "drops": [
            -56.6, -5.2, -0.8, -24.0, -4.4, -7.0, -1.8, -2.2, -2.8, -6.4, -10.6, -2.4,
            -0.6, -4.8, -7.0, -2.2, -1.8, -0.2, -0.8, -0.6, 1.0, -0.6, -2.2, -1.6,
            -0.6, -0.6, 0.0, 0.8, 0.8, -2.0, -0.6, 0.4
        ]
    }
}

x_train, y_train, feat_names = load_dataset(
    datasets_info["Qwen + Commonsense"]["path"],
    datasets_info["Qwen + Commonsense"]["layers"],
    datasets_info["Qwen + Commonsense"]["drops"]
)

x_train_mean = x_train.mean(dim=0)
x_train_std = x_train.std(dim=0) + 1e-8
x_train_norm = (x_train - x_train_mean) / x_train_std

y_train_mean = y_train.mean()
y_train_std = y_train.std() + 1e-8
y_train_norm = (y_train - y_train_mean) / y_train_std

model = nn.Linear(len(feat_names), 1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=0.5)

for _ in range(5000):
    optimizer.zero_grad()
    loss = criterion(model(x_train_norm), y_train_norm)
    loss.backward()
    optimizer.step()

model.eval()

weights = model.weight.data.numpy().flatten()
bias = model.bias.data.item()

feat_imp = sorted(
    zip(feat_names, weights),
    key=lambda item: abs(item[1]),
    reverse=True
)

formula = "Proxy ="
for name, weight in feat_imp:
    sign = "+" if weight > 0 else "-"
    formula += f"\n        {sign} {abs(weight):.4f} * ({name})"
formula += f"\n        {'+' if bias > 0 else '-'} {abs(bias):.4f} (Bias)\n"

print(formula)

for name, info in datasets_info.items():
    x_test, y_test, _ = load_dataset(info["path"], info["layers"], info["drops"])
    x_test_mean = x_test.mean(dim=0)
    x_test_std = x_test.std(dim=0) + 1e-8
    x_test_norm = (x_test - x_test_mean) / x_test_std
    with torch.no_grad():
        preds = (model(x_test_norm) * y_train_std) + y_train_mean
    spearman, _ = spearmanr(preds.numpy().flatten(), y_test.numpy().flatten())
    print(f"Спирмен на {name}: {spearman:.4f}")

'''
Proxy =
        + 0.2525 * (CKA L1_Dist)
        - 0.2474 * (Cosine_Dist CKA)
        - 0.1643 * (Cosine_Dist Pearson)
        + 0.1235 * (L1_Dist Pearson)
        + 0.1175 * (CKA L_Inf)
        + 0.1138 * (L_Inf Pearson)
        + 0.1014 * (Pearson^2)
        + 0.0942 * (CKA^2)
        + 0.0915 * (CKA Pearson)
        + 0.0773 * (L_Inf)
        + 0.0771 * (L_Inf Router_Entropy)
        + 0.0754 * (Pearson Router_Entropy)
        - 0.0752 * (Cosine_Dist)
        + 0.0751 * (Pearson)
        - 0.0751 * (Cosine_Dist Router_Entropy)
        + 0.0686 * (Router_Entropy^2)
        + 0.0684 * (Router_Entropy)
        + 0.0674 * (L1_Dist)
        + 0.0673 * (L1_Dist Router_Entropy)
        + 0.0643 * (CKA Router_Entropy)
        + 0.0640 * (CKA)
        + 0.0475 * (L_Inf^2)
        + 0.0416 * (MSE CKA)
        + 0.0285 * (MSE Pearson)
        - 0.0260 * (Res_Contrib CKA)
        + 0.0235 * (Cosine_Dist L1_Dist)
        + 0.0225 * (MSE)
        + 0.0225 * (MSE Router_Entropy)
        + 0.0225 * (L_Inf Var_Ratio)
        + 0.0214 * (Cosine_Dist L_Inf)
        + 0.0213 * (L1_Dist L_Inf)
        + 0.0211 * (Res_Contrib L1_Dist)
        + 0.0206 * (L1_Dist^2)
        + 0.0206 * (Res_Contrib L_Inf)
        + 0.0204 * (L1_Dist Var_Ratio)
        - 0.0202 * (Res_Contrib Pearson)
        + 0.0186 * (MSE Var_Ratio)
        + 0.0183 * (MSE L_Inf)
        + 0.0177 * (MSE Cosine_Dist)
        + 0.0173 * (MSE Res_Contrib)
        + 0.0171 * (MSE L1_Dist)
        + 0.0168 * (MSE^2)
        + 0.0165 * (CKA Var_Ratio)
        - 0.0152 * (Cosine_Dist^2)
        + 0.0119 * (Res_Contrib^2)
        - 0.0092 * (Var_Ratio^2)
        - 0.0059 * (Res_Contrib)
        - 0.0059 * (Res_Contrib Router_Entropy)
        - 0.0054 * (Cosine_Dist Var_Ratio)
        + 0.0052 * (Res_Contrib Var_Ratio)
        - 0.0052 * (Var_Ratio)
        - 0.0052 * (Var_Ratio Router_Entropy)
        - 0.0047 * (Var_Ratio Pearson)
        + 0.0038 * (Cosine_Dist Res_Contrib)
        - 0.0000 (Bias)

Спирмен на Qwen + Commonsense: 0.7765
Спирмен на Phi-Tiny + Commonsense: 0.7200
Спирмен на Qwen + SIQA: 0.8863
Спирмен на Phi-Tiny + SIQA: 0.6216
'''