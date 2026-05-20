import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr, spearmanr

benchmark_dir = r""

ablation_drops = [
    -70.7, -70.1,  -9.9, -58.0, -37.0, -13.0, -66.9,  -6.7, 
    -23.5, -39.6, -41.5, -11.6, -61.4, -46.5, -19.5,  -1.2, 
      1.7, -11.9,   0.1,   1.2,  -4.5,   1.3,   1.1,  -1.1
]

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
    filepath = os.path.join(benchmark_dir, filename)
    df = pd.read_csv(filepath, index_col=0)
    matrix = df.values
    
    for i in range(num_layers - 1):
        X_numpy[i, col_idx] = matrix[i, i + 1]
    X_numpy[23, col_idx] = matrix[22, 23]

feature_names.append("Router_Entropy")
df_ent = pd.read_csv(os.path.join(benchmark_dir, "metric_09_Router_Entropy.csv"), index_col=0)
X_numpy[:, 8] = df_ent['Avg_Router_Entropy'].values

print(f"Успешно загружено {num_layers} слоев и {num_features} признаков.\n")

X = torch.tensor(X_numpy, dtype=torch.float32)
y = torch.tensor(ablation_drops, dtype=torch.float32).view(-1, 1)

X_mean = X.mean(dim=0)
X_std = X.std(dim=0) + 1e-8
X_norm = (X - X_mean) / X_std

y_mean = y.mean()
y_std = y.std() + 1e-8
y_norm = (y - y_mean) / y_std

model = nn.Linear(num_features, 1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.1)

epochs = 3000
print("Начинаем обучение линейной регрессии (Градиентный спуск)...")

for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = model(X_norm)
    loss = criterion(predictions, y_norm)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 500 == 0:
        print(f"Эпоха {epoch+1:4d}/{epochs} | Ошибка (MSE Loss): {loss.item():.4f}")

model.eval()
with torch.no_grad():
    final_preds_norm = model(X_norm)
    final_preds = (final_preds_norm * y_std) + y_mean

preds_flat = final_preds.numpy().flatten()
y_flat = y.numpy().flatten()

pearson_corr, _ = pearsonr(preds_flat, y_flat)
spearman_corr, _ = spearmanr(preds_flat, y_flat)

print("\n" + "="*60)
print(f"ПИРСОН (Линейная зависимость): {pearson_corr:.4f}")
print(f"СПИРМЕН (Ранговая нелинейная зависимость): {spearman_corr:.4f}")
print("="*60)

weights = model.weight.data.numpy().flatten()
bias = model.bias.data.item()

print("\nВЕСА МЕТРИК (Влияние на падение точности слоя):")
print("-" * 60)

feature_importance = sorted(zip(feature_names, weights), key=lambda x: abs(x[1]), reverse=True)

formula = "Просадка = "
for name, weight in feature_importance:
    sign = "+" if weight > 0 else "-"
    print(f"{name:>15}: {sign} {abs(weight):.4f}")
    formula += f"\n           {sign} {abs(weight):.4f} * ({name})"

print("\nУРАВНЕНИЕ ПРОКСИ-МЕТРИКИ (Нормализованное):")
print(formula + f"\n           {'+' if bias > 0 else '-'} {abs(bias):.4f} (Bias)")
