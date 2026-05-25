import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

metrics_dir = os.path.abspath(os.path.join(BASE_DIR, "phi-tiny layers + metrics + proxy", "metric data", "metrics"))

actual_drops = np.array([
    -65.5, -10.4, -7.2, -42.8, -7.2, -6.8, -7.5, -4.6, -6.1, -7.5,
    -6.6, -6.3, -4.4, -7.9, -13.1, -1.2, -2.3, -3.4, -1.3, -2.4,
    -0.6, -0.4, -0.7, -2.9, -0.2, -0.7, -0.6, +0.6, -0.7, 0.0,
    -0.3, -1.0
])

weights = {
    'L1_Dist': 0.7280,
    'Cosine_Dist': -0.6287,
    'Pearson': 0.6283,
    'L_Inf': 0.4512,
    'MSE': 0.2983,
    'Router_Entropy': 0.1063,
    'Res_Contrib': 0.0540,
    'Var_Ratio': 0.0400,
    'CKA': 0.0205
}

features = {}

matrix_files = {
    'MSE': "metric_01_MSE.csv",
    'Cosine_Dist': "metric_02_Cosine_Distance.csv",
    'Res_Contrib': "metric_03_Residual_Contribution.csv",
    'CKA': "metric_04_CKA.csv",
    'L1_Dist': "metric_05_L1_Distance.csv",
    'L_Inf': "metric_06_L_Infinity.csv",
    'Var_Ratio': "metric_07_Variance_Ratio.csv",
    'Pearson': "metric_08_Pearson_Correlation.csv"
}

print("Загрузка метрик Phi-tiny-MoE...")
for feat_name, filename in matrix_files.items():
    filepath = os.path.join(metrics_dir, filename)
    df = pd.read_csv(filepath, index_col=0)
    features[feat_name] = np.diag(df.values, k=1)

entropy_path = os.path.join(metrics_dir, "metric_09_Router_Entropy.csv")
if os.path.exists(entropy_path):
    df_ent = pd.read_csv(entropy_path, index_col=0)
    features['Router_Entropy'] = df_ent.values[1:32].flatten()
else:
    print("Внимание: файл энтропии не найден, заполняем нулями.")
    features['Router_Entropy'] = np.zeros(31)

y_true = actual_drops[1:]

y_pred = np.zeros(31)

for feat_name, weight in weights.items():
    arr = features[feat_name]
    std_val = np.std(arr)
    if std_val != 0:
        arr_norm = (arr - np.mean(arr)) / std_val
    else:
        arr_norm = arr - np.mean(arr)
        
    y_pred += arr_norm * weight

pearson_corr, p_pearson = pearsonr(y_pred, y_true)
spearman_corr, p_spearman = spearmanr(y_pred, y_true)

print("\n" + "="*60)
print(" ТРАНСФЕР ПРОКСИ-МЕТРИКИ: QWEN -> PHI-TINY")
print("="*60)
print(f"ПИРСОН (Линейная зависимость):  {pearson_corr:.4f} (p-value: {p_pearson:.4f})")
print(f"СПИРМЕН (Ранговая сортировка):  {spearman_corr:.4f} (p-value: {p_spearman:.4f})")
print("="*60)
