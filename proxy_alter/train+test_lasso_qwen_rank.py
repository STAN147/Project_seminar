import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr, spearmanr, rankdata # Добавили rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
benchmark_dir = os.path.abspath(os.path.join(BASE_DIR, "commonsense", "Qwen1.5 metrics + layers + proxy", "metric data", "metrics"))

ablation_drops = [
    -70.7, -70.1,  -9.9, -58.0, -37.0, -13.0, -66.9,  -6.7, 
    -23.5, -39.6, -41.5, -11.6, -61.4, -46.5, -19.5,  -1.2, 
      1.7, -11.9,   0.1,   1.2,  -4.5,   1.3,   1.1,  -1.1
]

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
all_features = [] # Список для сбора колонок признаков
feature_names = [] # Имена колонок

for filename, feat_name in features_2d:
    filepath = os.path.join(benchmark_dir, filename)
    df = pd.read_csv(filepath, index_col=0)
    matrix = df.values
    
    # Временные массивы для новых признаков конкретной метрики
    f_prev = np.zeros(num_layers)
    f_next = np.zeros(num_layers)
    f_first = np.zeros(num_layers)
    f_last = np.zeros(num_layers)
    f_mean = np.zeros(num_layers)
    f_std = np.zeros(num_layers)
    
    for i in range(num_layers):
        # 1. Связь с предыдущим слоем
        f_prev[i] = matrix[i, i - 1] if i > 0 else matrix[i, i]
        
        # 2. Связь со следующим слоем
        f_next[i] = matrix[i, i + 1] if i < num_layers - 1 else matrix[i, i]
        
        # 3. Связь с самым первым слоем (входной контекст)
        f_first[i] = matrix[i, 0]
        
        # 4. Связь с самым последним слоем (выходной контекст)
        f_last[i] = matrix[i, num_layers - 1]
        
        # 5. Глобальные статистики строки
        f_mean[i] = np.mean(matrix[i, :])
        
        # 6. Разброс
        f_std[i] = np.std(matrix[i, :])

    all_features.extend([f_prev, f_next, f_first, f_last, f_mean, f_std])
    feature_names.extend([
        f"{feat_name}_prev", f"{feat_name}_next", 
        f"{feat_name}_first", f"{feat_name}_last", 
        f"{feat_name}_mean", f"{feat_name}_std"
    ])

df_ent = pd.read_csv(os.path.join(benchmark_dir, "metric_09_Router_Entropy.csv"), index_col=0)
all_features.append(df_ent['Avg_Router_Entropy'].values)
feature_names.append("Router_Entropy")

X_numpy = np.column_stack(all_features)
X_df = pd.DataFrame(X_numpy, columns=feature_names)
y_numpy = np.array(ablation_drops)

# --- ПРИМЕНЕНИЕ ПОДХОДА 1: ПЕРЕХОД К РАНГАМ ---
# Ранг 1 — самый важный слой (максимальное падение), Ранг 24 — самый бесполезный (около нуля или плюс)
y_ranks = rankdata(y_numpy) 

print(f"Размерность матрицы признаков Qwen: {X_df.shape}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_df.columns)

# Обучаем LASSO на рангах
loo = LeaveOneOut()
lasso_model = LassoCV(cv=loo, random_state=42, max_iter=1500000)
lasso_model.fit(X_scaled_df, y_ranks)

coefs = pd.Series(lasso_model.coef_, index=X_scaled_df.columns)
important_features = coefs[coefs != 0].sort_values(key=abs, ascending=False)

print(f"\n=== Результаты обучения LASSO на РАНГАХ (Qwen1.5) ===")
print(f"Оптимальный параметр alpha: {lasso_model.alpha_:.4f}")
print(f"Lasso оставила признаков: {len(important_features)} из {len(X_df.columns)}")
print("\nТоп признаков и их веса в формуле предсказания РАНГА:")
print(important_features)

y_pred_qwen = lasso_model.predict(X_scaled_df)
spearman_corr, _ = spearmanr(y_numpy, y_pred_qwen)
print(f"\nСпирмен на обучающей выборке (Qwen): {spearman_corr:.3f}")


# =====================================================================
# БЛОК 2: ТЕСТИРОВАНИЕ НА PHI-TINY (COMMONSENSE)
# =====================================================================
benchmark_dir = os.path.abspath(os.path.join(BASE_DIR, "commonsense", "phi-tiny metrics + layers + proxy", "metric data", "metrics"))

ablation_drops = [
    -65.5, -10.4, -7.2, -42.8, -7.2, -6.8, -7.5, -4.6, -6.1, -7.5,
    -6.6, -6.3, -4.4, -7.9, -13.1, -1.2, -2.3, -3.4, -1.3, -2.4,
    -0.6, -0.4, -0.7, -2.9, -0.2, -0.7, -0.6, +0.6, -0.7, 0.0,
    -0.3, -1.0
]

num_layers = len(ablation_drops)
all_features = []
feature_names = []

for filename, feat_name in features_2d:
    filepath = os.path.join(benchmark_dir, filename)
    df = pd.read_csv(filepath, index_col=0)
    matrix = df.values
    
    f_prev = np.zeros(num_layers)
    f_next = np.zeros(num_layers)
    f_first = np.zeros(num_layers)
    f_last = np.zeros(num_layers)
    f_mean = np.zeros(num_layers)
    f_std = np.zeros(num_layers)
    
    for i in range(num_layers):
        f_prev[i] = matrix[i, i - 1] if i > 0 else matrix[i, i]
        f_next[i] = matrix[i, i + 1] if i < num_layers - 1 else matrix[i, i]
        f_first[i] = matrix[i, 0]
        f_last[i] = matrix[i, num_layers - 1]
        f_mean[i] = np.mean(matrix[i, :])
        f_std[i] = np.std(matrix[i, :])

    all_features.extend([f_prev, f_next, f_first, f_last, f_mean, f_std])
    feature_names.extend([
        f"{feat_name}_prev", f"{feat_name}_next", 
        f"{feat_name}_first", f"{feat_name}_last", 
        f"{feat_name}_mean", f"{feat_name}_std"
    ])

df_ent = pd.read_csv(os.path.join(benchmark_dir, "metric_09_Router_Entropy.csv"), index_col=0)
all_features.append(df_ent['Avg_Router_Entropy'].values)
feature_names.append("Router_Entropy")

X_numpy = np.column_stack(all_features)
X_df = pd.DataFrame(X_numpy, columns=feature_names)
y_numpy = np.array(ablation_drops)

print(f"\nРазмерность матрицы признаков Phi-tiny (Commonsense): {X_df.shape}")

X_phi_scaled = scaler.transform(X_df) 
X_phi_scaled_df = pd.DataFrame(X_phi_scaled, columns=X_df.columns)

y_pred_phi = lasso_model.predict(X_phi_scaled_df)
spearman_corr_phi, p_value_phi = spearmanr(y_numpy, y_pred_phi)

print(f"=== Результаты генерализации на Phi-tiny (Commonsense) ===")
print(f"Спирмен на тестовой выборке: {spearman_corr_phi:.3f} (p-value: {p_value_phi:.3f})")


# =====================================================================
# БЛОК 3: ТЕСТИРОВАНИЕ НА PHI-TINY (SIQA)
# =====================================================================
benchmark_dir = os.path.abspath(os.path.join(BASE_DIR, "siqa", "phi-tiny metrics + layers", "metric data", "metrics"))
ablations_file_path = os.path.abspath(os.path.join(BASE_DIR, "siqa", "phi-tiny metrics + layers", "accuracy data", "phi_tiny_siqa_ablations.csv"))
df_ablations_siqa = pd.read_csv(ablations_file_path)
y_siqa_numpy = df_ablations_siqa["Ablation_Drop"].values
num_layers_siqa = len(y_siqa_numpy)
print(f"\nКоличество слоёв в тестовой выборке Phi-tiny (SIQA): {num_layers_siqa}")

all_features_siqa = []
feature_names_siqa = []

for filename, feat_name in features_2d:
    filepath = os.path.join(benchmark_dir, filename)
    df_mat = pd.read_csv(filepath, index_col=0)
    matrix = df_mat.values
    
    f_prev = np.zeros(num_layers_siqa)
    f_next = np.zeros(num_layers_siqa)
    f_first = np.zeros(num_layers_siqa)
    f_last = np.zeros(num_layers_siqa)
    f_mean = np.zeros(num_layers_siqa)
    f_std = np.zeros(num_layers_siqa)
    
    for i in range(num_layers_siqa):
        f_prev[i] = matrix[i, i - 1] if i > 0 else matrix[i, i]
        f_next[i] = matrix[i, i + 1] if i < num_layers_siqa - 1 else matrix[i, i]
        f_first[i] = matrix[i, 0]
        f_last[i] = matrix[i, num_layers_siqa - 1]
        f_mean[i] = np.mean(matrix[i, :])
        f_std[i] = np.std(matrix[i, :])

    all_features_siqa.extend([f_prev, f_next, f_first, f_last, f_mean, f_std])
    feature_names_siqa.extend([
        f"{feat_name}_prev", f"{feat_name}_next", 
        f"{feat_name}_first", f"{feat_name}_last", 
        f"{feat_name}_mean", f"{feat_name}_std"
    ])

df_ent_siqa = pd.read_csv(os.path.join(benchmark_dir, "metric_09_Router_Entropy.csv"), index_col=0)
all_features_siqa.append(df_ent_siqa['Avg_Router_Entropy'].values)
feature_names_siqa.append("Router_Entropy")

X_siqa_df = pd.DataFrame(np.column_stack(all_features_siqa), columns=feature_names_siqa)
X_siqa_scaled = scaler.transform(X_siqa_df)
X_siqa_scaled_df = pd.DataFrame(X_siqa_scaled, columns=X_siqa_df.columns)

y_pred_siqa = lasso_model.predict(X_siqa_scaled_df)
spearman_corr_siqa, p_value_siqa = spearmanr(y_siqa_numpy, y_pred_siqa)

print(f"=== Результаты проверки на Dataset Bias (Phi-tiny + SIQA) ===")
print(f"Спирмен на независимом датасете SIQA: {spearman_corr_siqa:.3f} (p-value: {p_value_siqa:.3f})")


# =====================================================================
# БЛОК 4: ТЕСТИРОВАНИЕ НА QWEN (SIQA)
# =====================================================================
benchmark_dir = os.path.abspath(os.path.join(BASE_DIR, "siqa", "Qwen metrics + layers", "metric data", "metrics"))
ablations_file_path = os.path.abspath(os.path.join(BASE_DIR, "siqa", "Qwen metrics + layers", "accuracy data", "phi_tiny_siqa_ablations.csv"))
df_ablations_siqa = pd.read_csv(ablations_file_path)
y_siqa_numpy = df_ablations_siqa["Ablation_Drop"].values
num_layers_siqa = len(y_siqa_numpy)
print(f"\nКоличество слоёв в тестовой выборке Qwen (SIQA): {num_layers_siqa}")

all_features_siqa = []
feature_names_siqa = []

for filename, feat_name in features_2d:
    filepath = os.path.join(benchmark_dir, filename)
    df_mat = pd.read_csv(filepath, index_col=0)
    matrix = df_mat.values
    
    f_prev = np.zeros(num_layers_siqa)
    f_next = np.zeros(num_layers_siqa)
    f_first = np.zeros(num_layers_siqa)
    f_last = np.zeros(num_layers_siqa)
    f_mean = np.zeros(num_layers_siqa)
    f_std = np.zeros(num_layers_siqa)
    
    for i in range(num_layers_siqa):
        f_prev[i] = matrix[i, i - 1] if i > 0 else matrix[i, i]
        f_next[i] = matrix[i, i + 1] if i < num_layers_siqa - 1 else matrix[i, i]
        f_first[i] = matrix[i, 0]
        f_last[i] = matrix[i, num_layers_siqa - 1]
        f_mean[i] = np.mean(matrix[i, :])
        f_std[i] = np.std(matrix[i, :])

    all_features_siqa.extend([f_prev, f_next, f_first, f_last, f_mean, f_std])
    feature_names_siqa.extend([
        f"{feat_name}_prev", f"{feat_name}_next", 
        f"{feat_name}_first", f"{feat_name}_last", 
        f"{feat_name}_mean", f"{feat_name}_std"
    ])

df_ent_siqa = pd.read_csv(os.path.join(benchmark_dir, "metric_09_Router_Entropy.csv"), index_col=0)
all_features_siqa.append(df_ent_siqa['Avg_Router_Entropy'].values)
feature_names_siqa.append("Router_Entropy")

X_siqa_df = pd.DataFrame(np.column_stack(all_features_siqa), columns=feature_names_siqa)
X_siqa_scaled = scaler.transform(X_siqa_df)
X_siqa_scaled_df = pd.DataFrame(X_siqa_scaled, columns=X_siqa_df.columns)

y_pred_siqa = lasso_model.predict(X_siqa_scaled_df)
spearman_corr_siqa, p_value_siqa = spearmanr(y_siqa_numpy, y_pred_siqa)

print(f"=== Результаты проверки на Dataset Bias (Qwen + SIQA) ===")
print(f"Спирмен на независимом датасете SIQA: {spearman_corr_siqa:.3f} (p-value: {p_value_siqa:.3f})")
print("=" * 60)




'''Размерность матрицы признаков Qwen: (24, 49)

=== Результаты обучения LASSO на РАНГАХ (Qwen1.5) ===
Оптимальный параметр alpha: 0.6079
Lasso оставила признаков: 7 из 49

Топ признаков и их веса в формуле предсказания РАНГА:
Cosine_Dist_std      2.233731
Cosine_Dist_first    1.974709
Cosine_Dist_last    -1.770079
Var_Ratio_prev       0.891725
L_Inf_next           0.355225
Pearson_last         0.080087
Pearson_first       -0.000008
dtype: float64

Спирмен на обучающей выборке (Qwen): 0.806

Размерность матрицы признаков Phi-tiny (Commonsense): (32, 49)
=== Результаты генерализации на Phi-tiny (Commonsense) ===
Спирмен на тестовой выборке: 0.872 (p-value: 0.000)

Количество слоёв в тестовой выборке Phi-tiny (SIQA): 32
=== Результаты проверки на Dataset Bias (Phi-tiny + SIQA) ===
Спирмен на независимом датасете SIQA: 0.761 (p-value: 0.000)

Количество слоёв в тестовой выборке Qwen (SIQA): 24
=== Результаты проверки на Dataset Bias (Qwen + SIQA) ===
Спирмен на независимом датасете SIQA: 0.913 (p-value: 0.000)
============================================================'''