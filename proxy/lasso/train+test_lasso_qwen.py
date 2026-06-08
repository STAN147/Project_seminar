import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
benchmark_dir = os.path.abspath(os.path.join(BASE_DIR, "commonsense", "Qwen1.5 metrics + layers", "metric data", "metrics"))

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
        # 1. Связь с предыдущим слоем (для 0-го слоя берем диагональ, то есть самого себя)
        f_prev[i] = matrix[i, i - 1] if i > 0 else matrix[i, i]
        
        # 2. Связь со следующим слоем (для последнего слоя берем диагональ)
        f_next[i] = matrix[i, i + 1] if i < num_layers - 1 else matrix[i, i]
        
        # 3. Связь с самым первым слоем (входной контекст)
        f_first[i] = matrix[i, 0]
        
        # 4. Связь с самым последним слоем (выходной контекст)
        f_last[i] = matrix[i, num_layers - 1]
        
        # 5. Глобальные статистики строки (насколько слой похож на ВСЕ остальные в среднем)
        f_mean[i] = np.mean(matrix[i, :])
        
        # 6. Разброс (специфичен ли слой в своих связях или одинаков со всеми)
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

print(f"Размерность матрицы признаков: {X_df.shape}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_df.columns)

loo = LeaveOneOut()
lasso_model = LassoCV(cv=loo, random_state=42, max_iter=500000)
lasso_model.fit(X_scaled_df, y_numpy)

coefs = pd.Series(lasso_model.coef_, index=X_scaled_df.columns)
important_features = coefs[coefs != 0].sort_values(key=abs, ascending=False)

print(f"Оптимальный параметр alpha: {lasso_model.alpha_:.4f}")
print(f"Lasso оставила признаков: {len(important_features)} из {len(X_df.columns)}")
print("\nТоп признаков и их веса в итоговой формуле:")
print(important_features)

y_pred_qwen = lasso_model.predict(X_scaled_df)
spearman_corr, p_value = spearmanr(y_numpy, y_pred_qwen)

print(f"\nСпирмен на обучающей выборке (Qwen): {spearman_corr:.3f} (p-value: {p_value:.3f})")

# --- 1. RIDGE (L2) ---
ridge_model = RidgeCV(cv=loo, scoring='neg_mean_squared_error')
ridge_model.fit(X_scaled_df, y_numpy)
y_pred_ridge = ridge_model.predict(X_scaled_df)
corr_ridge, _ = spearmanr(y_numpy, y_pred_ridge)
print(f"1. Ridge CV Spearman:      {corr_ridge:.3f}")

# --- 2. ELASTIC NET ---
# l1_ratio: перебираем баланс от 10% L1 до 99% L1
enet_model = ElasticNetCV(cv=loo, l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99], random_state=42, max_iter=500000)
enet_model.fit(X_scaled_df, y_numpy)
y_pred_enet = enet_model.predict(X_scaled_df)
corr_enet, _ = spearmanr(y_numpy, y_pred_enet)
print(f"2. ElasticNet CV Spearman: {corr_enet:.3f} (Оптимальный l1_ratio: {enet_model.l1_ratio_})")

# --- 3. СЛУЧАЙНЫЙ ЛЕС (Жестко ограниченный) ---
# max_depth=2 и min_samples_leaf=3 спасают от переобучения на 24 примерах
rf_model = RandomForestRegressor(n_estimators=100, max_depth=2, min_samples_leaf=3, random_state=42)
# Для леса LOO нужно запускать вручную через cross_val_predict, чтобы получить честную оценку
y_pred_rf_loo = cross_val_predict(rf_model, X_scaled_df, y_numpy, cv=loo)
corr_rf, _ = spearmanr(y_numpy, y_pred_rf_loo)
print(f"3. Random Forest Spearman: {corr_rf:.3f}")

benchmark_dir = os.path.abspath(os.path.join(BASE_DIR, "commonsense", "phi-tiny metrics + layers", "metric data", "metrics"))

ablation_drops = [
    -65.5, -10.4, -7.2, -42.8, -7.2, -6.8, -7.5, -4.6, -6.1, -7.5,
    -6.6, -6.3, -4.4, -7.9, -13.1, -1.2, -2.3, -3.4, -1.3, -2.4,
    -0.6, -0.4, -0.7, -2.9, -0.2, -0.7, -0.6, +0.6, -0.7, 0.0,
    -0.3, -1.0
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

num_layers = len(ablation_drops)
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
        # 1. Связь с предыдущим слоем (для 0-го слоя берем диагональ, то есть самого себя)
        f_prev[i] = matrix[i, i - 1] if i > 0 else matrix[i, i]
        
        # 2. Связь со следующим слоем (для последнего слоя берем диагональ)
        f_next[i] = matrix[i, i + 1] if i < num_layers - 1 else matrix[i, i]
        
        # 3. Связь с самым первым слоем (входной контекст)
        f_first[i] = matrix[i, 0]
        
        # 4. Связь с самым последним слоем (выходной контекст)
        f_last[i] = matrix[i, num_layers - 1]
        
        # 5. Глобальные статистики строки (насколько слой похож на ВСЕ остальные в среднем)
        f_mean[i] = np.mean(matrix[i, :])
        
        # 6. Разброс (специфичен ли слой в своих связях или одинаков со всеми)
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

print(f"Размерность матрицы признаков: {X_df.shape}")

X_phi_scaled = scaler.transform(X_df) 
X_phi_scaled_df = pd.DataFrame(X_phi_scaled, columns=X_df.columns)

# === ТЕСТИРОВАНИЕ ГЕНЕРАЛИЗАЦИИ ===
# Делаем предсказания нашей "формулой из 6 признаков"
y_pred_phi = lasso_model.predict(X_phi_scaled_df)

# Оцениваем, насколько хорошо сохранилось ранжирование
spearman_corr_phi, p_value_phi = spearmanr(y_numpy, y_pred_phi)

print(f"\n=== Результаты генерализации на Phi-tiny ===")
print(f"Спирмен на тестовой выборке (Phi-tiny): {spearman_corr_phi:.3f} (p-value: {p_value_phi:.3f})")

benchmark_dir = os.path.abspath(os.path.join(BASE_DIR, "siqa", "phi-tiny metrics + layers", "metric data", "metrics"))

ablations_file_path = os.path.abspath(os.path.join(BASE_DIR, "siqa", "phi-tiny metrics + layers", "accuracy data", "ablations.csv"))
df_ablations_siqa = pd.read_csv(ablations_file_path)
# Вытаскиваем столбец дельт в numpy массив
y_siqa_numpy = df_ablations_siqa["Ablation_Drop"].values
num_layers_siqa = len(y_siqa_numpy)
print(f"Количество слоёв в тестовой выборке SIQA: {num_layers_siqa}")

# 2. Очищаем списки и собираем новые признаки из папки SIQA
all_features_siqa = []
feature_names_siqa = []

for filename, feat_name in features_2d:
    filepath = os.path.join(benchmark_dir, filename) # Читает из обновленного benchmark_dir (папка siqa)
    df_mat = pd.read_csv(filepath, index_col=0)
    matrix = df_mat.values
    
    # Инициализируем временные массивы под текущую размерность модели (32 слоя)
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

# Читаем Router Entropy для SIQA
df_ent_siqa = pd.read_csv(os.path.join(benchmark_dir, "metric_09_Router_Entropy.csv"), index_col=0)
all_features_siqa.append(df_ent_siqa['Avg_Router_Entropy'].values)
feature_names_siqa.append("Router_Entropy")

# Формируем итоговую матрицу признаков для SIQA
X_siqa_df = pd.DataFrame(np.column_stack(all_features_siqa), columns=feature_names_siqa)
print(f"Размерность новой матрицы признаков SIQA: {X_siqa_df.shape}")

# 3. Масштабируем признаки СТАРЫМ скейлером (обученным на Qwen Commonsense)
# Никакого fit_transform, используем только transform()!
X_siqa_scaled = scaler.transform(X_siqa_df)
X_siqa_scaled_df = pd.DataFrame(X_siqa_scaled, columns=X_siqa_df.columns)

# 4. Делаем предсказание нашей обученной формулой Lasso
y_pred_siqa = lasso_model.predict(X_siqa_scaled_df)

# 5. Считаем финальный Спирмен для проверки устойчивости к смене датасета
spearman_corr_siqa, p_value_siqa = spearmanr(y_siqa_numpy, y_pred_siqa)

print(f"\n=== Финальные результаты проверки на Dataset Bias ===")
print(f"Спирмен на независимом датасете SIQA: {spearman_corr_siqa:.3f} (p-value: {p_value_siqa:.3f})")
print("=" * 60)

benchmark_dir = os.path.abspath(os.path.join(BASE_DIR, "siqa", "Qwen1.5 metrics + layers", "metric data", "metrics"))

ablations_file_path = os.path.abspath(os.path.join(BASE_DIR, "siqa", "Qwen1.5 metrics + layers", "accuracy data", "ablations.csv"))
df_ablations_siqa = pd.read_csv(ablations_file_path)
# Вытаскиваем столбец дельт в numpy массив
y_siqa_numpy = df_ablations_siqa["Ablation_Drop"].values
num_layers_siqa = len(y_siqa_numpy)
print(f"Количество слоёв в тестовой выборке SIQA: {num_layers_siqa}")

# 2. Очищаем списки и собираем новые признаки из папки SIQA
all_features_siqa = []
feature_names_siqa = []

for filename, feat_name in features_2d:
    filepath = os.path.join(benchmark_dir, filename) # Читает из обновленного benchmark_dir (папка siqa)
    df_mat = pd.read_csv(filepath, index_col=0)
    matrix = df_mat.values
    
    # Инициализируем временные массивы под текущую размерность модели (32 слоя)
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

# Читаем Router Entropy для SIQA
df_ent_siqa = pd.read_csv(os.path.join(benchmark_dir, "metric_09_Router_Entropy.csv"), index_col=0)
all_features_siqa.append(df_ent_siqa['Avg_Router_Entropy'].values)
feature_names_siqa.append("Router_Entropy")

# Формируем итоговую матрицу признаков для SIQA
X_siqa_df = pd.DataFrame(np.column_stack(all_features_siqa), columns=feature_names_siqa)
print(f"Размерность новой матрицы признаков SIQA: {X_siqa_df.shape}")

# 3. Масштабируем признаки СТАРЫМ скейлером (обученным на Qwen Commonsense)
# Никакого fit_transform, используем только transform()!
X_siqa_scaled = scaler.transform(X_siqa_df)
X_siqa_scaled_df = pd.DataFrame(X_siqa_scaled, columns=X_siqa_df.columns)

# 4. Делаем предсказание нашей обученной формулой Lasso
y_pred_siqa = lasso_model.predict(X_siqa_scaled_df)

# 5. Считаем финальный Спирмен для проверки устойчивости к смене датасета
spearman_corr_siqa, p_value_siqa = spearmanr(y_siqa_numpy, y_pred_siqa)

print(f"\n=== Финальные результаты проверки на Dataset Bias ===")
print(f"Спирмен на независимом датасете SIQA: {spearman_corr_siqa:.3f} (p-value: {p_value_siqa:.3f})")
print("=" * 60)


'''Размерность матрицы признаков: (24, 49)
Оптимальный параметр alpha: 2.3985
Lasso оставила признаков: 6 из 49

Топ признаков и их веса в итоговой формуле:
Cosine_Dist_first    8.865018
Var_Ratio_prev       5.345146
Cosine_Dist_last    -5.328091
Cosine_Dist_std      4.972951
L_Inf_next           1.218786
Pearson_last         0.145107
dtype: float64

Спирмен на обучающей выборке (Qwen): 0.811 (p-value: 0.000)
1. Ridge CV Spearman:      0.800
2. ElasticNet CV Spearman: 0.777 (Оптимальный l1_ratio: 0.7)
3. Random Forest Spearman: 0.310
Размерность матрицы признаков: (32, 49)

=== Результаты генерализации на Phi-tiny ===
Спирмен на тестовой выборке (Phi-tiny): 0.867 (p-value: 0.000)
Количество слоёв в тестовой выборке SIQA: 32
Размерность новой матрицы признаков SIQA: (32, 49)

=== Финальные результаты проверки на Dataset Bias ===
Спирмен на независимом датасете SIQA: 0.778 (p-value: 0.000)
============================================================
Количество слоёв в тестовой выборке SIQA: 24
Размерность новой матрицы признаков SIQA: (24, 49)

=== Финальные результаты проверки на Dataset Bias ===
Спирмен на независимом датасете SIQA: 0.912 (p-value: 0.000)
============================================================'''