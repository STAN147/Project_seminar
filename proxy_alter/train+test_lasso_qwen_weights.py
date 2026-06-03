import os
import sys
import pandas as pd
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import LeaveOneOut

# Функция для подсчета попаданий в топ-5 бесполезных слоев (для регрессии)
def get_top_5_overlap_regression(y_true, y_pred):
    # При сортировке по возрастанию (от -70 до +1), последние 5 элементов - это слои с падением около 0 (наш мусор)
    actual_top5 = set(np.argsort(y_true)[-5:])
    predicted_top5 = set(np.argsort(y_pred)[-5:])
    overlap = len(actual_top5.intersection(predicted_top5))
    return overlap, actual_top5, predicted_top5

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

# =====================================================================
# ПРИМЕНЕНИЕ ПОДХОДА 3: ВЗВЕШИВАНИЕ ПРИМЕРОВ (SAMPLE WEIGHTS)
# =====================================================================
# 1. Создаем базовый массив весов (по умолчанию 1.0)
weights = np.ones(len(y_numpy))

# 2. Находим индексы 5 "мусорных" слоев (самые большие значения, т.е. около 0 или в плюсе)
useless_indices = np.argsort(y_numpy)[-5:]

# 3. Умножаем штраф за ошибку на этих слоях в 10 раз
weights[useless_indices] = 10.0

print(f"Размерность матрицы признаков Qwen: {X_df.shape}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_df.columns)

loo = LeaveOneOut()
lasso_model = LassoCV(cv=loo, random_state=42, max_iter=1500000)

# 4. Обучаем модель с передачей весов!
lasso_model.fit(X_scaled_df, y_numpy, sample_weight=weights)

coefs = pd.Series(lasso_model.coef_, index=X_scaled_df.columns)
important_features = coefs[coefs != 0].sort_values(key=abs, ascending=False)

print(f"\n=== Результаты обучения ВЗВЕШЕННОГО LASSO (Qwen1.5) ===")
print(f"Оптимальный параметр alpha: {lasso_model.alpha_:.4f}")
print(f"Lasso оставила признаков: {len(important_features)} из {len(X_df.columns)}")
print("\nТоп признаков и их веса:")
print(important_features)

y_pred_qwen = lasso_model.predict(X_scaled_df)
spearman_corr, _ = spearmanr(y_numpy, y_pred_qwen)
overlap_qwen, act_qwen, pred_qwen = get_top_5_overlap_regression(y_numpy, y_pred_qwen)

print(f"\nСпирмен на обучающей выборке (Qwen): {spearman_corr:.3f}")
print(f"Точность прунинга (Qwen): Угадано {overlap_qwen} из 5 слоёв на удаление.")
print(f"Реальные индексы: {sorted(list(act_qwen))}")
print(f"Предсказанные:    {sorted(list(pred_qwen))}")


# =====================================================================
# БЛОК 2: ТЕСТИРОВАНИЕ НА PHI-TINY (COMMONSENSE)
# =====================================================================
benchmark_dir = os.path.abspath(os.path.join(BASE_DIR, "commonsense", "phi-tiny metrics + layers + proxy", "metric data", "metrics"))

ablation_drops_phi = [
    -65.5, -10.4, -7.2, -42.8, -7.2, -6.8, -7.5, -4.6, -6.1, -7.5,
    -6.6, -6.3, -4.4, -7.9, -13.1, -1.2, -2.3, -3.4, -1.3, -2.4,
    -0.6, -0.4, -0.7, -2.9, -0.2, -0.7, -0.6, +0.6, -0.7, 0.0,
    -0.3, -1.0
]

num_layers_phi = len(ablation_drops_phi)
all_features_phi = []
feature_names_phi = []

for filename, feat_name in features_2d:
    filepath = os.path.join(benchmark_dir, filename)
    df = pd.read_csv(filepath, index_col=0)
    matrix = df.values
    
    f_prev = np.zeros(num_layers_phi)
    f_next = np.zeros(num_layers_phi)
    f_first = np.zeros(num_layers_phi)
    f_last = np.zeros(num_layers_phi)
    f_mean = np.zeros(num_layers_phi)
    f_std = np.zeros(num_layers_phi)
    
    for i in range(num_layers_phi):
        f_prev[i] = matrix[i, i - 1] if i > 0 else matrix[i, i]
        f_next[i] = matrix[i, i + 1] if i < num_layers_phi - 1 else matrix[i, i]
        f_first[i] = matrix[i, 0]
        f_last[i] = matrix[i, num_layers_phi - 1]
        f_mean[i] = np.mean(matrix[i, :])
        f_std[i] = np.std(matrix[i, :])

    all_features_phi.extend([f_prev, f_next, f_first, f_last, f_mean, f_std])
    feature_names_phi.extend([
        f"{feat_name}_prev", f"{feat_name}_next", 
        f"{feat_name}_first", f"{feat_name}_last", 
        f"{feat_name}_mean", f"{feat_name}_std"
    ])

df_ent_phi = pd.read_csv(os.path.join(benchmark_dir, "metric_09_Router_Entropy.csv"), index_col=0)
all_features_phi.append(df_ent_phi['Avg_Router_Entropy'].values)
feature_names_phi.append("Router_Entropy")

X_phi_df = pd.DataFrame(np.column_stack(all_features_phi), columns=feature_names_phi)
y_phi_numpy = np.array(ablation_drops_phi)

X_phi_scaled = scaler.transform(X_phi_df) 
X_phi_scaled_df = pd.DataFrame(X_phi_scaled, columns=X_phi_df.columns)

y_pred_phi = lasso_model.predict(X_phi_scaled_df)
spearman_corr_phi, p_value_phi = spearmanr(y_phi_numpy, y_pred_phi)
overlap_phi, act_phi, pred_phi = get_top_5_overlap_regression(y_phi_numpy, y_pred_phi)

print(f"\n=== Генерализация на Phi-tiny (Commonsense) ===")
print(f"Спирмен на тестовой выборке: {spearman_corr_phi:.3f} (p-value: {p_value_phi:.3f})")
print(f"Точность прунинга: Угадано {overlap_phi} из 5 слоёв на удаление.")
print(f"Реальные индексы: {sorted(list(act_phi))}")
print(f"Предсказанные:    {sorted(list(pred_phi))}")


# =====================================================================
# БЛОК 3: ТЕСТИРОВАНИЕ НА PHI-TINY (SIQA)
# =====================================================================
benchmark_dir = os.path.abspath(os.path.join(BASE_DIR, "siqa", "phi-tiny metrics + layers", "metric data", "metrics"))
ablations_file_path = os.path.abspath(os.path.join(BASE_DIR, "siqa", "phi-tiny metrics + layers", "accuracy data", "phi_tiny_siqa_ablations.csv"))
df_ablations_siqa_phi = pd.read_csv(ablations_file_path)
y_siqa_numpy_phi = df_ablations_siqa_phi["Ablation_Drop"].values
num_layers_siqa_phi = len(y_siqa_numpy_phi)

all_features_siqa_phi = []
feature_names_siqa_phi = []

for filename, feat_name in features_2d:
    filepath = os.path.join(benchmark_dir, filename)
    df_mat = pd.read_csv(filepath, index_col=0)
    matrix = df_mat.values
    
    f_prev = np.zeros(num_layers_siqa_phi)
    f_next = np.zeros(num_layers_siqa_phi)
    f_first = np.zeros(num_layers_siqa_phi)
    f_last = np.zeros(num_layers_siqa_phi)
    f_mean = np.zeros(num_layers_siqa_phi)
    f_std = np.zeros(num_layers_siqa_phi)
    
    for i in range(num_layers_siqa_phi):
        f_prev[i] = matrix[i, i - 1] if i > 0 else matrix[i, i]
        f_next[i] = matrix[i, i + 1] if i < num_layers_siqa_phi - 1 else matrix[i, i]
        f_first[i] = matrix[i, 0]
        f_last[i] = matrix[i, num_layers_siqa_phi - 1]
        f_mean[i] = np.mean(matrix[i, :])
        f_std[i] = np.std(matrix[i, :])

    all_features_siqa_phi.extend([f_prev, f_next, f_first, f_last, f_mean, f_std])
    feature_names_siqa_phi.extend([
        f"{feat_name}_prev", f"{feat_name}_next", 
        f"{feat_name}_first", f"{feat_name}_last", 
        f"{feat_name}_mean", f"{feat_name}_std"
    ])

df_ent_siqa_phi = pd.read_csv(os.path.join(benchmark_dir, "metric_09_Router_Entropy.csv"), index_col=0)
all_features_siqa_phi.append(df_ent_siqa_phi['Avg_Router_Entropy'].values)
feature_names_siqa_phi.append("Router_Entropy")

X_siqa_phi_df = pd.DataFrame(np.column_stack(all_features_siqa_phi), columns=feature_names_siqa_phi)
X_siqa_phi_scaled = scaler.transform(X_siqa_phi_df)
X_siqa_phi_scaled_df = pd.DataFrame(X_siqa_phi_scaled, columns=X_siqa_phi_df.columns)

y_pred_siqa_phi = lasso_model.predict(X_siqa_phi_scaled_df)
spearman_corr_siqa_phi, p_value_siqa_phi = spearmanr(y_siqa_numpy_phi, y_pred_siqa_phi)
overlap_siqa_phi, act_siqa_phi, pred_siqa_phi = get_top_5_overlap_regression(y_siqa_numpy_phi, y_pred_siqa_phi)

print(f"\n=== Проверка Dataset Bias (Phi-tiny + SIQA) ===")
print(f"Спирмен на независимом датасете SIQA: {spearman_corr_siqa_phi:.3f} (p-value: {p_value_siqa_phi:.3f})")
print(f"Точность прунинга: Угадано {overlap_siqa_phi} из 5 слоёв на удаление.")
print(f"Реальные индексы: {sorted(list(act_siqa_phi))}")
print(f"Предсказанные:    {sorted(list(pred_siqa_phi))}")


# =====================================================================
# БЛОК 4: ТЕСТИРОВАНИЕ НА QWEN (SIQA)
# =====================================================================
benchmark_dir = os.path.abspath(os.path.join(BASE_DIR, "siqa", "Qwen metrics + layers", "metric data", "metrics"))
ablations_file_path = os.path.abspath(os.path.join(BASE_DIR, "siqa", "Qwen metrics + layers", "accuracy data", "phi_tiny_siqa_ablations.csv"))
df_ablations_siqa_qwen = pd.read_csv(ablations_file_path)
y_siqa_numpy_qwen = df_ablations_siqa_qwen["Ablation_Drop"].values
num_layers_siqa_qwen = len(y_siqa_numpy_qwen)

all_features_siqa_qwen = []
feature_names_siqa_qwen = []

for filename, feat_name in features_2d:
    filepath = os.path.join(benchmark_dir, filename)
    df_mat = pd.read_csv(filepath, index_col=0)
    matrix = df_mat.values
    
    f_prev = np.zeros(num_layers_siqa_qwen)
    f_next = np.zeros(num_layers_siqa_qwen)
    f_first = np.zeros(num_layers_siqa_qwen)
    f_last = np.zeros(num_layers_siqa_qwen)
    f_mean = np.zeros(num_layers_siqa_qwen)
    f_std = np.zeros(num_layers_siqa_qwen)
    
    for i in range(num_layers_siqa_qwen):
        f_prev[i] = matrix[i, i - 1] if i > 0 else matrix[i, i]
        f_next[i] = matrix[i, i + 1] if i < num_layers_siqa_qwen - 1 else matrix[i, i]
        f_first[i] = matrix[i, 0]
        f_last[i] = matrix[i, num_layers_siqa_qwen - 1]
        f_mean[i] = np.mean(matrix[i, :])
        f_std[i] = np.std(matrix[i, :])

    all_features_siqa_qwen.extend([f_prev, f_next, f_first, f_last, f_mean, f_std])
    feature_names_siqa_qwen.extend([
        f"{feat_name}_prev", f"{feat_name}_next", 
        f"{feat_name}_first", f"{feat_name}_last", 
        f"{feat_name}_mean", f"{feat_name}_std"
    ])

df_ent_siqa_qwen = pd.read_csv(os.path.join(benchmark_dir, "metric_09_Router_Entropy.csv"), index_col=0)
all_features_siqa_qwen.append(df_ent_siqa_qwen['Avg_Router_Entropy'].values)
feature_names_siqa_qwen.append("Router_Entropy")

X_siqa_qwen_df = pd.DataFrame(np.column_stack(all_features_siqa_qwen), columns=feature_names_siqa_qwen)
X_siqa_qwen_scaled = scaler.transform(X_siqa_qwen_df)
X_siqa_qwen_scaled_df = pd.DataFrame(X_siqa_qwen_scaled, columns=X_siqa_qwen_df.columns)

y_pred_siqa_qwen = lasso_model.predict(X_siqa_qwen_scaled_df)
spearman_corr_siqa_qwen, p_value_siqa_qwen = spearmanr(y_siqa_numpy_qwen, y_pred_siqa_qwen)
overlap_siqa_qwen, act_siqa_qwen, pred_siqa_qwen = get_top_5_overlap_regression(y_siqa_numpy_qwen, y_pred_siqa_qwen)

print(f"\n=== Проверка Dataset Bias (Qwen + SIQA) ===")
print(f"Спирмен на независимом датасете SIQA: {spearman_corr_siqa_qwen:.3f} (p-value: {p_value_siqa_qwen:.3f})")
print(f"Точность прунинга: Угадано {overlap_siqa_qwen} из 5 слоёв на удаление.")
print(f"Реальные индексы: {sorted(list(act_siqa_qwen))}")
print(f"Предсказанные:    {sorted(list(pred_siqa_qwen))}")
print("=" * 60)

'''
Размерность матрицы признаков Qwen: (24, 49)

=== Результаты обучения ВЗВЕШЕННОГО LASSO (Qwen1.5) ===
Оптимальный параметр alpha: 2.2708
Lasso оставила признаков: 5 из 49

Топ признаков и их веса:
Res_Contrib_std     -10.358358
Cosine_Dist_std       6.028815
Cosine_Dist_first     1.292899
L_Inf_next            1.175685
Router_Entropy       -0.228446
dtype: float64

Спирмен на обучающей выборке (Qwen): 0.754
Точность прунинга (Qwen): Угадано 4 из 5 слоёв на удаление.
Реальные индексы: [np.int64(16), np.int64(18), np.int64(19), np.int64(21), np.int64(22)]
Предсказанные:    [np.int64(18), np.int64(19), np.int64(20), np.int64(21), np.int64(22)]

=== Генерализация на Phi-tiny (Commonsense) ===
Спирмен на тестовой выборке: 0.238 (p-value: 0.191)
Точность прунинга: Угадано 0 из 5 слоёв на удаление.
Реальные индексы: [np.int64(21), np.int64(24), np.int64(27), np.int64(29), np.int64(30)]
Предсказанные:    [np.int64(14), np.int64(15), np.int64(16), np.int64(17), np.int64(20)]

=== Проверка Dataset Bias (Phi-tiny + SIQA) ===
Спирмен на независимом датасете SIQA: 0.257 (p-value: 0.155)
Точность прунинга: Угадано 1 из 5 слоёв на удаление.
Реальные индексы: [np.int64(20), np.int64(26), np.int64(27), np.int64(28), np.int64(31)]
Предсказанные:    [np.int64(14), np.int64(15), np.int64(16), np.int64(17), np.int64(20)]

=== Проверка Dataset Bias (Qwen + SIQA) ===
Спирмен на независимом датасете SIQA: 0.867 (p-value: 0.000)
Точность прунинга: Угадано 3 из 5 слоёв на удаление.
Реальные индексы: [np.int64(17), np.int64(18), np.int64(19), np.int64(21), np.int64(23)]
Предсказанные:    [np.int64(18), np.int64(19), np.int64(20), np.int64(21), np.int64(22)]
============================================================

крч тоже залупа

'''