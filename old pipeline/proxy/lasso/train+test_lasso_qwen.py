import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import LeaveOneOut


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
    all_features = []
    feature_names = []
    for filename, feat_name in features_2d:
        df = pd.read_csv(os.path.join(dataset_path, filename), index_col=0)
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
    df_ent = pd.read_csv(os.path.join(dataset_path, "metric_09_Router_Entropy.csv"), index_col=0)
    all_features.append(df_ent["Avg_Router_Entropy"].values)
    feature_names.append("Router_Entropy")
    x_df = pd.DataFrame(np.column_stack(all_features), columns=feature_names)
    y_numpy = np.array(drops)
    return x_df, y_numpy


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

x_train, y_train = load_dataset(
    datasets_info["Qwen + Commonsense"]["path"],
    datasets_info["Qwen + Commonsense"]["layers"],
    datasets_info["Qwen + Commonsense"]["drops"]
)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_train_scaled_df = pd.DataFrame(x_train_scaled, columns=x_train.columns)

loo = LeaveOneOut()
lasso_model = LassoCV(cv=loo, random_state=42, max_iter=500000)
lasso_model.fit(x_train_scaled_df, y_train)

coefs = pd.Series(lasso_model.coef_, index=x_train_scaled_df.columns)
important_features = coefs[coefs != 0]
important_features = important_features / important_features.abs().sum()
important_features = important_features.sort_values(key=abs, ascending=False)

print(important_features)
print()

for name, info in datasets_info.items():
    x_test, y_test = load_dataset(info["path"], info["layers"], info["drops"])
    x_test_scaled = scaler.transform(x_test)
    x_test_scaled_df = pd.DataFrame(x_test_scaled, columns=x_test.columns)
    y_pred = lasso_model.predict(x_test_scaled_df)
    spearman, _ = spearmanr(y_test, y_pred)
    print(f"Спирмен на {name}: {spearman:.4f}")

'''
Cosine_Dist_first    0.342608
Var_Ratio_prev       0.206575
Cosine_Dist_last    -0.205916
Cosine_Dist_std      0.192191
L_Inf_next           0.047103
Pearson_last         0.005608
dtype: float64

Спирмен на Qwen + Commonsense: 0.8113
Спирмен на Phi-Tiny + Commonsense: 0.8675
Спирмен на Qwen + SIQA: 0.9115
Спирмен на Phi-Tiny + SIQA: 0.7780
'''