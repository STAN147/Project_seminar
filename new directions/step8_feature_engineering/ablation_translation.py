import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "..", "..", "actual dataset")

def load_dataset(model, task):
    metrics_path = os.path.join(DATASET_DIR, "metrics", model, task)
    datafree_path = os.path.join(DATASET_DIR, "metrics", model, "data-free")
    target_path = os.path.join(DATASET_DIR, "target values", model, task, "PPL_drops.csv")

    df_ppl = pd.read_csv(target_path)
    num_layers = len(df_ppl)
    
    ranks = df_ppl['PPL_Degradation'].rank(ascending=False).astype(float)
    r_min, r_max = ranks.min(), ranks.max()
    if r_max > r_min:
        y_numpy = ((ranks - r_min) / (r_max - r_min)).values
    else:
        y_numpy = np.zeros(num_layers)

    all_features = []
    feature_names = []

    metrics_1d = [
        (metrics_path, "metric_09_Router_Entropy.csv", "Router_Entropy"),
        (datafree_path, "metric_11_SVD_Entropy.csv", "SVD_Entropy"),
        (metrics_path, "metric_12_KL_noise.csv", "KL_Noise"),
        (metrics_path, "metric_13_LogitLens.csv", "LogitLens")
    ]

    for dir_path, filename, feat_name in metrics_1d:
        path = os.path.join(dir_path, filename)
        if os.path.exists(path):
            df_1d = pd.read_csv(path, index_col=0)
            all_features.append(df_1d.iloc[:, 0].values)
        else:
            all_features.append(np.zeros(num_layers))
        feature_names.append(feat_name)

    x_df = pd.DataFrame(np.column_stack(all_features), columns=feature_names)
    
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_df)
    x_scaled_df = pd.DataFrame(x_scaled, columns=feature_names)
    
    return x_scaled_df, y_numpy, num_layers

def main():
    models = ["phi-tiny", "Qwen", "gemma"]
    tasks = ["csqa", "siqa"]

    X_train_list = []
    y_train_list = []
    test_data = {}

    for model in models:
        for task in tasks:
            try:
                x_df, y_np, n_layers = load_dataset(model, task)
                
                name = f"{model} + {task}"
                if model in ["phi-tiny", "Qwen"]:
                    X_train_list.append(x_df)
                    y_train_list.append(y_np)
                else:
                    test_data[name] = (x_df, y_np, n_layers)
            except Exception:
                pass

    X_train = pd.concat(X_train_list, ignore_index=True)
    y_train = np.concatenate(y_train_list)

    ridge_model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0])
    ridge_model.fit(X_train, y_train)

    coefs = pd.Series(ridge_model.coef_, index=X_train.columns)
    important_features = coefs.sort_values(key=abs, ascending=False)
    
    print(important_features)
    print()

    for name, (x_test, y_test, num_layers) in test_data.items():
        y_pred = ridge_model.predict(x_test)
        spearman, _ = spearmanr(y_test, y_pred)
        
        y_count = max(1, int(num_layers * 0.2))
        
        real_indices = np.argsort(y_test)[-y_count:]
        pred_indices = np.argsort(y_pred)[-y_count:]
        
        x_count = len(set(real_indices).intersection(set(pred_indices)))
        
        print(f"Спирмен на {name}: {spearman:.4f}")
        print(f"Угадано {x_count} из {y_count} неважных слоёв\n")

if __name__ == "__main__":
    main()
