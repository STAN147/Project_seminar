import os
import re
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATASET_DIR = os.path.join(BASE_DIR, "actual dataset")

MODELS = ["phi-tiny", "Qwen", "gemma"]
TASKS = ["csqa", "siqa"]

def load_metric(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    result = {}
    
    is_matrix = len(df.columns) > 10
    
    if is_matrix:
        num_layers = len(df)
        row_labels = df.iloc[:, 0].astype(str)
        for i in range(num_layers):
            nums = re.findall(r'\d+', row_labels[i])
            if not nums: continue
            l_idx = int(nums[-1])
            
            val_start = df.iloc[i, 1]
            val_next = df.iloc[i, min(i + 2, num_layers)]
            val_end = df.iloc[i, num_layers]
            val_mean = df.iloc[i, 1:].astype(float).mean()
            
            result[l_idx] = {
                '_next': float(val_next),
                '_start': float(val_start),
                '_end': float(val_end),
                '_mean': float(val_mean)
            }
        return result
    else:
        layer_col = next((c for c in df.columns if str(c).lower() in ['layer', 'unnamed: 0', 'index']), None)
        if not layer_col:
            for c in df.columns:
                first_val = df[c].iloc[0]
                if isinstance(first_val, str) and ('layer' in first_val.lower() or 'moe' in first_val.lower()):
                    layer_col = c
                    break
        if layer_col and len(df.columns) > 2:
            for _, row in df.iterrows():
                l_val = row[layer_col]
                idx = None
                if isinstance(l_val, (int, float)) and pd.notna(l_val):
                    idx = int(l_val)
                else:
                    nums = re.findall(r'\d+', str(l_val))
                    if nums: idx = int(nums[-1])
                    
                if idx is not None:
                    result[idx] = {}
                    for c in df.columns:
                        if c != layer_col:
                            result[idx][f"_{c}"] = float(row[c])
            return result
        val_col = next((c for c in df.columns if 'value' in str(c).lower()), None)
        if not val_col:
            val_col = df.columns[-1]
        if layer_col:
            for _, row in df.iterrows():
                l_val = row[layer_col]
                if isinstance(l_val, (int, float)):
                    if pd.notna(l_val):
                        result[int(l_val)] = {'': float(row[val_col])}
                else:
                    nums = re.findall(r'\d+', str(l_val))
                    if nums:
                        result[int(nums[-1])] = {'': float(row[val_col])}
            return result
        else:
            return {int(i): {'': float(val)} for i, val in enumerate(df[val_col])}

def main():
    all_data = []

    for model in MODELS:
        for task in TASKS:
            metrics_dir = os.path.join(DATASET_DIR, "metrics", model, task)
            metrics_datafree_dir = os.path.join(DATASET_DIR, "metrics", model, "data-free")
            ablations_path = os.path.join(DATASET_DIR, "ablations", model, task, "ablations.csv")

            acc_dict = {}
            if os.path.exists(ablations_path):
                df_acc = pd.read_csv(ablations_path)
                baseline_acc = df_acc.loc[df_acc['Layer'] == 'Baseline', 'Accuracy'].values[0] if 'Baseline' in df_acc['Layer'].values else df_acc['Accuracy'].max()
                df_acc = df_acc[df_acc['Layer'] != 'Baseline'].copy()
                df_acc['Layer'] = df_acc['Layer'].astype(int)
                if baseline_acc != 0:
                    acc_dict = df_acc.set_index('Layer')['Accuracy'].apply(lambda x: (baseline_acc - x) / baseline_acc).to_dict()
                else:
                    acc_dict = df_acc.set_index('Layer')['Accuracy'].apply(lambda x: 0.0).to_dict()

            num_layers = len(acc_dict)
            if num_layers == 0: 
                continue

            m1_dict = load_metric(os.path.join(metrics_dir, "metric_01_MSE.csv"))
            m2_dict = load_metric(os.path.join(metrics_dir, "metric_02_Cosine_Distance.csv"))
            m3_dict = load_metric(os.path.join(metrics_dir, "metric_03_Residual_Contribution.csv"))
            m4_dict = load_metric(os.path.join(metrics_dir, "metric_04_CKA.csv"))
            m5_dict = load_metric(os.path.join(metrics_dir, "metric_05_L1_Distance.csv"))
            m6_dict = load_metric(os.path.join(metrics_dir, "metric_06_L_Infinity.csv"))
            m7_dict = load_metric(os.path.join(metrics_dir, "metric_07_Variance_Ratio.csv"))
            m8_dict = load_metric(os.path.join(metrics_dir, "metric_08_Pearson_Correlation.csv"))
            m10_dict = load_metric(os.path.join(metrics_datafree_dir, "metric_10_Router_Weights.csv"))
            m11_dict = load_metric(os.path.join(metrics_datafree_dir, "metric_11_SVD_Entropy.csv"))
            m12_dict = load_metric(os.path.join(metrics_dir, "metric_12_KL_noise.csv"))
            m13_dict = load_metric(os.path.join(metrics_dir, "metric_13_LogitLens.csv"))
            m14_dict = load_metric(os.path.join(metrics_datafree_dir, "metric_14_Spectral_Norm.csv"))
            m15_dict = load_metric(os.path.join(metrics_datafree_dir, "metric_15_Frobenius_Norm.csv"))
            m16_dict = load_metric(os.path.join(metrics_datafree_dir, "metric_16_Effective_Rank.csv"))

            def add_to_row(name, m_dict, row_obj, l_idx):
                if m_dict and l_idx in m_dict:
                    for suffix, val in m_dict[l_idx].items():
                        row_obj[f"{name}{suffix}"] = val

            for layer in range(num_layers):
                row = {
                    "Model": model,
                    "Task": task,
                    "Layer": layer,
                    "Relative_Depth": layer / (num_layers - 1) if num_layers > 1 else 0.0,
                    "Accuracy_Drop": acc_dict.get(layer, np.nan)
                }

                add_to_row("M1_MSE", m1_dict, row, layer)
                add_to_row("M2_Cosine", m2_dict, row, layer)
                add_to_row("M3_Residual", m3_dict, row, layer)
                add_to_row("M4_CKA", m4_dict, row, layer)
                add_to_row("M5_L1", m5_dict, row, layer)
                add_to_row("M6_L_Inf", m6_dict, row, layer)
                add_to_row("M7_Var_Ratio", m7_dict, row, layer)
                add_to_row("M8_Pearson", m8_dict, row, layer)
                add_to_row("M10_Router", m10_dict, row, layer)
                add_to_row("M11_SVD_Ent", m11_dict, row, layer)
                add_to_row("M12_KL_Noise", m12_dict, row, layer)
                add_to_row("M13_LogitLens", m13_dict, row, layer)
                add_to_row("M14_Spectral_Norm", m14_dict, row, layer)
                add_to_row("M15_Frobenius_Norm", m15_dict, row, layer)
                add_to_row("M16_Effective_Rank", m16_dict, row, layer)

                all_data.append(row)

    master_df = pd.DataFrame(all_data)
    master_df = master_df.sort_values(by=['Model', 'Task', 'Layer']).reset_index(drop=True)
    
    base_1d_metrics = [
        'M11_SVD_Ent', 'M12_KL_Noise', 'M13_LogitLens', 
        'M14_Spectral_Norm', 'M15_Frobenius_Norm', 'M16_Effective_Rank'
    ]
    
    for col in base_1d_metrics:
        if col in master_df.columns:
            master_df[f'{col}_delta1'] = master_df.groupby(['Model', 'Task'])[col].diff()
            master_df[f'{col}_delta2'] = master_df.groupby(['Model', 'Task'])[f'{col}_delta1'].diff()
            master_df[f'{col}_rel_change'] = master_df.groupby(['Model', 'Task'])[col].pct_change()

    pairwise_starts = [c for c in master_df.columns if c.endswith('_start')]
    for col in pairwise_starts:
        master_df[f'{col}_delta1'] = master_df.groupby(['Model', 'Task'])[col].diff()

    master_df = master_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    master_df = master_df.dropna(axis=1, how='all')
    
    out_path = os.path.join(SCRIPT_DIR, "master_dataset.csv")
    master_df.to_csv(out_path, index=False)
    print(f"Dataset compiled and cleaned. Final size: {master_df.shape}")

if __name__ == "__main__":
    main()
