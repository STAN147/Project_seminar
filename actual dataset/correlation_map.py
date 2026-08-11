import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

def main():
    dataset_path = "dataset.csv"
    df = pd.read_csv(dataset_path)
    features = [c for c in df.columns if c.startswith('M') and ('_' in c or 'delta' in c)]
    target = 'Accuracy_Drop'
    models = df['Model'].unique() 
    results = []
    for f in features:
        row = {'Feature': f}
        abs_corrs = []
        for m in models:
            mask = df['Model'] == m
            sub_df = df[mask]
            if len(sub_df) > 1 and f in sub_df.columns:
                corr, _ = spearmanr(sub_df[f], sub_df[target])
                if np.isnan(corr):
                    corr = 0.0
                row[m] = corr
                abs_corrs.append(abs(corr))
            else:
                row[m] = 0.0
        row['Mean_Abs_Corr'] = np.mean(abs_corrs) if abs_corrs else 0.0
        results.append(row)
    corr_df = pd.DataFrame(results)
    corr_df = corr_df.sort_values(by='Mean_Abs_Corr', ascending=False).reset_index(drop=True)
    print(f"\n{'Feature':<35} | " + " | ".join([f"{m:<10}" for m in models]) + " | Mean |Corr|")
    for _, row in corr_df.head(20).iterrows():
        feature_name = row['Feature'][:33]
        model_corrs = " | ".join([f"{row[m]:+10.4f}" for m in models])
        mean_abs = f"{row['Mean_Abs_Corr']:.4f}"
        print(f"{feature_name:<35} | {model_corrs} | {mean_abs}")

if __name__ == "__main__":
    main()
