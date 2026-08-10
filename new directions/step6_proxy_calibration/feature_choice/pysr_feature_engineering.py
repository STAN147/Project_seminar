import os
import pandas as pd
import numpy as np
import sympy as sp
from pysr import PySRRegressor
import warnings

warnings.filterwarnings("ignore")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "dataset.csv")
    df = pd.read_csv(dataset_path)
    selected_features = [
        'M5_L1_local_global_ratio',
        'M5_L1_local_mean',
        'M3_Residual_end',
        'M2_Cosine_end',
        'M16_Effective_Rank',
        'M13_LogitLens',
        'M17_Var_Shift',
        'M18_Outlier_IoU',
        'M19_Attn_MLP_Ratio',
        'Relative_Depth'
    ]
    
    features = [f for f in selected_features if f in df.columns]
    target = 'Accuracy_Drop'
    
    models = df['Model'].unique()
    
    for model_name in models:
        print(f"testing {model_name}")
        mask = (df['Model'] == model_name) & (df['Task'] != 'copa')
        sub_df = df[mask].dropna(subset=features + [target])
        X = sub_df[features].astype(float)
        y = sub_df[target].astype(float).values
        model = PySRRegressor(
            niterations=40,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=[
                "exp", 
                "abs", 
                "log_abs(x) = log(abs(x) + 1f-6)",
                "sqrt_abs(x) = sqrt(abs(x))"
            ],
            extra_sympy_mappings={
                "log_abs": lambda x: sp.log(sp.Abs(x) + 1e-6),
                "sqrt_abs": lambda x: sp.sqrt(sp.Abs(x))
            },
            model_selection="best",
            loss="loss(prediction, target) = (prediction - target)^2",
            random_state=42,
            verbosity=0
        )
        model.fit(X, y)
        print(f"\nTop equations for {model_name}:")
        equations = model.equations_
        top_eqs = equations.sort_values(by="score", ascending=False).head(5)
        
        for idx, row in top_eqs.iterrows():
            print(f"\nComplexity: {row['complexity']} | Loss: {row['loss']:.6f} | Score: {row['score']:.4f}")
            print(f"Formula: {row['equation']}")

if __name__ == "__main__":
    main()
