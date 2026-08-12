import os
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

def apply_hybrid_transform(df, features):
    df_trans = df.copy()
    models = df['Model'].unique()
    rank_keywords = ['KL', 'Rank', 'Ent', 'LogitLens', 'L1', 'F2', 'F3']
    rank_features = [f for f in features if any(k in f for k in rank_keywords)]
    z_features = [f for f in features if f not in rank_features]
    for m in models:
        mask = df_trans['Model'] == m
        for f in rank_features:
            if f in df_trans.columns:
                df_trans.loc[mask, f] = df_trans.loc[mask, f].rank(pct=True, method='average')
        for f in z_features:
            if f in df_trans.columns:
                mean = df_trans.loc[mask, f].mean()
                std = df_trans.loc[mask, f].std()
                if std > 1e-6:
                    df_trans.loc[mask, f] = (df_trans.loc[mask, f] - mean) / std
                else:
                    df_trans.loc[mask, f] = 0.0
    return df_trans

def main():
    df = pd.read_csv("../dataset.csv")
    top_features = [
        'F3_NonLinear_Depth',
        'F2_Depth_Penalized_Cosine',
        'F1_Rank_Normalized_Residual',
        'M3_Residual_end',
        'M5_L1_local_global_ratio',
        'M5_L1_local_mean',
        'M16_Effective_Rank',
        'M17_Var_Shift',
        'M18_Outlier_IoU',
        'M11_SVD_Ent'
    ]
    features = [f for f in top_features if f in df.columns]
    df_trans = apply_hybrid_transform(df, features)
    df_trans['qid'] = df_trans.groupby(['Model', 'Task']).ngroup()
    df_trans['Accuracy_Drop_deterministic'] = df_trans['Accuracy_Drop'] + df_trans['Layer'] * 1e-9

    def make_relevance(series):
        ranks = series.rank(method='first')
        return pd.qcut(ranks, q=10, labels=[9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

    df_trans['relevance'] = df_trans.groupby('qid')['Accuracy_Drop_deterministic'].transform(make_relevance).astype(int)
    train_mask = (df_trans['Task'] != 'copa') & (~df_trans['Model'].str.lower().isin(['tinyllama']))
    test_mask = df_trans['Task'] == 'copa'
    train_df = df_trans[train_mask].sort_values(['qid', 'Layer'])
    test_df = df_trans[test_mask].sort_values(['qid', 'Layer'])
    X_train = train_df[features].fillna(0).astype(float)
    y_train = train_df['relevance']
    qid_train = train_df['qid']
    X_test = test_df[features].fillna(0).astype(float)
    ranker = xgb.XGBRanker(
        tree_method="hist",
        device="cuda",
        objective="rank:pairwise",
        n_estimators=300,
        learning_rate=0.10,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.01,
        reg_lambda=1.00,
        random_state=42
    )
    try:
        ranker.fit(X_train, y_train, qid=qid_train, verbose=False)
    except Exception:
        ranker.set_params(device="cpu")
        ranker.fit(X_train, y_train, qid=qid_train, verbose=False)
    preds = ranker.predict(X_test)
    test_df['pred_score'] = preds
    for model in test_df['Model'].unique():
        sub = test_df[test_df['Model'] == model]
        if len(sub) < 3: 
            continue
        corr, _ = spearmanr(sub['pred_score'], sub['Accuracy_Drop'])
        spearman_val = abs(corr) if not np.isnan(corr) else 0.0
        k_budget = max(1, int(len(sub) * 0.20))
        t_idx = sub.nsmallest(k_budget, 'Accuracy_Drop').index
        p_idx = sub.nlargest(k_budget, 'pred_score').index
        hr_budget = len(set(t_idx).intersection(set(p_idx))) / k_budget
        true_relevance = -sub['Accuracy_Drop'].values
        true_relevance = true_relevance - np.min(true_relevance) 
        pred_scores = sub['pred_score'].values
        ndcg_1 = ndcg_score([true_relevance], [pred_scores], k=1)
        ndcg_budget = ndcg_score([true_relevance], [pred_scores], k=k_budget)
        print(f"  Model: {model:<10} | Task: copa  | Spearman: {spearman_val:.4f} | "
              f"HR-{k_budget} (20%): {hr_budget:.2%} | NDCG-1: {ndcg_1:.4f} | NDCG-{k_budget} (20%): {ndcg_budget:.4f}")

if __name__ == "__main__":
    main()
