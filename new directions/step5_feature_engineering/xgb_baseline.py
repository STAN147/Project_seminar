import os
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore")

def apply_hybrid_transform(df, features):
    df_trans = df.copy()
    models = df['Model'].unique()
    rank_keywords = ['KL', 'Rank', 'Ent', 'LogitLens', 'L1']
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
    df = pd.read_csv("master_dataset.csv")
    top_features = [
        'M3_Residual_end',     # + Corr
        'M3_Residual_mean',    # + Corr
        'M16_Effective_Rank',  # - Corr
        'M13_LogitLens',       # - Corr
        'M5_L1_start',         # - Corr
        'M8_Pearson_end',      # - Corr
        'M2_Cosine_end',       # + Corr
        'M3_Residual_next'     # + Corr
    ]
    features = [f for f in top_features if f in df.columns]
    constraints = [1 if f in ['M16_Effective_Rank', 'M13_LogitLens', 'M5_L1_start', 'M8_Pearson_end'] else -1 for f in features]
    monotone_constraints = tuple(constraints)
    df_trans = apply_hybrid_transform(df, features)
    df_trans['qid'] = df_trans.groupby(['Model', 'Task']).ngroup()
    df_trans['Accuracy_Drop_deterministic'] = df_trans['Accuracy_Drop'] + df_trans['Layer'] * 1e-9

    def make_relevance(series):
        ranks = series.rank(method='first')
        return pd.qcut(ranks, q=10, labels=[9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
        
    df_trans['relevance'] = df_trans.groupby('qid')['Accuracy_Drop_deterministic'].transform(make_relevance).astype(int)

    experiments = [
        (['gemma'], 'Qwen'),
        (['Qwen'], 'gemma'),
        (['Qwen', 'gemma'], 'phi-tiny'),
        (['Qwen'], 'phi-tiny'),
        (['gemma'], 'phi-tiny')
    ]
    
    for train_models_list, test_model in experiments:
        train_mask = df_trans['Model'].isin(train_models_list) & (df_trans['Task'] != 'copa')
        test_mask = (df_trans['Model'] == test_model)
        train_df = df_trans[train_mask].sort_values(['qid', 'Layer'])
        test_df = df_trans[test_mask].sort_values(['qid', 'Layer'])
        if len(train_df) == 0 or len(test_df) == 0:
            continue
        X_train = train_df[features].fillna(0).astype(float)
        y_train = train_df['relevance']
        qid_train = train_df['qid']
        X_test = test_df[features].fillna(0).astype(float)
        ranker = xgb.XGBRanker(
            tree_method="hist",
            device="cuda",
            objective="rank:pairwise",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=2.0,
            monotone_constraints=monotone_constraints,
            random_state=42
        )
        ranker.fit(X_train, y_train, qid=qid_train, verbose=False)
        preds = ranker.predict(X_test)
        test_df['pred_score'] = preds
        print(f"train: {train_models_list}")
        print(f"test: {test_model}")
        for task in test_df['Task'].unique():
            sub = test_df[test_df['Task'] == task]
            if len(sub) < 3: continue
            corr, _ = spearmanr(sub['pred_score'], sub['Accuracy_Drop'])
            spearman_val = abs(corr)
            def get_hit_rate(k_val):
                k_val = max(1, min(int(k_val), len(sub) - 1))
                t_idx = sub.nsmallest(k_val, 'Accuracy_Drop').index
                p_idx = sub.nlargest(k_val, 'pred_score').index
                return len(set(t_idx).intersection(set(p_idx))) / k_val
            preds_desc = np.sort(sub['pred_score'].values)[::-1]

            diffs = np.abs(np.diff(preds_desc))
            k_jump = np.argmax(diffs) + 1  
            hr_jump = get_hit_rate(k_jump)

            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(preds_desc.reshape(-1, 1))
            centers = kmeans.cluster_centers_.flatten()
            drop_cluster_idx = np.argmax(centers) 
            k_kmeans = np.sum(labels == drop_cluster_idx)
            hr_kmeans = get_hit_rate(k_kmeans)
            
            print(f"  Задача {task:<5} | Spearman: {spearman_val:.4f} | "
                  f"Hit Rate Max Jump: {hr_jump:.2%} | "
                  f"Hit Rate K-Means: {hr_kmeans:.2%}")

if __name__ == "__main__":
    main()
