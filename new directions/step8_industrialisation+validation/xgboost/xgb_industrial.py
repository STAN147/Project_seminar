import os
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
import warnings

warnings.filterwarnings("ignore")

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
    neg_corr = ['F3_NonLinear_Depth', 'M5_L1_local_global_ratio', 'M5_L1_local_mean', 'M16_Effective_Rank', 'M13_LogitLens']
    constraints = [1 if f in neg_corr else -1 for f in features]
    monotone_constraints = tuple(constraints)
    df_trans = apply_hybrid_transform(df, features)
    df_trans['qid'] = df_trans.groupby(['Model', 'Task']).ngroup()
    df_trans['Accuracy_Drop_deterministic'] = df_trans['Accuracy_Drop'] + df_trans['Layer'] * 1e-9

    def make_relevance(series):
        ranks = series.rank(method='first')
        return pd.qcut(ranks, q=10, labels=[9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
        
    df_trans['relevance'] = df_trans.groupby('qid')['Accuracy_Drop_deterministic'].transform(make_relevance).astype(int)

    experiments = [
        (['gemma', 'phi-tiny', 'llama'], 'Qwen'),
        (['Qwen', 'phi-tiny', 'llama'], 'gemma'),
        (['Qwen', 'gemma', 'llama'], 'phi-tiny'),
        (['Qwen', 'gemma', 'phi-tiny'], 'llama')
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
        
        print(f"\ntrain: {train_models_list}")
        print(f"test: {test_model}")
        
        for task in test_df['Task'].unique():
            sub = test_df[test_df['Task'] == task]
            if len(sub) < 3: continue
            
            corr, _ = spearmanr(sub['pred_score'], sub['Accuracy_Drop'])
            spearman_val = abs(corr)
            
            k_budget = max(1, int(len(sub) * 0.20))
            
            t_idx = sub.nsmallest(k_budget, 'Accuracy_Drop').index
            p_idx = sub.nlargest(k_budget, 'pred_score').index
            hr_budget = len(set(t_idx).intersection(set(p_idx))) / k_budget
            
            true_relevance = -sub['Accuracy_Drop'].values
            true_relevance = true_relevance - np.min(true_relevance) 
            pred_scores = sub['pred_score'].values
            
            ndcg_1 = ndcg_score([true_relevance], [pred_scores], k=1)
            ndcg_budget = ndcg_score([true_relevance], [pred_scores], k=k_budget)
            
            print(f"   Task: {task:<5} | Spearman: {spearman_val:.4f} | "
                  f"HR-{k_budget} (20%): {hr_budget:.2%} | NDCG-1: {ndcg_1:.4f} | NDCG-{k_budget} (20%): {ndcg_budget:.4f}")

if __name__ == "__main__":
    main()

'''
train: ['gemma', 'phi-tiny', 'llama']
test: Qwen
   Task: csqa  | Spearman: 0.8122 | HR-4 (20%): 50.00% | NDCG-1: 0.9945 | NDCG-4 (20%): 0.9715
   Task: siqa  | Spearman: 0.8450 | HR-4 (20%): 25.00% | NDCG-1: 0.9763 | NDCG-4 (20%): 0.9760

train: ['Qwen', 'phi-tiny', 'llama']
test: gemma
   Task: csqa  | Spearman: 0.7726 | HR-6 (20%): 33.33% | NDCG-1: 0.9904 | NDCG-6 (20%): 0.9899
   Task: siqa  | Spearman: 0.7492 | HR-6 (20%): 33.33% | NDCG-1: 0.9818 | NDCG-6 (20%): 0.9900

train: ['Qwen', 'gemma', 'llama']
test: phi-tiny
   Task: csqa  | Spearman: 0.8319 | HR-6 (20%): 50.00% | NDCG-1: 0.9803 | NDCG-6 (20%): 0.9870
   Task: siqa  | Spearman: 0.7342 | HR-6 (20%): 33.33% | NDCG-1: 0.9722 | NDCG-6 (20%): 0.9832

train: ['Qwen', 'gemma', 'phi-tiny']
test: llama
   Task: csqa  | Spearman: 0.7955 | HR-5 (20%): 40.00% | NDCG-1: 0.9472 | NDCG-5 (20%): 0.9485
   Task: siqa  | Spearman: 0.8060 | HR-5 (20%): 40.00% | NDCG-1: 0.9383 | NDCG-5 (20%): 0.9761
'''