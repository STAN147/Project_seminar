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
    df = pd.read_csv("baseline_dataset.csv")
    top_features = [
        'M5_L1_start',
        'M3_Residual_end',
        'M12_KL_Noise',
        'M2_Cosine_end',
        'M8_Pearson_end',
        'M3_Residual_mean',
        'M5_L1_mean',
        'M16_Effective_Rank'
    ]
    
    features = [f for f in top_features if f in df.columns]
    df_trans = apply_hybrid_transform(df, features)
    df_trans['qid'] = df_trans.groupby(['Model', 'Task']).ngroup()
    df_trans['Accuracy_Drop_deterministic'] = df_trans['Accuracy_Drop'] + df_trans['Layer'] * 1e-9

    def make_relevance(series):
        ranks = series.rank(method='first')
        return pd.qcut(ranks, q=10, labels=[9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
        
    df_trans['relevance'] = df_trans.groupby('qid')['Accuracy_Drop_deterministic'].transform(make_relevance).astype(int)

    experiments = [
        (['gemma', 'phi-tiny'], 'Qwen'),
        (['Qwen', 'phi-tiny'], 'gemma'),
        (['Qwen', 'gemma'], 'phi-tiny')
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
            random_state=42
        )
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
            
            ndcg_budget = ndcg_score([true_relevance], [pred_scores], k=k_budget)
            
            print(f"  Task: {task:<5} | Spearman: {spearman_val:.4f} | "
                  f"HR-{k_budget} (20%): {hr_budget:.2%} | NDCG-{k_budget} (20%): {ndcg_budget:.4f}")

if __name__ == "__main__":
    main()

'''
train: ['gemma', 'phi-tiny']
test: Qwen
  Task: csqa  | Spearman: 0.7426 | HR-4 (20%): 50.00% | NDCG-4 (20%): 0.9714
  Task: siqa  | Spearman: 0.7410 | HR-4 (20%): 50.00% | NDCG-4 (20%): 0.9894

train: ['Qwen', 'phi-tiny']
test: gemma
  Task: csqa  | Spearman: 0.7240 | HR-6 (20%): 16.67% | NDCG-6 (20%): 0.9861
  Task: siqa  | Spearman: 0.7766 | HR-6 (20%): 50.00% | NDCG-6 (20%): 0.9904

train: ['Qwen', 'gemma']
test: phi-tiny
  Task: csqa  | Spearman: 0.8481 | HR-6 (20%): 66.67% | NDCG-6 (20%): 0.9905
  Task: siqa  | Spearman: 0.6135 | HR-6 (20%): 16.67% | NDCG-6 (20%): 0.9693
'''