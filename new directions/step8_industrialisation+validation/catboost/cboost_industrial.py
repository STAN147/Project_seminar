import os
import pandas as pd
import numpy as np
import catboost as cb
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
import warnings

warnings.filterwarnings("ignore")

def apply_hybrid_transform(df, features):
    df_trans = df.copy()
    models = df['Model'].unique()
    for m in models:
        mask = df_trans['Model'] == m
        for f in features:
            if f in df_trans.columns:
                df_trans.loc[mask, f] = df_trans.loc[mask, f].rank(pct=True, method='average')
    return df_trans

def main():
    dataset_path = os.path.join("..", "dataset.csv")
    df = pd.read_csv(dataset_path)

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
    
    constraints_dict = None

    print("Used features:", features)

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
        
        train_pool = cb.Pool(data=X_train, label=y_train, group_id=qid_train)
        
        ranker = cb.CatBoostRanker(
            loss_function='QuerySoftMax',
            iterations=300,
            learning_rate=0.05,
            depth=3,
            l2_leaf_reg=1.0,
            monotone_constraints=constraints_dict,
            random_seed=42,
            verbose=False
        )
        
        ranker.fit(train_pool)
        preds = ranker.predict(X_test)
        test_df['pred_score'] = preds
        
        print(f"\ntrain: {train_models_list}")
        print(f"test: {test_model}")
        
        for task in test_df['Task'].unique():
            sub = test_df[test_df['Task'] == task]
            if len(sub) < 3: continue
            
            corr, _ = spearmanr(sub['pred_score'], sub['Accuracy_Drop'])
            spearman_val = abs(corr) if not np.isnan(corr) else 0.0
            k_budget = max(1, int(len(sub) * 0.20))
            t_idx = sub.nsmallest(k_budget, 'Accuracy_Drop').index
            p_idx = sub.nlargest(k_budget, 'pred_score').index
            hr_budget = len(set(t_idx).intersection(set(p_idx))) / k_budget
            true_relevance = -sub['Accuracy_Drop'].values
            true_relevance = true_relevance - np.min(true_relevance) 
            pred_scores = sub['pred_score'].values
            
            ndcg_budget = ndcg_score([true_relevance], [pred_scores], k=k_budget)
            ndcg_1 = ndcg_score([true_relevance], [pred_scores], k=1)
            
            print(f"  Task: {task:<5} | Spearman: {spearman_val:.4f} | "
                  f"HR-{k_budget} (20%): {hr_budget:.2%} | NDCG-1: {ndcg_1:.4f} | NDCG-{k_budget} (20%): {ndcg_budget:.4f}")

if __name__ == "__main__":
    main()

'''
train: ['gemma', 'phi-tiny', 'llama']
test: Qwen
  Task: csqa  | Spearman: 0.7817 | HR-4 (20%): 50.00% | NDCG-1: 0.9917 | NDCG-4 (20%): 0.9760
  Task: siqa  | Spearman: 0.7863 | HR-4 (20%): 25.00% | NDCG-1: 0.9763 | NDCG-4 (20%): 0.9734

train: ['Qwen', 'phi-tiny', 'llama']
test: gemma
  Task: csqa  | Spearman: 0.7775 | HR-6 (20%): 33.33% | NDCG-1: 0.9904 | NDCG-6 (20%): 0.9884
  Task: siqa  | Spearman: 0.8064 | HR-6 (20%): 66.67% | NDCG-1: 0.9818 | NDCG-6 (20%): 0.9926

train: ['Qwen', 'gemma', 'llama']
test: phi-tiny
  Task: csqa  | Spearman: 0.8603 | HR-6 (20%): 66.67% | NDCG-1: 0.9864 | NDCG-6 (20%): 0.9900
  Task: siqa  | Spearman: 0.7429 | HR-6 (20%): 33.33% | NDCG-1: 0.9722 | NDCG-6 (20%): 0.9808

train: ['Qwen', 'gemma', 'phi-tiny']
test: llama
  Task: csqa  | Spearman: 0.8464 | HR-5 (20%): 40.00% | NDCG-1: 0.8393 | NDCG-5 (20%): 0.9369
  Task: siqa  | Spearman: 0.7715 | HR-5 (20%): 20.00% | NDCG-1: 0.9383 | NDCG-5 (20%): 0.9748
'''