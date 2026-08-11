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

    df = pd.read_csv("../dataset.csv")

    top_features = ['Relative_Depth',
                    'M1_MSE_next',
                    'F1_Rank_Normalized_Residual_delta1',
                    'M11_SVD_Ent',
                    'M1_MSE_local_mean',
                    'M18_Outlier_IoU',
                    'M17_Var_Shift',
                    'M8_Pearson_next',
                    'M10_Router_Router_Norm_Min']
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

    train_mask = df_trans['Task'] != 'copa'
    test_mask = df_trans['Task'] == 'copa'
    
    train_df = df_trans[train_mask].sort_values(['qid', 'Layer'])
    test_df = df_trans[test_mask].sort_values(['qid', 'Layer'])
    
    if len(train_df) == 0 or len(test_df) == 0:
        return
        
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
    
    print("\ntrain: All models, tasks excluding copa")
    print("test: All models, task copa")
    
    for model in test_df['Model'].unique():
        sub = test_df[test_df['Model'] == model]
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
        
        print(f"  Model: {model:<10} | Spearman: {spearman_val:.4f} | "
              f"HR-{k_budget} (20%): {hr_budget:.2%} | NDCG-1: {ndcg_1:.4f} | NDCG-{k_budget} (20%): {ndcg_budget:.4f}")

if __name__ == "__main__":
    main()

'''
Used features: ['Relative_Depth', 'M1_MSE_next', 'F1_Rank_Normalized_Residual_delta1', 'M11_SVD_Ent', 'M1_MSE_local_mean', 'M18_Outlier_IoU', 'M17_Var_Shift', 'M8_Pearson_next', 'M10_Router_Router_Norm_Min']

train: All models, tasks excluding copa
test: All models, task copa
  Model: Qwen       | Spearman: 0.4504 | HR-4 (20%): 25.00% | NDCG-1: 0.9328 | NDCG-4 (20%): 0.9074
  Model: gemma      | Spearman: 0.4479 | HR-6 (20%): 16.67% | NDCG-1: 0.8662 | NDCG-6 (20%): 0.9272
'''