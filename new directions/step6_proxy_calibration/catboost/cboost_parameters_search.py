import os
import pandas as pd
import numpy as np
import catboost as cb
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
from itertools import product
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
    dataset_path = os.path.join("..", "dataset.csv")
    df = pd.read_csv(dataset_path)
    
    top_features = [
        'F3_NonLinear_Depth',
        'F2_Depth_Penalized_Cosine',
        'F1_Rank_Normalized_Residual',
        'M5_L1_local_global_ratio',
        'M5_L1_local_mean',
        'M3_Residual_end',
        'M13_LogitLens',
        'M16_Effective_Rank',
        'M17_Var_Shift',
        'M18_Outlier_IoU'
    ]
    features = [f for f in top_features if f in df.columns]
    neg_corr = ['F3_NonLinear_Depth', 'M5_L1_local_global_ratio', 'M5_L1_local_mean', 'M16_Effective_Rank', 'M13_LogitLens']
    
    constraints_dict = {f: (1 if f in neg_corr else -1) for f in features}
    
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
    
    losses = ['YetiRank', 'PairLogit', 'QueryRMSE']
    depths = [2, 3, 4, 5]
    l2_regs = [0.01, 0.1, 0.5, 1.0, 3.0, 5.0, 10.0]
    lrs = [0.01, 0.05, 0.1]
    
    results = []

    for idx, (loss, d, l2, lr) in enumerate(product(losses, depths, l2_regs, lrs)):
        ndcg_scores = []
        spearman_scores = []
        
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
                loss_function=loss,
                iterations=300,
                learning_rate=lr,
                depth=d,
                l2_leaf_reg=l2,
                monotone_constraints=constraints_dict,
                random_seed=42,
                verbose=False
            )
            
            ranker.fit(train_pool)
            preds = ranker.predict(X_test)
            test_df['pred_score'] = preds
            
            for task in test_df['Task'].unique():
                sub = test_df[test_df['Task'] == task]
                if len(sub) < 3: continue
                
                corr, _ = spearmanr(sub['pred_score'], sub['Accuracy_Drop'])
                spearman_scores.append(abs(corr))
                
                k_budget = max(1, int(len(sub) * 0.20))
                
                true_relevance = -sub['Accuracy_Drop'].values
                true_relevance = true_relevance - np.min(true_relevance) 
                pred_scores = sub['pred_score'].values
                
                ndcg_budget = ndcg_score([true_relevance], [pred_scores], k=k_budget)
                ndcg_scores.append(ndcg_budget)

        mean_ndcg = np.mean(ndcg_scores)
        mean_spearman = np.mean(spearman_scores)
        results.append({
            'loss': loss, 
            'depth': d, 
            'l2': l2, 
            'lr': lr,
            'mean_ndcg': mean_ndcg, 
            'mean_spearman': mean_spearman
        })

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(by=['mean_ndcg', 'mean_spearman'], ascending=[False, False]).reset_index(drop=True)
    
    print(f"{'Rank':<5} | {'Loss':<12} | {'Depth':<5} | {'L2':<5} | {'LR':<5} | {'Mean NDCG-20%':<15} | {'Mean Spearman':<15}")
    
    for i, row in res_df.head(10).iterrows():
        print(f"{i+1:<5} | {row['loss']:<12} | {row['depth']:<5.0f} | {row['l2']:<5.2f} | {row['lr']:<5.2f} | {row['mean_ndcg']:<15.4f} | {row['mean_spearman']:<15.4f}")

if __name__ == "__main__":
    main()
