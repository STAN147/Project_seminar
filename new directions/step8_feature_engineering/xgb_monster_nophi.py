import os
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore")

def apply_hybrid_transform(df, features):
    """
    Гибридная нормализация:
    - Перцентильный ранг для метрик с тяжелыми хвостами (KL, Энтропии, Ранги, L1).
    - Z-Score для ограниченных метрик и дистанций (Residuals, Cosine, Pearson).
    """
    df_trans = df.copy()
    models = df['Model'].unique()
    
    rank_keywords = ['KL', 'Rank', 'Ent', 'LogitLens', 'L1']
    rank_features = [f for f in features if any(k in f for k in rank_keywords)]
    z_features = [f for f in features if f not in rank_features]
    
    for m in models:
        mask = df_trans['Model'] == m
        
        # 1. Percentile Rank
        for f in rank_features:
            if f in df_trans.columns:
                df_trans.loc[mask, f] = df_trans.loc[mask, f].rank(pct=True, method='average')
                
        # 2. Z-Score
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
    dataset_path = "master_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"Ошибка: Файл {dataset_path} не найден.")
        return

    df = pd.read_csv(dataset_path)
    
    # ---------------------------------------------------------
    # ПЛАТИНОВЫЙ ФОНД (Только стабильные инварианты)
    # ---------------------------------------------------------
    platinum_features = [
        'M3_Residual_end',     # + Corr
        'M12_KL_Noise',        # + Corr
        'M3_Residual_mean',    # + Corr
        'M16_Effective_Rank',  # - Corr
        'M13_LogitLens',       # - Corr
        'M5_L1_start',         # - Corr
        'M8_Pearson_end',      # - Corr
        'M2_Cosine_end',       # + Corr
        'M3_Residual_next'     # + Corr
    ]
    
    features = [f for f in platinum_features if f in df.columns]
    
    print("="*105)
    print("XGBOOST: ПЛАТИНОВЫЙ ФОНД + ДИНАМИЧЕСКИЕ ПОРОГИ (ABLATION STUDY)")
    print("="*105 + "\n")

    # Формируем монотонные ограничения строго по знаку корреляции
    constraints = [1 if f in ['M16_Effective_Rank', 'M13_LogitLens', 'M5_L1_start', 'M8_Pearson_end'] else -1 for f in features]
    monotone_constraints = tuple(constraints)

    # Применяем нормализацию (phi-tiny тоже нормализуется в рамках своей архитектуры)
    df_trans = apply_hybrid_transform(df, features)
    df_trans['qid'] = df_trans.groupby(['Model', 'Task']).ngroup()
    
    # Таргет: 4 = Избыточный слой (самый безопасный), 0 = Критический
    def make_relevance(series):
        ranks = series.rank(method='first')
        return pd.qcut(ranks, q=5, labels=[4, 3, 2, 1, 0])
        
    df_trans['relevance'] = df_trans.groupby('qid')['Accuracy_Drop'].transform(make_relevance).astype(int)

    # Явно задаем сценарии тестирования (Train Models, Test Model, Test Name)
    experiments = [
        (['gemma'], 'Qwen', 'Zero-Shot'),
        (['Qwen'], 'gemma', 'Zero-Shot'),
        (['Qwen', 'gemma'], 'phi-tiny', 'OOD UNICORN (Joint Train)'),
        (['Qwen'], 'phi-tiny', 'OOD UNICORN (Train: Qwen only)'),
        (['gemma'], 'phi-tiny', 'OOD UNICORN (Train: Gemma only)')
    ]
    
    for train_models_list, test_model, test_name in experiments:
        # В трейн идут только указанные модели, COPA по-прежнему исключена из обучения
        train_mask = df_trans['Model'].isin(train_models_list) & (df_trans['Task'] != 'copa')
        test_mask = (df_trans['Model'] == test_model)
        
        train_df = df_trans[train_mask].sort_values('qid')
        test_df = df_trans[test_mask].sort_values('qid')
        
        if len(train_df) == 0 or len(test_df) == 0:
            continue
            
        X_train = train_df[features].fillna(0).astype(float)
        y_train = train_df['relevance']
        qid_train = train_df['qid']
        
        X_test = test_df[features].fillna(0).astype(float)
        
        ranker = xgb.XGBRanker(
            tree_method="hist",
            device="cuda",
            objective="rank:ndcg",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            monotone_constraints=monotone_constraints,
            random_state=42
        )
        
        try:
            ranker.fit(X_train, y_train, qid=qid_train, verbose=False)
        except Exception as e:
            ranker.set_params(device="cpu")
            ranker.fit(X_train, y_train, qid=qid_train, verbose=False)
            
        preds = ranker.predict(X_test)
        test_df['pred_score'] = preds
        
        print(f"ОБУЧЕНИЕ НА: {train_models_list} (CSQA, SIQA)")
        print(f"ТЕСТ НА ({test_name}): {test_model}")
        print("-" * 105)
        
        for task in test_df['Task'].unique():
            sub = test_df[test_df['Task'] == task]
            if len(sub) < 3: continue
            
            # --- 0. Базовые метрики ---
            corr, _ = spearmanr(sub['pred_score'], sub['Accuracy_Drop'])
            spearman_val = abs(corr)
            
            def get_hit_rate(k_val):
                k_val = max(1, min(int(k_val), len(sub) - 1))
                t_idx = sub.nsmallest(k_val, 'Accuracy_Drop').index
                p_idx = sub.nlargest(k_val, 'pred_score').index
                return len(set(t_idx).intersection(set(p_idx))) / k_val
            
            preds_desc = np.sort(sub['pred_score'].values)[::-1]
            
            # --- 1. Классика: Top-20% ---
            k_20 = int(0.2 * len(sub))
            hr_20 = get_hit_rate(k_20)
            
            # --- 2. Метод Максимального Скачка (Max Jump) ---
            diffs = np.abs(np.diff(preds_desc))
            k_jump = np.argmax(diffs) + 1  
            hr_jump = get_hit_rate(k_jump)
            
            # --- 3. Метод 1D K-Means (2 кластера) ---
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(preds_desc.reshape(-1, 1))
            centers = kmeans.cluster_centers_.flatten()
            drop_cluster_idx = np.argmax(centers) 
            k_kmeans = np.sum(labels == drop_cluster_idx)
            hr_kmeans = get_hit_rate(k_kmeans)
            
            print(f"  Задача {task.upper():<5} | Spearman: {spearman_val:.4f} | "
                  f"Hit Rate Top-20%: {hr_20:.2%} | "
                  f"Hit Rate Max Jump: {hr_jump:.2%} | "
                  f"Hit Rate K-Means: {hr_kmeans:.2%}")
        
        print("="*105 + "\n")

if __name__ == "__main__":
    main()
