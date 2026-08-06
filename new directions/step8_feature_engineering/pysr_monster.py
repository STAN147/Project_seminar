import os
import pandas as pd
import numpy as np
from pysr import PySRRegressor
from scipy.stats import spearmanr
import warnings

warnings.filterwarnings("ignore")

def apply_hybrid_transform(df, features):
    """
    Гибридная нормализация для Платинового фонда:
    - Перцентильный ранг для L1, KL, Rank, LogitLens.
    - Z-Score для Residuals, Pearson, Cosine.
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
    
    # ПЛАТИНОВЫЙ ФОНД
    platinum_features = [
        'M3_Residual_end',     
        'M12_KL_Noise',        
        'M3_Residual_mean',    
        'M16_Effective_Rank',  
        'M13_LogitLens',       
        'M5_L1_start',         
        'M8_Pearson_end',      
        'M2_Cosine_end',       
        'M3_Residual_next'     
    ]
    
    features = [f for f in platinum_features if f in df.columns]
    target = 'Accuracy_Drop'

    df_trans = apply_hybrid_transform(df, features)
    models = df_trans['Model'].unique().tolist()
    
    print("="*60)
    print("PySR: ПЛАТИНОВЫЙ ФОНД + ГИБРИДНАЯ НОРМАЛИЗАЦИЯ")
    print("="*60)

    for test_model in models:
        # Трейн БЕЗ copa
        train_mask = (df_trans['Model'] != test_model) & (df_trans['Task'] != 'copa')
        test_mask = (df_trans['Model'] == test_model)
        
        train_models_list = df_trans.loc[train_mask, 'Model'].unique().tolist()
        
        X_train = df_trans.loc[train_mask, features].fillna(0).astype(float)
        y_train = df.loc[train_mask, target].fillna(0).astype(float) 
        
        X_test = df_trans.loc[test_mask, features].fillna(0).astype(float)
        y_test = df.loc[test_mask, target].fillna(0).astype(float)
        test_tasks = df.loc[test_mask, 'Task']

        print(f"\n--- ОБУЧЕНИЕ НА: {train_models_list} (CSQA, SIQA) ---")
        print(f"--- ТЕСТ НА (Zero-Shot): {test_model} ---")

        # Настраиваем PySR на простые, интерпретируемые уравнения
        model = PySRRegressor(
            procs="auto",
            niterations=150,           # Чуть больше итераций, так как фичей меньше
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["exp"],   # Оставляем exp для нелинейных порогов
            select_k_features=5,       # Заставляем выбрать Топ-5 из 9 платиновых
            model_selection="best",    
            loss="loss(prediction, target) = (prediction - target)^2", 
            random_state=42,
            verbosity=0
        )

        try:
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"Ошибка при обучении на фолде {test_model}: {e}")
            continue

        print("\nЛУЧШАЯ НАЙДЕННАЯ ФОРМУЛА:")
        print(model.sympy())

        preds = model.predict(X_test)
        
        print(f"\nРЕЗУЛЬТАТЫ ПО ЗАДАЧАМ:")
        for task in test_tasks.unique():
            task_mask = test_tasks == task
            y_test_task = y_test[task_mask]
            preds_task = preds[task_mask]
            
            if len(y_test_task) < 2:
                continue
                
            spearman_corr, _ = spearmanr(preds_task, y_test_task)
            
            k = max(1, int(0.2 * len(y_test_task)))
            true_idx = y_test_task.nsmallest(k).index
            pred_idx = pd.Series(preds_task, index=y_test_task.index).nsmallest(k).index
            hit_rate = len(set(true_idx).intersection(set(pred_idx))) / k
            
            print(f"  {task.upper():<5} -> Spearman: {spearman_corr:+.4f} | Hit Rate (Top-20%): {hit_rate:.2%}")
        
        print("="*60)

if __name__ == "__main__":
    main()
