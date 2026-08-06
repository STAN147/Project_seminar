import os
import pandas as pd
import numpy as np
from pysr import PySRRegressor
from scipy.stats import spearmanr
import warnings

warnings.filterwarnings("ignore")

def rank_transform_per_model(df, features):
    """
    Перцентильное ранжирование внутри каждой модели для инвариантности к масштабу.
    """
    df_rank = df.copy()
    models = df['Model'].unique()
    for m in models:
        mask = df_rank['Model'] == m
        for f in features:
            if f in df_rank.columns:
                df_rank.loc[mask, f] = df_rank.loc[mask, f].rank(pct=True, method='average')
    return df_rank

def main():
    dataset_path = "master_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"Ошибка: Файл {dataset_path} не найден.")
        return

    df = pd.read_csv(dataset_path)
    
    # Изолируем Data-Free и KL метрики
    target_prefixes = tuple(f"M{i}" for i in range(11, 17))
    features = [c for c in df.columns if c.startswith(target_prefixes)]
    target = 'Accuracy_Drop'

    df_norm = rank_transform_per_model(df, features)
    models = df_norm['Model'].unique().tolist()
    
    print(f"Отобрано Data-Free и KL признаков: {len(features)}")
    print("Исключаем задачу 'copa' из обучающей выборки для чистоты MSE-лосса.\n")
    print("="*60)

    for test_model in models:
        # Трейн: все модели кроме тестовой, и строго БЕЗ copa
        train_mask = (df_norm['Model'] != test_model) & (df_norm['Task'] != 'copa')
        test_mask = (df_norm['Model'] == test_model)
        
        train_models_list = df_norm.loc[train_mask, 'Model'].unique().tolist()
        
        X_train = df_norm.loc[train_mask, features].fillna(0).astype(float)
        y_train = df.loc[train_mask, target].fillna(0).astype(float) 
        
        X_test = df_norm.loc[test_mask, features].fillna(0).astype(float)
        y_test = df.loc[test_mask, target].fillna(0).astype(float)
        test_tasks = df.loc[test_mask, 'Task']

        print(f"\n--- ОБУЧЕНИЕ НА: {train_models_list} (CSQA, SIQA) ---")
        print(f"--- ТЕСТ НА (Zero-Shot): {test_model} ---")

        model = PySRRegressor(
            procs="auto",
            niterations=100,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["exp"],   
            select_k_features=7,
            model_selection="best",    
            loss="loss(prediction, target) = (prediction - target)^2", 
            random_state=42,
            verbosity=0  # Выключим консольный спам, чтобы видеть только итоги фолдов
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

'''
Отобрано Data-Free и KL признаков: 24
Исключаем задачу 'copa' из обучающей выборки для чистоты MSE-лосса.

============================================================

--- ОБУЧЕНИЕ НА: ['gemma', 'phi-tiny'] (CSQA, SIQA) ---
--- ТЕСТ НА (Zero-Shot): Qwen ---
Using features ['M11_SVD_Ent' 'M12_KL_Noise' 'M13_LogitLens' 'M15_Frobenius_Norm'
 'M16_Effective_Rank' 'M11_SVD_Ent_rel_change' 'M13_LogitLens_rel_change']

ЛУЧШАЯ НАЙДЕННАЯ ФОРМУЛА:
M12_KL_Noise*M13_LogitLens_rel_change*(0.036937412/M16_Effective_Rank + 0.08265989/M13_LogitLens)

РЕЗУЛЬТАТЫ ПО ЗАДАЧАМ:
  COPA  -> Spearman: +0.4182 | Hit Rate (Top-20%): 0.00%
  CSQA  -> Spearman: +0.7322 | Hit Rate (Top-20%): 75.00%
  SIQA  -> Spearman: +0.5410 | Hit Rate (Top-20%): 25.00%
============================================================

--- ОБУЧЕНИЕ НА: ['Qwen', 'phi-tiny'] (CSQA, SIQA) ---
--- ТЕСТ НА (Zero-Shot): gemma ---
Using features ['M11_SVD_Ent' 'M12_KL_Noise' 'M13_LogitLens' 'M14_Spectral_Norm'
 'M15_Frobenius_Norm' 'M16_Effective_Rank' 'M12_KL_Noise_delta2']

ЛУЧШАЯ НАЙДЕННАЯ ФОРМУЛА:
M12_KL_Noise*exp(M16_Effective_Rank*(M11_SVD_Ent + M12_KL_Noise_delta2/(M12_KL_Noise - (M11_SVD_Ent + M16_Effective_Rank) - 1*0.62668854) + M15_Frobenius_Norm - exp(M13_LogitLens/((exp(M15_Frobenius_Norm)/7.0197806)))))

РЕЗУЛЬТАТЫ ПО ЗАДАЧАМ:
  COPA  -> Spearman: +0.0598 | Hit Rate (Top-20%): 16.67%
  CSQA  -> Spearman: +0.8090 | Hit Rate (Top-20%): 50.00%
  SIQA  -> Spearman: +0.7848 | Hit Rate (Top-20%): 16.67%
============================================================

--- ОБУЧЕНИЕ НА: ['Qwen', 'gemma'] (CSQA, SIQA) ---
--- ТЕСТ НА (Zero-Shot): phi-tiny ---
Using features ['M11_SVD_Ent' 'M12_KL_Noise' 'M13_LogitLens' 'M14_Spectral_Norm'
 'M15_Frobenius_Norm' 'M16_Effective_Rank' 'M11_SVD_Ent_delta1']

ЛУЧШАЯ НАЙДЕННАЯ ФОРМУЛА:
M12_KL_Noise*(-(M13_LogitLens - M15_Frobenius_Norm - 1*0.3897517) + exp(exp(M14_Spectral_Norm) - exp(exp(M11_SVD_Ent)) + 0.12143961/M15_Frobenius_Norm))/2.042314

РЕЗУЛЬТАТЫ ПО ЗАДАЧАМ:
  CSQA  -> Spearman: +0.5009 | Hit Rate (Top-20%): 66.67%
  SIQA  -> Spearman: +0.4622 | Hit Rate (Top-20%): 50.00%
============================================================
'''