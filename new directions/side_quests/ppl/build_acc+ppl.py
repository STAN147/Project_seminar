import os
import pandas as pd
import numpy as np

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
    PPL_DIR = os.path.join(BASE_DIR, "actual dataset", "experiments")
    ABLATIONS_DIR = os.path.join(BASE_DIR, "actual dataset", "ablations")

    model_map = {
        "gemma": "gemma",
        "phi-tiny": "phi-tiny",
        "qwen": "Qwen"
    }

    all_data = []

    if not os.path.exists(PPL_DIR):
        return

    for file in os.listdir(PPL_DIR):
        if file.startswith("ppl_drops_") and file.endswith(".csv"):
            filename_clean = file.replace("ppl_drops_", "").replace(".csv", "")
            parts = filename_clean.split("_")
            if len(parts) != 2:
                continue
                
            model_name = parts[0]
            task = parts[1]
            model_folder = model_map.get(model_name, model_name)

            ppl_path = os.path.join(PPL_DIR, file)
            df_ppl = pd.read_csv(ppl_path)
            
            if 'PPL_Degradation' not in df_ppl.columns:
                continue

            abl_path = os.path.join(ABLATIONS_DIR, model_folder, task, "ablations.csv")
            
            if not os.path.exists(abl_path):
                continue
                
            df_abl = pd.read_csv(abl_path)
            
            if 'Accuracy' in df_abl.columns:
                baseline_mask = df_abl['Layer'].astype(str).str.lower() == 'baseline'
                if baseline_mask.any():
                    baseline_acc = df_abl[baseline_mask]['Accuracy'].values[0]
                    df_abl = df_abl[~baseline_mask].copy()
                    df_abl['Ablations'] = baseline_acc - df_abl['Accuracy']
                else:
                    df_abl['Ablations'] = df_abl['Accuracy']
            else:
                val_col = [c for c in df_abl.columns if c.lower() != 'layer'][0]
                df_abl['Ablations'] = df_abl[val_col]

            df_ppl['Layer'] = df_ppl['Layer'].astype(str)
            df_abl['Layer'] = df_abl['Layer'].astype(str)
            
            df_merged = pd.merge(df_ppl[['Layer', 'PPL_Degradation']], df_abl[['Layer', 'Ablations']], on='Layer', how='inner')
            
            df_merged.rename(columns={'PPL_Degradation': 'PPL'}, inplace=True)
            
            df_merged['PPL'] = np.log1p(df_merged['PPL'] - df_merged['PPL'].min())
            
            df_merged['Model'] = model_name
            df_merged['Task'] = task
            
            df_merged = df_merged[['Model', 'Task', 'Layer', 'PPL', 'Ablations']]
            all_data.append(df_merged)

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        out_path = "ppl_vs_ablations.csv"
        final_df.to_csv(out_path, index=False)

if __name__ == "__main__":
    main()
