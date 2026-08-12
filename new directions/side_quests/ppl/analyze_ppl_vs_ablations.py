import os
import pandas as pd
from scipy.stats import spearmanr

def main():
    file_path = "ppl_vs_ablations.csv"
    df = pd.read_csv(file_path)
    for (model, task), group in df.groupby(['Model', 'Task']):
        if len(group) > 1:
            corr, _ = spearmanr(group['PPL'], group['Ablations'])
            print(f"Model: {model:<15} | Task: {task:<10} | Spearman: {corr:.4f}")

if __name__ == "__main__":
    main()
