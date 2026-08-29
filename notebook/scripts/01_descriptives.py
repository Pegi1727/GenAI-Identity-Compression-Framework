"""
01_descriptives.py
------------------
Descriptive statistics per condition (Raw vs AI_Voice) for the five linguistic
features (mean, SD, skew, kurtosis) plus Shapiro-Wilk normality tests per cell.

Input : /mnt/data/clean_analysis_df.csv
Output: /mnt/data/results/01_descriptives.csv
"""
import os
os.makedirs('/mnt/data/results', exist_ok=True)

import numpy as np
import pandas as pd
from scipy import stats

DATA = "/mnt/data/clean_analysis_df.csv"
OUT  = "/mnt/data/results/01_descriptives.csv"
FEATURES = ["Grammar_Errors", "MTLD", "Stance_Markers",
            "First_Person", "Authenticity_Gap_Proxy"]

def load():
    """Load analysis dataframe and normalise condition labels."""
    df = pd.read_csv(DATA)
    if "Condition" not in df.columns:
        df["Condition"] = df["Condition_Code"].map({0: "Raw", 1: "AI_Voice"})
    if "First_Person" not in df.columns and "First_Person_Expressions" in df.columns:
        df["First_Person"] = df["First_Person_Expressions"]
    return df

def main():
    df = load()
    rows = []
    for cond, g in df.groupby("Condition"):
        for f in FEATURES:
            x = g[f].dropna()
            W, p = stats.shapiro(x) if len(x) >= 3 else (np.nan, np.nan)
            rows.append({"Feature": f, "Condition": cond, "N": len(x),
                         "Mean": x.mean(), "SD": x.std(ddof=1),
                         "Skew": stats.skew(x, bias=False),
                         "Kurtosis": stats.kurtosis(x, bias=False),
                         "Shapiro_W": W, "Shapiro_p": p})
    out = pd.DataFrame(rows).sort_values(["Feature", "Condition"])
    out.to_csv(OUT, index=False)
    print(out.round(4).to_string(index=False))

if __name__ == "__main__":
    main()
