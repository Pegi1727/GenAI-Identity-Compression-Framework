"""
02_paired_ttests.py
-------------------
Paired t-tests (Raw vs AI_Voice) with Cohen's d (paired, dz), 95% CI of the
mean difference, and Holm-Bonferroni correction across the five features.

Input : /mnt/data/clean_analysis_df.csv
Output: /mnt/data/results/02_ttests.csv
"""
import os
os.makedirs('/mnt/data/results', exist_ok=True)

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

DATA = "/mnt/data/clean_analysis_df.csv"
OUT  = "/mnt/data/results/02_ttests.csv"
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
    for f in FEATURES:
        w = df.pivot(index="Participant_ID", columns="Condition", values=f).dropna()
        diff = w["AI_Voice"] - w["Raw"]           # AI_Voice minus Raw
        n = len(diff)
        t, p = stats.ttest_rel(w["AI_Voice"], w["Raw"])
        dfree, m, sd = n - 1, diff.mean(), diff.std(ddof=1)
        d = m / sd if sd > 0 else np.nan          # dz (paired)
        se = sd / np.sqrt(n)
        tc = stats.t.ppf(0.975, dfree)
        rows.append({"Feature": f, "N": n, "Mean_Diff_AI_minus_Raw": m,
                     "t": t, "df": dfree, "p": p, "Cohens_d_paired": d,
                     "CI95_low": m - tc * se, "CI95_high": m + tc * se})
    res = pd.DataFrame(rows)
    reject, p_holm, _, _ = multipletests(res["p"], method="holm")
    res["p_holm"], res["significant_holm"] = p_holm, reject
    res.to_csv(OUT, index=False)
    print(res.round(5).to_string(index=False))

if __name__ == "__main__":
    main()
