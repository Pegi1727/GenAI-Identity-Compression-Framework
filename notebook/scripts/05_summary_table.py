"""
05_summary_table.py
-------------------
Combines descriptives, paired t-tests, mixed-model effects and path-model
coefficients into one summary table (CSV + Excel workbook).

Inputs : /mnt/data/results/01..04 CSVs
Outputs: /mnt/data/results/05_summary_table.csv (and .xlsx)
"""
import os
os.makedirs('/mnt/data/results', exist_ok=True)

import pandas as pd

R = "/mnt/data/results"

def main():
    desc = pd.read_csv(f"{R}/01_descriptives.csv")
    tt   = pd.read_csv(f"{R}/02_ttests.csv")
    lmem = pd.read_csv(f"{R}/03_lmem.csv")
    path = pd.read_csv(f"{R}/04_path_model.csv")

    dv = desc.pivot(index="Feature", columns="Condition", values=["Mean", "SD"])
    dv.columns = [f"{a}_{b}" for a, b in dv.columns]
    dv = dv.reset_index()

    summary = dv.merge(tt[["Feature", "Mean_Diff_AI_minus_Raw", "t", "df", "p",
                           "p_holm", "Cohens_d_paired", "CI95_low", "CI95_high",
                           "significant_holm"]], on="Feature", how="left")
    summary = summary.merge(
        lmem[["Feature", "Cond_Effect", "z", "p"]].rename(
            columns={"Cond_Effect": "LMEM_Cond_Effect", "z": "LMEM_z",
                     "p": "LMEM_p"}), on="Feature", how="left")
    summary.to_csv(f"{R}/05_summary_table.csv", index=False)
    with pd.ExcelWriter(f"{R}/05_summary_table.xlsx") as xl:
        summary.to_excel(xl, sheet_name="Ttests_LMEM", index=False)
        path.to_excel(xl, sheet_name="Path_Model", index=False)
    pd.set_option("display.width", 250)
    print(summary.round(4).to_string(index=False))

if __name__ == "__main__":
    main()
