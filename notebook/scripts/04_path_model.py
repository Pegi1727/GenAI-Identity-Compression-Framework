"""
04_path_model.py
----------------
Path-style OLS regressions on participant-level data, merging AGS subscales
(Excel, sheet AGS_Subscales) with linguistic change scores (AI_Voice - Raw):
  (a) Grammar_Errors change ~ AGS_Total
  (b) Stance_Markers change ~ AGS_Total + Grammar_Errors change

Inputs : /mnt/data/clean_analysis_df.csv,
         /mnt/data/brave world Research_Data_Complete.xlsx
Output : /mnt/data/results/04_path_model.csv
"""
import os
os.makedirs('/mnt/data/results', exist_ok=True)

import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA = "/mnt/data/clean_analysis_df.csv"
XLSX = "/mnt/data/brave world Research_Data_Complete.xlsx"
OUT  = "/mnt/data/results/04_path_model.csv"

def main():
    df = pd.read_csv(DATA)
    ags = pd.read_excel(XLSX, sheet_name="AGS_Subscales")[
        ["Participant_ID", "AGS_Total_Mean"]]
    w = df.pivot(index="Participant_ID", columns="Condition_Code",
                 values=["Grammar_Errors", "Stance_Markers"])
    w.columns = [f"{a}_AI" if b == 1 else f"{a}_Raw" for a, b in w.columns]
    w = w.reset_index()
    w["Grammar_Change"] = w["Grammar_Errors_AI"] - w["Grammar_Errors_Raw"]
    w["Stance_Change"]  = w["Stance_Markers_AI"] - w["Stance_Markers_Raw"]
    m = w.merge(ags, on="Participant_ID").dropna(
        subset=["AGS_Total_Mean", "Grammar_Change", "Stance_Change"])

    ma = sm.OLS(m["Grammar_Change"],
                sm.add_constant(m["AGS_Total_Mean"])).fit()
    mb = sm.OLS(m["Stance_Change"],
                sm.add_constant(m[["AGS_Total_Mean", "Grammar_Change"]])).fit()

    rows = []
    for name, model, preds in [
            ("a_Grammar_Change", ma, ["AGS_Total_Mean"]),
            ("b_Stance_Change", mb, ["AGS_Total_Mean", "Grammar_Change"])]:
        for pred in preds:
            ci = model.conf_int().loc[pred]
            rows.append({"Model": name, "Predictor": pred,
                         "B": model.params[pred], "SE": model.bse[pred],
                         "t": model.tvalues[pred], "p": model.pvalues[pred],
                         "CI95_low": ci[0], "CI95_high": ci[1]})
        rows.append({"Model": name, "Predictor": "MODEL_FIT(R2,F)",
                     "B": np.nan, "SE": np.nan, "t": np.nan, "p": np.nan,
                     "CI95_low": model.rsquared, "CI95_high": model.fvalue})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out.round(5).to_string(index=False))
    print("R2 a=%.4f, b=%.4f" % (ma.rsquared, mb.rsquared))

if __name__ == "__main__":
    main()
