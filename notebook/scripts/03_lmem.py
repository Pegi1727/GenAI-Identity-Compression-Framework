"""
03_lmem.py
----------
Linear mixed-effects models (statsmodels MixedLM, ML) with a random intercept
per Participant_ID, predicting each linguistic feature from Condition
(0 = Raw, 1 = AI_Voice).

Input : /mnt/data/clean_analysis_df.csv
Output: /mnt/data/results/03_lmem.csv
"""
import os
os.makedirs('/mnt/data/results', exist_ok=True)

import pandas as pd
import statsmodels.formula.api as smf

DATA = "/mnt/data/clean_analysis_df.csv"
OUT  = "/mnt/data/results/03_lmem.csv"
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
    import numpy as np
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    df = load()
    df["Cond"] = df["Condition_Code"] if "Condition_Code" in df.columns \
        else df["Condition"].map({"Raw": 0, "AI_Voice": 1})
    rows = []
    for f in FEATURES:
        d = df[["Participant_ID", "Cond", f]].dropna()
        try:
            try:
                fit = smf.mixedlm(f"{f} ~ Cond", d, groups=d["Participant_ID"]).fit(
                    reml=False, method="lbfgs")
            except Exception as e1:
                raise RuntimeError(f"lbfgs failed: {e1}")
        except Exception:
            try:
                fit = smf.mixedlm(f"{f} ~ Cond", d, groups=d["Participant_ID"]).fit(
                    reml=False, method="powell")
            except Exception as e2:
                rows.append({"Feature": f, "Intercept": np.nan, "Cond_Effect": np.nan,
                             "SE": np.nan, "z": np.nan, "p": np.nan,
                             "CI95_low": np.nan, "CI95_high": np.nan,
                             "Random_Intercept_Var": np.nan, "Converged": False,
                             "Note": f"fit failed ({type(e2).__name__}: {e2})"})
                continue
        ci = fit.conf_int().loc["Cond"]
        rows.append({"Feature": f, "Intercept": fit.fe_params["Intercept"],
                     "Cond_Effect": fit.fe_params["Cond"],
                     "SE": fit.bse["Cond"], "z": fit.tvalues["Cond"],
                     "p": fit.pvalues["Cond"], "CI95_low": ci[0],
                     "CI95_high": ci[1],
                     "Random_Intercept_Var": fit.cov_re.iloc[0, 0],
                     "Converged": fit.converged, "Note": ""})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out.round(5).to_string(index=False))

if __name__ == "__main__":
    main()
