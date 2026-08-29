"""
Script: 09_syntactic_complexity_lu2010.py
Description: Computes Multi-Dimensional Syntactic Complexity Indices (Lu, 2010; 2011)
             including MLS, MLT, Clause Subordination (C/T), Complex Nominals (CN/C), 
             and Syntactic Variance (Uniformity/Standardization Index).
"""

import numpy as np
import pandas as pd
import spacy
from scipy.stats import ttest_rel, wilcoxon, levene

# بارگذاری مدل spaCy با قابلیت Dependency Parsing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# ------------------------------------------------------------------------------
# 1. تابع استخراج شاخص‌های چندگانه پیچیدگی نحوی
# ------------------------------------------------------------------------------
def compute_syntactic_indices(text: str) -> dict:
    """
    استخراج متغیرهای استاندارد Lu (2010):
    - MLS: Mean Length of Sentence
    - MLT: Mean Length of T-unit
    - C/T: Clauses per T-unit (Subordination)
    - CN/C: Complex Nominals per Clause (Phrasal Compression)
    - Sent_SD: انحراف معیار طول جملات (تنوع ساختاری)
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {
            'MLS': np.nan, 'MLT': np.nan, 'C_T': np.nan, 
            'CN_C': np.nan, 'Sent_SD': np.nan, 'num_sentences': 0
        }
        
    doc = nlp(text)
    sentences = list(doc.sents)
    if len(sentences) == 0:
        return {
            'MLS': np.nan, 'MLT': np.nan, 'C_T': np.nan, 
            'CN_C': np.nan, 'Sent_SD': np.nan, 'num_sentences': 0
        }

    # طول جملات و کلمات معنادار (بدون علائم نگارشی صرف)
    sent_lengths = []
    total_words = 0
    total_clauses = 0
    total_t_units = 0
    total_complex_nominals = 0

    for sent in sentences:
        words = [token for token in sent if not token.is_punct and not token.is_space]
        n_words = len(words)
        if n_words == 0:
            continue
        sent_lengths.append(n_words)
        total_words += n_words

        # شمارش افعال دارای مسند/ریشه برای شناسایی Clause
        clauses = [token for token in sent if token.pos_ == "VERB" and token.dep_ in ["ROOT", "advcl", "relcl", "ccomp", "xcomp", "conj"]]
        clause_count = max(len(clauses), 1)
        total_clauses += clause_count

        # تقریب T-units: بندهای پایه و بندهای هم‌پایه (Coordinated main clauses)
        coord_roots = [token for token in sent if token.dep_ == "conj" and token.pos_ == "VERB" and token.head.dep_ == "ROOT"]
        t_units_in_sent = 1 + len(coord_roots)
        total_t_units += t_units_in_sent

        # شناسایی Complex Nominals (بر اساس تگ‌های اسمی با وابسته، صفت، یا موصول)
        for token in sent:
            if token.pos_ in ["NOUN", "PROPN"]:
                has_adj = any(child.dep_ in ["amod", "prep", "relcl", "appos"] for child in token.children)
                if has_adj:
                    total_complex_nominals += 1

    num_sents = len(sent_lengths)
    if num_sents == 0 or total_t_units == 0 or total_clauses == 0:
        return {
            'MLS': np.nan, 'MLT': np.nan, 'C_T': np.nan, 
            'CN_C': np.nan, 'Sent_SD': np.nan, 'num_sentences': 0
        }

    # محاسبات نسبت‌ها
    mls = total_words / num_sents
    mlt = total_words / total_t_units
    c_t = total_clauses / total_t_units
    cn_c = total_complex_nominals / total_clauses
    sent_sd = float(np.std(sent_lengths, ddof=1)) if num_sents > 1 else 0.0

    return {
        'MLS': round(mls, 3),
        'MLT': round(mlt, 3),
        'C_T': round(c_t, 3),
        'CN_C': round(cn_c, 3),
        'Sent_SD': round(sent_sd, 3),
        'num_sentences': num_sents
    }

# ------------------------------------------------------------------------------
# 2. پایپ‌لاین تحلیل مقایسه‌ای و پدیده‌ی Standardization
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv("combined_results.csv")
    
    # استخراج شاخص‌ها برای متون مستقل و بازنویسی‌شده توسط هوش مصنوعی
    indep_metrics = df['text_indep'].apply(compute_syntactic_indices)
    ai_metrics = df['text_ai'].apply(compute_syntactic_indices)
    
    df_indep_syn = pd.DataFrame(list(indep_metrics)).add_suffix('_indep')
    df_ai_syn = pd.DataFrame(list(ai_metrics)).add_suffix('_ai')
    
    df = pd.concat([df, df_indep_syn, df_ai_syn], axis=1)
    df.to_csv("syntactic_complexity_evaluated.csv", index=False)
    
    # --------------------------------------------------------------------------
    # 3. گزارش آماری و آزمون فرضیه همگن‌سازی نحوی (Syntactic Homogenization)
    # --------------------------------------------------------------------------
    valid = df.dropna(subset=['MLS_indep', 'MLS_ai', 'C_T_indep', 'C_T_ai'])
    
    # آزمون‌های تی زوجی برای تغییر در شاخص‌های ساختاری
    t_mls, p_mls = ttest_rel(valid['MLS_indep'], valid['MLS_ai'])
    t_ct, p_ct = ttest_rel(valid['C_T_indep'], valid['C_T_ai'])
    t_cnc, p_cnc = ttest_rel(valid['CN_C_indep'], valid['CN_C_ai'])
    t_var, p_var = ttest_rel(valid['Sent_SD_indep'], valid['Sent_SD_ai'])
    
    print("=" * 75)
    print("MULTI-DIMENSIONAL SYNTACTIC COMPLEXITY ANALYSIS (Lu, 2010; 2011)")
    print("=" * 75)
    print(f"1. Mean Length of Sentence (MLS):")
    print(f"   Indep: {valid['MLS_indep'].mean():.2f} (SD={valid['MLS_indep'].std():.2f}) | AI: {valid['MLS_ai'].mean():.2f} (SD={valid['MLS_ai'].std():.2f})")
    print(f"   Paired t-test: t = {t_mls:.3f}, p = {p_mls:.4e}")
    print("-" * 75)
    print(f"2. Clausal Subordination Ratio (C/T):")
    print(f"   Indep: {valid['C_T_indep'].mean():.2f} | AI: {valid['C_T_ai'].mean():.2f}")
    print(f"   Paired t-test: t = {t_ct:.3f}, p = {p_ct:.4e}")
    print("-" * 75)
    print(f"3. Phrasal Complexity / Complex Nominals per Clause (CN/C):")
    print(f"   Indep: {valid['CN_C_indep'].mean():.2f} | AI: {valid['CN_C_ai'].mean():.2f}")
    print(f"   Paired t-test: t = {t_cnc:.3f}, p = {p_cnc:.4e}")
    print("-" * 75)
    print(f"4. Sentence Length Variance / Rhythm Uniformity (Within-Text SD):")
    print(f"   Indep SD: {valid['Sent_SD_indep'].mean():.2f} | AI SD: {valid['Sent_SD_ai'].mean():.2f}")
    print(f"   Uniformity Drop: t = {t_var:.3f}, p = {p_var:.4e} (Drop in SD = Flattened Sentence Rhythm)")
    print("=" * 75)
