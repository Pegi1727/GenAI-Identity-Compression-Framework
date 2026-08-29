"""
Script: 07_semantic_affective_shift.py
Description: Computes Valence and Subjectivity Shifts between Independent and AI-revised texts,
             categorizing shifts into Neutralization, Polarization, and Loss of Voice,
             and correlates them with the Authenticity Gap Score (AGS).
"""

import numpy as np
import pandas as pd
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.stats import pearsonr, spearmanr, ttest_rel, wilcoxon

# دانلود داده‌های لکسیکال VADER در صورت نیاز
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# ------------------------------------------------------------------------------
# 1. تابع استخراج ابعاد عاطفی و ذهنی متن (Affective & Subjectivity Profiler)
# ------------------------------------------------------------------------------
def compute_tone_metrics(text: str, sia: SentimentIntensityAnalyzer) -> dict:
    """
    استخراج:
    - Polarity / Valence (Compound Score: -1 to +1) از طریق VADER
    - Pos/Neu/Neg proportions
    - Subjectivity (0: کاملاً عینی/آبجکتیو تا 1: کاملاً ذهنی/سابجکتیو)
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {
            'compound': np.nan,
            'pos': np.nan,
            'neu': np.nan,
            'neg': np.nan,
            'subjectivity': np.nan
        }
    
    scores = sia.polarity_scores(text)
    tb_obj = TextBlob(text)
    subjectivity = tb_obj.sentiment.subjectivity
    
    return {
        'compound': scores['compound'],
        'pos': scores['pos'],
        'neu': scores['neu'],
        'neg': scores['neg'],
        'subjectivity': subjectivity
    }

# ------------------------------------------------------------------------------
# 2. تابع محاسبه میزان و جهت تغییر لحن (Shift & Trajectory)
# ------------------------------------------------------------------------------
def evaluate_semantic_shift(orig_metrics: dict, rev_metrics: dict) -> dict:
    """
    محاسبه تغییر قدرمطلق و جهت‌دار در ظرفیت عاطفی و ذهنیت متن
    """
    val_orig = orig_metrics['compound']
    val_rev = rev_metrics['compound']
    subj_orig = orig_metrics['subjectivity']
    subj_rev = rev_metrics['subjectivity']
    
    if np.isnan(val_orig) or np.isnan(val_rev):
        return {
            'valence_shift_abs': np.nan,
            'valence_shift_dir': np.nan,
            'subjectivity_shift_abs': np.nan,
            'subjectivity_shift_dir': np.nan,
            'neutralization_index': np.nan
        }
        
    # شیفت عاطفی: قدر مطلق و برداری
    valence_shift_abs = abs(val_rev - val_orig)
    valence_shift_dir = val_rev - val_orig  # مثبت: خوش‌بینانه‌تر/منفی: بدبینانه‌تر
    
    # شیفت ذهنیت: میزان از دست رفتن صدای شخصی و رفتن به سمت لحن صرفاً عینی
    subj_shift_abs = abs(subj_rev - subj_orig)
    subj_shift_dir = subj_rev - subj_orig   # مقادیر منفی نشان‌دهنده ابجکتیو شدن و سلب هویت است
    
    # شاخص خنثی‌سازی (Neutralization): حرکت از حالت قطبی به سمت خنثی (0)
    orig_abs_extremity = abs(val_orig)
    rev_abs_extremity = abs(val_rev)
    neutralization_index = orig_abs_extremity - rev_abs_extremity # مثبت = متن خنثی‌تر و استانداردتر شده
    
    return {
        'valence_shift_abs': round(valence_shift_abs, 4),
        'valence_shift_dir': round(valence_shift_dir, 4),
        'subjectivity_shift_abs': round(subj_shift_abs, 4),
        'subjectivity_shift_dir': round(subj_shift_dir, 4),
        'neutralization_index': round(neutralization_index, 4)
    }

# ------------------------------------------------------------------------------
# 3. اجرای پایپ‌لاین بر روی دیتاست
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    sia = SentimentIntensityAnalyzer()
    df = pd.read_csv("combined_results.csv")
    
    # استخراج معیارهای لحن برای هر دو نسخه
    indep_metrics = df['text_indep'].apply(lambda t: compute_tone_metrics(t, sia))
    ai_metrics = df['text_ai'].apply(lambda t: compute_tone_metrics(t, sia))
    
    # تبدیل به دیتافریم‌های موقت
    df_indep_tone = pd.DataFrame(list(indep_metrics)).add_suffix('_indep')
    df_ai_tone = pd.DataFrame(list(ai_metrics)).add_suffix('_ai')
    
    # ادغام با دیتافریم اصلی
    df = pd.concat([df, df_indep_tone, df_ai_tone], axis=1)
    
    # محاسبه تغییرات معنایی/عاطفی
    shift_results = []
    for idx, row in df.iterrows():
        orig_m = {'compound': row['compound_indep'], 'subjectivity': row['subjectivity_indep']}
        rev_m = {'compound': row['compound_ai'], 'subjectivity': row['subjectivity_ai']}
        shift_results.append(evaluate_semantic_shift(orig_m, rev_m))
        
    df_shifts = pd.DataFrame(shift_results)
    df = pd.concat([df, df_shifts], axis=1)
    
    # ذخیره در فایل CSV
    df.to_csv("semantic_shift_evaluated.csv", index=False)
    
    # --------------------------------------------------------------------------
    # 4. گزارش آماری جامع
    # --------------------------------------------------------------------------
    valid = df.dropna(subset=['valence_shift_abs', 'AGS_score'])
    
    # آزمون زوجی تغییر ذهنیت (Subjectivity: آیا متن‌ها رباتیک‌تر/عینی‌تر شده‌اند؟)
    t_subj, p_subj = ttest_rel(valid['subjectivity_indep'], valid['subjectivity_ai'])
    
    # همبستگی شیفت لحن با شکاف اصالت (AGS)
    r_val, p_val = spearmanr(valid['valence_shift_abs'], valid['AGS_score'])
    r_subj, p_subj_corr = spearmanr(valid['subjectivity_shift_abs'], valid['AGS_score'])
    r_neut, p_neut = spearmanr(valid['neutralization_index'], valid['AGS_score'])
    
    print("=" * 70)
    print("AFFECTIVE & SEMANTIC SHIFT ANALYSIS (VADER + SUBJECTIVITY)")
    print("=" * 70)
    print(f"Mean Indep Subjectivity : {valid['subjectivity_indep'].mean():.3f} (SD = {valid['subjectivity_indep'].std():.3f})")
    print(f"Mean AI Subjectivity    : {valid['subjectivity_ai'].mean():.3f} (SD = {valid['subjectivity_ai'].std():.3f})")
    print(f"Subjectivity Drop Test  : t = {t_subj:.3f}, p = {p_subj:.4e}")
    print("-" * 70)
    print(f"Mean Valence Shift (|Δ|): {valid['valence_shift_abs'].mean():.3f}")
    print(f"Mean Neutralization Idx : {valid['neutralization_index'].mean():.3f} (Positive = Standardized/Flattened)")
    print("-" * 70)
    print("CORRELATIONS WITH AUTHENTICITY GAP SCORE (AGS):")
    print(f"• Absolute Valence Shift vs AGS  : rho = {r_val:.3f} (p = {p_val:.4e})")
    print(f"• Subjectivity Shift vs AGS     : rho = {r_subj:.3f} (p = {p_subj_corr:.4e})")
    print(f"• Neutralization Index vs AGS   : rho = {r_neut:.3f} (p = {p_neut:.4e})")
    print("=" * 70)
