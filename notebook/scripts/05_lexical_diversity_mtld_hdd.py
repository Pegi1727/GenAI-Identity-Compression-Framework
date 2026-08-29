"""
Script: 05_lexical_diversity_mtld_hdd.py
Description: Computes length-invariant lexical diversity indices (MTLD and HD-D)
             following McCarthy & Jarvis (2010) and McCarthy (2005) standards.
"""

import math
import re
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


# ------------------------------------------------------------------------------
# 1. تابع پیش‌پردازش استاندارد برای شاخص‌های تنوع واژگانی
# ------------------------------------------------------------------------------
def tokenize_clean(text: str) -> list:
    """استخراج کلمات معتبر الفبایی بدون علائم نگارشی و اعداد"""
    if not isinstance(text, str):
        return []
    # تبدیل به حروف کوچک و استخراج واژگان الفبایی
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())


# ------------------------------------------------------------------------------
# 2. پیاده‌سازی الگوریتم دقیق MTLD (Forward + Backward)
# ------------------------------------------------------------------------------
def compute_mtld(tokens: list, ttr_threshold: float = 0.72) -> float:
    """محاسبه MTLD بر اساس متد استاندارد McCarthy & Jarvis (2010).

    ttr_threshold پیش‌فرض: 0.72
    """
    if len(tokens) == 0:
        return 0.0

    def _eval_seq(seq):
        factors = 0.0
        types = set()
        token_count = 0
        current_ttr = 1.0

        for token in seq:
            types.add(token)
            token_count += 1
            current_ttr = len(types) / token_count

            # اگر TTR به زیر آستانه رسید، یک فاکتور کامل ثبت و پنجره ریست می‌شود
            if current_ttr <= ttr_threshold:
                factors += 1.0
                types = set()
                token_count = 0

        # محاسبه فاکتور کسری برای واژگان باقی‌مانده انتهای متن
        if token_count > 0:
            excess_drop = 1.0 - current_ttr
            needed_drop = 1.0 - ttr_threshold
            if needed_drop > 0:
                factors += excess_drop / needed_drop

        return len(seq) / factors if factors > 0 else float(len(seq))

    # میانگین اجرای رو به جلو و رو به عقب
    mtld_fwd = _eval_seq(tokens)
    mtld_bwd = _eval_seq(tokens[::-1])
    return round((mtld_fwd + mtld_bwd) / 2.0, 3)


# ------------------------------------------------------------------------------
# 3. پیاده‌سازی مکمل HD-D (Hypergeometric Distribution D)
# ------------------------------------------------------------------------------
def compute_hdd(tokens: list, sample_size: int = 42) -> float:
    """محاسبه شاخص HD-D (vocd alternative).

    مناسب برای اثبات بیشتر عدم وابستگی به طول متن.
    """
    if len(tokens) < sample_size:
        return np.nan

    N = len(tokens)
    freqs = Counter(tokens)
    hdd_sum = 0.0

    for word, count in freqs.items():
        # احتمال مشاهده حداقل یک‌بار کلمه در نمونه تصادفی به اندازه sample_size
        prob = 1.0 - (
            math.comb(N - count, sample_size) / math.comb(N, sample_size)
        )
        hdd_sum += prob

    return round(hdd_sum, 3)


# ------------------------------------------------------------------------------
# 4. اجرای پایپ‌لاین بر روی داده و گزارش آماری زوجی (Paired Evaluation)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # فایل ورودی با ستون‌های: StudentID, text_independent, text_ai
    df = pd.read_csv("combined_results.csv")

    # استخراج توکن‌ها
    df["tokens_indep"] = df["text_independent"].apply(tokenize_clean)
    df["tokens_ai"] = df["text_ai"].apply(tokenize_clean)

    # محاسبه MTLD
    df["mtld_indep"] = df["tokens_indep"].apply(compute_mtld)
    df["mtld_ai"] = df["tokens_ai"].apply(compute_mtld)

    # محاسبه HD-D
    df["hdd_indep"] = df["tokens_indep"].apply(compute_hdd)
    df["hdd_ai"] = df["tokens_ai"].apply(compute_hdd)

    # ذخیره دیتاست خروجی تمیز بدون لیست توکن‌ها
    df_clean = df.drop(columns=["tokens_indep", "tokens_ai"])
    df_clean.to_csv("lexical_diversity_evaluated.csv", index=False)

    # محاسبات آماری توصیفی برای MTLD
    m_indep, sd_indep = df["mtld_indep"].mean(), df["mtld_indep"].std()
    m_ai, sd_ai = df["mtld_ai"].mean(), df["mtld_ai"].std()
    delta_mtld = m_ai - m_indep

    # محاسبه اندازه اثر کوهن (Cohen's d برای داده‌های زوجی)
    diff = df["mtld_ai"] - df["mtld_indep"]
    cohen_d = diff.mean() / diff.std() if diff.std() > 0 else 0.0

    # آزمون‌های آماری (پارامتریک و ناپارامتریک)
    t_stat, p_val_t = ttest_rel(
        df["mtld_ai"], df["mtld_indep"], nan_policy="omit"
    )
    w_stat, p_val_w = wilcoxon(df["mtld_ai"], df["mtld_indep"])

    print("=" * 60)
    print("LEXICAL DIVERSITY ANALYSIS (MTLD & HD-D)")
    print("=" * 60)
    print(f"Independent MTLD  : Mean = {m_indep:.3f}, SD = {sd_indep:.3f}")
    print(f"AI-Revised  MTLD  : Mean = {m_ai:.3f}, SD = {sd_ai:.3f}")
    print(f"Mean Difference   : {delta_mtld:+.3f}")
    print(f"Paired t-test     : t = {t_stat:.3f}, p = {p_val_t:.4e}")
    print(f"Wilcoxon signed   : W = {w_stat:.3f}, p = {p_val_w:.4e}")
    print(f"Effect Size (d)   : {cohen_d:.3f}")
    print("=" * 60)
    print(
        "Results successfully saved to 'lexical_diversity_evaluated.csv' for LME modelling.csv", index=False)

    # محاسبات آماری توصیفی برای MTLD
    m_indep, sd_indep = df["mtld_indep"].mean(), df["mtld_indep"].std()
    m_ai, sd_ai = df["mtld_ai"].mean(), df["mtld_ai"].std()
    delta_mtld = m_ai - m_indep

    # محاسبه اندازه اثر کوهن (Cohen's d برای داده‌های زوجی)
    diff = df["mtld_ai"] - df["mtld_indep"]
    cohen_d = diff.mean() / diff.std() if diff.std() > 0 else 0.0

    # آزمون‌های آماری (پارامتریک و ناپارامتریک)
    t_stat, p_val_t = ttest_rel(
        df["mtld_ai"], df["mtld_indep"], nan_policy="omit"
    )
    w_stat, p_val_w = wilcoxon(df["mtld_ai"], df["mtld_indep"])

    print("=" * 60)
    print("LEXICAL DIVERSITY ANALYSIS (MTLD & HD-D)")
    print("=" * 60)
    print(f"Independent MTLD  : Mean = {m_indep:.3f}, SD = {sd_indep:.3f}")
    print(f"AI-Revised  MTLD  : Mean = {m_ai:.3f}, SD = {sd_ai:.3f}")
    print(f"Mean Difference   : {delta_mtld:+.3f}")
    print(f"Paired t-test     : t = {t_stat:.3f}, p = {p_val_t:.4e}")
    print(f"Wilcoxon signed   : W = {w_stat:.3f}, p = {p_val_w:.4e}")
    print(f"Effect Size (d)   : {cohen_d:.3f}")
    print("=" * 60)
    print(
        "Results successfully saved to 'lexical_diversity_evaluated.csv' for LME modelling."
    )
