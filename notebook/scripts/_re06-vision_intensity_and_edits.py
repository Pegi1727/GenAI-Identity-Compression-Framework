"""
Script: 06_revision_intensity_and_edits.py
Description: Computes Word-Level Edit Distance (WER / Levenshtein), Edit Operations Breakdown 
             (Insertions, Deletions, Substitutions), and correlates Revision Intensity 
             with Authenticity Gap Score (AGS).
"""

import difflib
import re
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


# ------------------------------------------------------------------------------
# 1. تابع توکنایز سریع برای مقایسه سطح کلمه
# ------------------------------------------------------------------------------
def get_words(text: str) -> list:
    """استخراج لیست کلمات به صورت lowercase بدون علائم نگارشی"""
    if not isinstance(text, str):
        return []
    return re.findall(r"\b\w+\b", text.lower())


# ------------------------------------------------------------------------------
# 2. تحلیل دقیق شدت و نوع ویرایش در سطح واژه (Word-Level Edit Profiling)
# ------------------------------------------------------------------------------
def profile_word_edits(orig_text: str, revised_text: str) -> dict:
    """محاسبه معیارهای تغییرات در سطح کلمه بر اساس SequenceMatcher OpCodes:

    - Word Error Rate (WER) / Word Edit Distance
    - Insertions, Deletions, Substitutions
    - Revision Intensity (Normalized Edit Distance)
    """
    orig_words = get_words(orig_text)
    rev_words = get_words(revised_text)

    len_orig = len(orig_words)
    len_rev = len(rev_words)

    if len_orig == 0:
        return {
            "revision_intensity": np.nan,
            "word_edit_distance": np.nan,
            "substitutions": 0,
            "insertions": len_rev,
            "deletions": 0,
            "preserved_words_ratio": 0.0,
        }

    matcher = difflib.SequenceMatcher(None, orig_words, rev_words)

    subs = 0
    inss = 0
    dels = 0
    matches = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            matches += i2 - i1
        elif tag == "replace":
            # تغییر مستقیم کلمات
            subs += max(i2 - i1, j2 - j1)
        elif tag == "insert":
            inss += j2 - j1
        elif tag == "delete":
            dels += i2 - i1

    total_edits = subs + inss + dels
    max_len = max(len_orig, len_rev)

    # شدت ویرایش نرمال‌شده بر حداکثر طول متن (بازه ۰ تا ۱)
    revision_intensity = total_edits / max_len if max_len > 0 else 0.0
    preserved_ratio = matches / len_orig if len_orig > 0 else 0.0

    return {
        "revision_intensity": round(revision_intensity, 4),
        "word_edit_distance": total_edits,
        "substitutions": subs,
        "insertions": inss,
        "deletions": dels,
        "preserved_words_ratio": round(preserved_ratio, 4),
    }


# ------------------------------------------------------------------------------
# 3. اجرای تحلیل بر روی دیتافریم و ارزیابی همبستگی با AGS
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # فرض بر ساختار دیتافریم: StudentID, text_indep, text_ai, AGS_score
    df = pd.read_csv("combined_results.csv")

    # اعمال تحلیل بر روی تمام متون
    edit_profiles = df.apply(
        lambda row: profile_word_edits(row["text_indep"], row["text_ai"]),
        axis=1,
    )
    edit_df = pd.DataFrame(list(edit_profiles))

    # الحاق نتایج به دیتافریم اصلی
    df = pd.concat([df, edit_df], axis=1)

    # فیلتر مقادیر نامعتبر برای محاسبات همبستگی
    valid_data = df.dropna(subset=["revision_intensity", "AGS_score"])

    # همبستگی پیرسون (خطی) و اسپیرمن (رتبه‌ای / برای داده‌های لیکرت)
    r_pearson, p_pearson = pearsonr(
        valid_data["revision_intensity"], valid_data["AGS_score"]
    )
    r_spearman, p_spearman = spearmanr(
        valid_data["revision_intensity"], valid_data["AGS_score"]
    )

    print("=" * 65)
    print("REVISION INTENSITY & AUTHENTICITY GAP (AGS) CORRELATION")
    print("=" * 65)
    print(f"Mean Word Edit Distance : {valid_data['word_edit_distance'].mean():.2f} words")
    print(f"Mean Revision Intensity : {valid_data['revision_intensity'].mean():.3f} (SD = {valid_data['revision_intensity'].std():.3f})")
    print(f"Mean Preserved Words    : {valid_data['preserved_words_ratio'].mean() * 100:.1f}%")
    print("-" * 65)
    print(f"Pearson Correlation (r) : {r_pearson:.3f} (p = {p_pearson:.4e})")
    print(f"Spearman Rank (rho)     : {r_spearman:.3f} (p = {p_spearman:.4e})")
    print("=" * 65)

    # ذخیره فایل تجمیعی
    df.to_csv("revision_intensity_evaluated.csv", index=False)
