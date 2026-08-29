import re
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize

# دانلود منابع مورد نیاز nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ------------------------------------------------------------------------------
# 1. الگوریتم محاسبه MTLD (Measure of Textual Lexical Diversity)
# ------------------------------------------------------------------------------
def calculate_mtld(tokens, ttr_threshold=0.72):
    """
    محاسبه MTLD بر اساس روش McCarthy & Jarvis (2010).
    معیار مقاوم در برابر طول متن.
    """
    if len(tokens) == 0:
        return 0.0

    def _mtld_calc(token_seq):
        factors = 0.0
        types = set()
        token_count = 0

        for token in token_seq:
            types.add(token)
            token_count += 1
            current_ttr = len(types) / token_count
            if current_ttr <= ttr_threshold:
                factors += 1.0
                types = set()
                token_count = 0

        if token_count > 0:
            # کسر فاکتور باقیمانده
            excess_ttr_drop = 1.0 - current_ttr
            needed_drop = 1.0 - ttr_threshold
            if needed_drop != 0:
                factors += excess_ttr_drop / needed_drop

        return len(token_seq) / factors if factors > 0 else len(token_seq)

    forward = _mtld_calc(tokens)
    backward = _mtld_calc(tokens[::-1])
    return (forward + backward) / 2.0


# ------------------------------------------------------------------------------
# 2. دیکشنری‌های نشانگرهای صوتی و موضع‌گیری زبانی (Hyland's Metadiscourse Model)
# ------------------------------------------------------------------------------
AUTHORIAL_VOICE = {
    'self_mention_singular': {'i', 'me', 'my', 'mine', 'myself'},
    'self_mention_plural': {'we', 'us', 'our', 'ours', 'ourselves'},
    'hedges': {
        'perhaps', 'maybe', 'possibly', 'possible', 'likely', 'suggests', 'suggest',
        'suggested', 'could', 'might', 'may', 'seem', 'seems', 'seemed', 'appear',
        'appears', 'appeared', 'roughly', 'about', 'probably', 'tend', 'tends'
    },
    'boosters': {
        'clearly', 'definitely', 'obviously', 'certainly', 'demonstrates', 'demonstrate',
        'prove', 'proves', 'proved', 'undoubtedly', 'always', 'never', 'indeed', 'showed'
    }
}

# ------------------------------------------------------------------------------
# 3. تابع جامع تحلیل متن
# ------------------------------------------------------------------------------
def analyze_text_metrics(text: str) -> dict:
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {
            'total_tokens': 0,
            'clean_words_count': 0,
            'ttr': 0.0,
            'mtld': 0.0,
            'fp_sing_raw': 0,
            'fp_sing_per100': 0.0,
            'fp_plur_raw': 0,
            'fp_plur_per100': 0.0,
            'hedges_raw': 0,
            'hedges_per100': 0.0,
            'boosters_raw': 0,
            'boosters_per100': 0.0,
            'compression_ratio_estimate': 0.0
        }

    # توکنایز و پاک‌سازی علائم نگارشی
    raw_tokens = word_tokenize(text.lower())
    # فقط کلمات الفبایی (بدون علائم نگارشی)
    words = [t for t in raw_tokens if re.match(r'^[a-zA-Z]+$', t)]
    total_words = len(words)

    if total_words == 0:
        return {'total_tokens': len(raw_tokens), 'clean_words_count': 0, 'mtld': 0.0}

    # الف) تنوع واژگانی
    unique_words = len(set(words))
    ttr = unique_words / total_words
    mtld = calculate_mtld(words)

    # ب) شمارش نشانگرهای هویت و موضع‌گیری (Voice / Identity / Stance)
    fp_sing_count = sum(1 for w in words if w in AUTHORIAL_VOICE['self_mention_singular'])
    fp_plur_count = sum(1 for w in words if w in AUTHORIAL_VOICE['self_mention_plural'])
    hedges_count = sum(1 for w in words if w in AUTHORIAL_VOICE['hedges'])
    boosters_count = sum(1 for w in words if w in AUTHORIAL_VOICE['boosters'])

    # ج) نرمال‌سازی در مقیاس ۱۰۰ کلمه
    norm_factor = 100.0 / total_words

    return {
        'total_tokens': len(raw_tokens),
        'clean_words_count': total_words,
        'ttr': round(ttr, 4),
        'mtld': round(mtld, 2),
        'fp_sing_raw': fp_sing_count,
        'fp_sing_per100': round(fp_sing_count * norm_factor, 3),
        'fp_plur_raw': fp_plur_count,
        'fp_plur_per100': round(fp_plur_count * norm_factor, 3),
        'hedges_raw': hedges_count,
        'hedges_per100': round(hedges_count * norm_factor, 3),
        'boosters_raw': boosters_count,
        'boosters_per100': round(boosters_count * norm_factor, 3)
    }

# ------------------------------------------------------------------------------
# 4. اجرای پایپ‌لاین بر روی DataFrame
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # فرضا فایل ورودی texts.csv با ستون‌های StudentID, Condition, Text
    df = pd.read_csv("texts.csv")

    metrics_records = df['Text'].apply(analyze_text_metrics).tolist()
    metrics_df = pd.DataFrame.from_records(metrics_records)

    final_df = pd.concat([df[['StudentID', 'Condition']], metrics_df], axis=1)

    # ذخیره نتایج برای ورود به پایپ‌لاین R (LME)
    final_df.to_csv("textual_analysis_results.csv", index=False)
    print("Feature extraction complete. Shape:", final_df.shape)
    print(final_df[['StudentID', 'Condition', 'clean_words_count', 'mtld', 'fp_sing_per100', 'hedges_per100']].head())
