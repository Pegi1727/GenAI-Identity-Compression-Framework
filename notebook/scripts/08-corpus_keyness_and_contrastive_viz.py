"""
Script: 08_corpus_keyness_and_contrastive_viz.py
Description: Computes statistical Keyness (Log-Likelihood G^2) between Independent and 
             AI-revised sub-corpora (Rayson & Garside, 2000), producing publication-grade 
             contrastive visualizations and Keyness tables.
"""

import re
import math
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk

# دانلود لیست کلمات توقف انگلیسی
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

STOPWORDS = set(stopwords.words('english'))

# ------------------------------------------------------------------------------
# 1. توکنایز دقیق و فیلتر کردن علائم نگارشی و کلمات متداول
# ------------------------------------------------------------------------------
def tokenize_for_keyness(texts: list) -> list:
    """استخراج کلمات معنادار الفبایی بدون علائم و کلمات توقف عمومی"""
    words = []
    for text in texts:
        if isinstance(text, str):
            tokens = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            words.extend([t for t in tokens if t not in STOPWORDS])
    return words

# ------------------------------------------------------------------------------
# 2. محاسبه آزمون آماری برجستگی لغوی (Log-Likelihood G^2 / Keyness)
# ------------------------------------------------------------------------------
def calculate_keyness_table(tokens_c1: list, tokens_c2: list, min_freq: int = 3) -> pd.DataFrame:
    """
    محاسبه Log-Likelihood (LL / G^2) بر اساس فرمول استاندارد Rayson & Garside (2000).
    C1: پیکره اول (Independent)
    C2: پیکره دوم (AI-Revised)
    """
    freq1 = Counter(tokens_c1)
    freq2 = Counter(tokens_c2)
    
    n1 = len(tokens_c1)
    n2 = len(tokens_c2)
    
    vocabulary = set(freq1.keys()).union(set(freq2.keys()))
    records = []
    
    for word in vocabulary:
        o1 = freq1.get(word, 0)
        o2 = freq2.get(word, 0)
        
        # فیلتر واژگان با تکرار بسیار کم برای پایداری آماری
        if (o1 + o2) < min_freq:
            continue
            
        # مقادیر مورد انتظار (Expected Frequencies)
        e1 = n1 * (o1 + o2) / (n1 + n2)
        e2 = n2 * (o1 + o2) / (n1 + n2)
        
        # محاسبه Log-Likelihood (G^2)
        ll1 = o1 * math.log(o1 / e1) if o1 > 0 else 0
        ll2 = o2 * math.log(o2 / e2) if o2 > 0 else 0
        g2 = 2.0 * (ll1 + ll2)
        
        # جهت اثر: مثبت = شاخص در متن دانشجو (Indep) | منفی = شاخص در متن هوش مصنوعی (AI)
        direction = 1 if (o1 / n1) > (o2 / n2) else -1
        signed_g2 = round(direction * g2, 3)
        
        records.append({
            'word': word,
            'freq_indep': o1,
            'freq_ai': o2,
            'log_likelihood': g2,
            'signed_keyness': signed_g2,
            'category': 'Distinctive to Independent' if direction > 0 else 'Distinctive to AI'
        })
        
    df_keyness = pd.DataFrame(records).sort_values(by='log_likelihood', ascending=False)
    return df_keyness

# ------------------------------------------------------------------------------
# 3. بصری‌سازی تطبیقی (Keyness-Weighted Contrastive Cloud & Barplot)
# ------------------------------------------------------------------------------
def plot_contrastive_visuals(df_keyness: pd.DataFrame, top_k: int = 15):
    # تنظیم استایل گرافیک برای ژورنال
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 10
    
    # تفکیک واژگان شاخص هر دو گروه
    top_indep = df_keyness[df_keyness['signed_keyness'] > 3.84].head(100) # p < 0.05 بحرانی
    top_ai = df_keyness[df_keyness['signed_keyness'] < -3.84].head(100)
    
    dict_indep = dict(zip(top_indep['word'], top_indep['log_likelihood']))
    dict_ai = dict(zip(top_ai['word'], top_ai['log_likelihood']))
    
    # الف) ابر کلمات وزن‌دار شده با ضریب برجستگی آماری (Keyness)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    
    wc_indep = WordCloud(width=800, height=450, background_color="white", colormap="viridis", max_words=60)
    wc_ai = WordCloud(width=800, height=450, background_color="white", colormap="plasma", max_words=60)
    
    if dict_indep:
        axes[0].imshow(wc_indep.generate_from_frequencies(dict_indep), interpolation="bilinear")
    axes[0].set_title("Keywords Suppressed / Distinctive to Independent Texts\n(Higher Log-Likelihood in Student Drafts)", fontsize=13, fontweight='bold')
    axes[0].axis("off")
    
    if dict_ai:
        axes[1].imshow(wc_ai.generate_from_frequencies(dict_ai), interpolation="bilinear")
    axes[1].set_title("Keywords Injected / Distinctive to AI Revisions\n(Higher Log-Likelihood in AI Drafts)", fontsize=13, fontweight='bold')
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.savefig("figure3_keyness_wordclouds.png", dpi=300)
    plt.close()
    
    # ب) نمودار میله‌ای افقی واژگان برتر برای چاپ در مقاله (Top-K Keyness Barplot)
    top_contrast = pd.concat([
        df_keyness[df_keyness['signed_keyness'] > 0].head(top_k),
        df_keyness[df_keyness['signed_keyness'] < 0].head(top_k)
    ]).sort_values(by='signed_keyness')
    
    plt.figure(figsize=(10, 8), dpi=300)
    colors = ['#d95f02' if x < 0 else '#1b9e77' for x in top_contrast['signed_keyness']]
    
    bars = plt.barh(top_contrast['word'], top_contrast['signed_keyness'], color=colors, edgecolor='black', alpha=0.85)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.xlabel("Signed Log-Likelihood ($G^2$)", fontsize=11, fontweight='bold')
    plt.title(f"Top {top_k} Overrepresented Lexical Items (Independent vs. AI Revisions)", fontsize=12, fontweight='bold')
    
    # راهنمای رنگ
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1b9e77', label='Overrepresented in Student Drafts (Personal/Specific)'),
        Patch(facecolor='#d95f02', label='Overrepresented in AI Revisions (Standardized/Formal)')
    ]
    plt.legend(handles=legend_elements, loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig("figure4_top_keyness_barplot.png", dpi=300)
    plt.close()

# ------------------------------------------------------------------------------
# 4. بدنه اجرایی
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv("combined_results.csv")
    
    tokens_indep = tokenize_for_keyness(df['text_indep'].tolist())
    tokens_ai = tokenize_for_keyness(df['text_ai'].tolist())
    
    # استخراج جدول برجستگی آماری
    df_keyness = calculate_keyness_table(tokens_c1=tokens_indep, tokens_c2=tokens_ai, min_freq=3)
    df_keyness.to_csv("keyness_analysis_results.csv", index=False)
    
    # ترسیم و ذخیره اشکال استاندارد ژورنالی
    plot_contrastive_visuals(df_keyness, top_k=12)
    
    print("=" * 65)
    print("CORPUS KEYNESS ANALYSIS COMPLETED")
    print("=" * 65)
    print("Key statistics for top overused words saved to 'keyness_analysis_results.csv'")
    print("Plots saved: 'figure3_keyness_wordclouds.png' and 'figure4_top_keyness_barplot.png'")
    print("=" * 65)
