import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from scipy import stats

# ------------------------------------------------------------------------------
# 1. توابع محاسبه شباهت درون‌گروهی (Pairwise Cosine Similarity)
# ------------------------------------------------------------------------------
def compute_pairwise_similarities(vectors: np.ndarray) -> np.ndarray:
    """
    محاسبه ماتریس شباهت کسینوسی و استخراج مقادیر مثلث بالایی (بدون قطر اصلی)
    """
    sim_matrix = cosine_similarity(vectors)
    # استخراج اندیس‌های مثلث بالایی (Pairwise comparisons without self-loops)
    triu_indices = np.triu_indices_from(sim_matrix, k=1)
    return sim_matrix[triu_indices]

def run_homogenization_analysis(indep_texts, ai_texts, method='tfidf', model_name='all-MiniLM-L6-v2'):
    """
    تحلیل همگونی با روش TF-IDF (سطح واژگان) یا Sentence-Transformers (سطح معنایی/سبکی)
    """
    if method == 'tfidf':
        # استفاده از 1-to-3 grams همراه با کلمات نقشی
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
        # برازش بردارساز روی کل متون برای فضای برداری یکسان
        vectorizer.fit(indep_texts + ai_texts)
        vecs_indep = vectorizer.transform(indep_texts).toarray()
        vecs_ai = vectorizer.transform(ai_texts).toarray()
    
    elif method == 'sbert':
        model = SentenceTransformer(model_name)
        vecs_indep = model.encode(indep_texts, show_progress_bar=False, normalize_embeddings=True)
        vecs_ai = model.encode(ai_texts, show_progress_bar=False, normalize_embeddings=True)
    else:
        raise ValueError("Method must be 'tfidf' or 'sbert'")

    sims_indep = compute_pairwise_similarities(vecs_indep)
    sims_ai = compute_pairwise_similarities(vecs_ai)

    # شاخص‌های توصیفی
    mean_indep, std_indep = np.mean(sims_indep), np.std(sims_indep, ddof=1)
    mean_ai, std_ai = np.mean(sims_ai), np.std(sims_ai, ddof=1)

    # اندازه اثر کوهن (Cohen's d)
    pooled_std = np.sqrt(((len(sims_indep)-1)*std_indep**2 + (len(sims_ai)-1)*std_ai**2) / (len(sims_indep) + len(sims_ai) - 2))
    cohens_d = (mean_ai - mean_indep) / pooled_std if pooled_std > 0 else 0.0

    # آزمون آماری ناپارامتریک (Mann-Whitney U یا Permutation Test)
    u_stat, p_val = stats.mannwhitneyu(sims_ai, sims_indep, alternative='greater')

    return {
        'method': method,
        'mean_sim_indep': round(mean_indep, 4),
        'std_indep': round(std_indep, 4),
        'mean_sim_ai': round(mean_ai, 4),
        'std_ai': round(std_ai, 4),
        'homogenization_delta': round(mean_ai - mean_indep, 4),
        'cohens_d': round(cohens_d, 3),
        'p_value': p_val
    }

# ------------------------------------------------------------------------------
# 2. نحوه اجرا بر روی داده‌ها
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv("texts.csv") # ستون‌ها: StudentID, Condition, Text
    
    indep_texts = df[df['Condition'] == 'Independent']['Text'].tolist()
    ai_texts = df[df['Condition'] == 'AI-Revised']['Text'].tolist()

    # الف) ارزیابی واژگانی با TF-IDF
    tfidf_results = run_homogenization_analysis(indep_texts, ai_texts, method='tfidf')
    print("=== TF-IDF Lexical Homogenization Results ===")
    for k, v in tfidf_results.items():
        print(f"  {k}: {v}")

    # ب) ارزیابی معنایی/بازنمایی با Sentence-BERT (اختیاری ولی به شدت معتبر در Q1)
    try:
        sbert_results = run_homogenization_analysis(indep_texts, ai_texts, method='sbert')
        print("\n=== SBERT Semantic/Stylistic Homogenization Results ===")
        for k, v in sbert_results.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"\nSBERT skipped or library not installed: {e}")
