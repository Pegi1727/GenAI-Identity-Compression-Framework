import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# ------------------------------------------------------------------------------
# 1. بارگذاری و پاک‌سازی داده‌ها
# ------------------------------------------------------------------------------
data = pd.read_csv("combined_results.csv")

# انتخاب فقط ستون‌های عددی و معنادار برای همبستگی (جلوگیری از خطای رشته‌ها و ID)
metric_cols = [
    'mtld',                # یا lexical_diversity
    'fp_sing_per100',      # Self-mention (I, my, me)
    'fp_plur_per100',      # We, our
    'hedges_per100',       # Hedges
    'boosters_per100',     # Boosters
    'clean_words_count',   # Text Length
    'AGS_score'            # Authenticity Gap Score
]

# اگر نام ستون‌ها در دیتای شما فرق دارد، ستون‌های عددی موجود را انتخاب می‌کنیم:
available_cols = [c for c in metric_cols if c in data.columns]
if not available_cols:
    available_cols = data.select_dtypes(include=[np.number]).columns.drop('StudentID', errors='ignore').tolist()

sub_data = data[available_cols].dropna()

# ------------------------------------------------------------------------------
# 2. محاسبه دقیق ضریب همبستگی و p-value برای ماتریس (همراه با ستاره‌های معناداری)
# ------------------------------------------------------------------------------
corr_matrix = sub_data.corr(method='pearson')
pval_matrix = pd.DataFrame(np.zeros_like(corr_matrix), index=corr_matrix.index, columns=corr_matrix.columns)

for r in corr_matrix.index:
    for c in corr_matrix.columns:
        if r == c:
            pval_matrix.loc[r, c] = 0.0
        else:
            _, p_val = pearsonr(sub_data[r], sub_data[c])
            pval_matrix.loc[r, c] = p_val

# ایجاد لیبل‌های ترکیبی: ضریب همبستگی + ستاره معناداری (* p < .05, ** p < .01, *** p < .001)
def get_annot_labels(corr, pval):
    annot = np.empty(corr.shape, dtype=object)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            val = corr.iloc[i, j]
            p = pval.iloc[i, j]
            stars = ""
            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            elif p < 0.05:
                stars = "*"
            annot[i, j] = f"{val:.2f}{stars}"
    return annot

annot_labels = get_annot_labels(corr_matrix, pval_matrix)

# ------------------------------------------------------------------------------
# 3. ترسیم Heatmap با ماسک مثلثی (برای جلوگیری از تکرار قرینه)
# ------------------------------------------------------------------------------
plt.style.use('seaborn-v0_8-white' if 'seaborn-v0_8-white' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1.2, 1]})

# ماسک برای ماتریس بالایی
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=annot_labels,
    fmt="",
    cmap="vlag",
    center=0,
    vmin=-1, vmax=1,
    square=True,
    linewidths=0.7,
    cbar_kws={"shrink": 0.8, "label": "Pearson Correlation (r)"},
    ax=ax1
)
ax1.set_title("A) Correlation Matrix with Significance Levels\n(*p<.05, **p<.01, ***p<.001)", fontsize=12, fontweight='bold', pad=10)

# ------------------------------------------------------------------------------
# 4. ترسیم نمودار رگرسیون Scatter Plot برای رابطه اصلی (Diversity / Voice vs AGS)
# ------------------------------------------------------------------------------
target_x = 'mtld' if 'mtld' in sub_data.columns else 'lexical_diversity'
target_y = 'AGS_score'

r_val, p_val = pearsonr(sub_data[target_x], sub_data[target_y])

sns.regplot(
    data=sub_data,
    x=target_x,
    y=target_y,
    scatter_kws={'alpha': 0.6, 'color': '#2b5c8f', 's': 45},
    line_kws={'color': '#d95f02', 'linewidth': 2},
    ax=ax2
)

ax2.set_title(f"B) Linear Trajectory: {target_x} vs. AGS\n(r = {r_val:.2f}, p = {p_val:.4e})", fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel("Lexical Diversity (MTLD)", fontweight='bold')
ax2.set_ylabel("Authenticity Gap Score (AGS)", fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("authenticity_gap_correlation_analysis.png", dpi=300, bbox_inches='tight')
plt.show()
