# Methodological Specifications & Analytic Notes

## 1. Generative AI Prompt Protocol
Each raw student essay ($N = 60$) was processed using the following standardized zero-shot instructional prompt:

> *"Please revise the following academic essay written by an L2 English learner to improve grammatical accuracy, coherence, and lexical precision while strictly preserving the student's original communicative intent and core argument. Do not alter the overarching structure unless necessary for basic intelligibility."*

- **Model:** GPT-4o / GenAI writing assistant interface
- **Temperature:** Default (0.7)
- **Conditioning:** Within-subject paired design (Raw Draft vs. AI-Assisted Output)

---

## 2. Feature Extraction & Operationalization
- **Grammar Errors:** Total grammatical, morphological, and mechanical errors normalized per text.
- **Lexical Diversity (MTLD):** Measure of Textual Lexical Diversity computed to prevent text-length sensitivity.
- **Stance Markers:** Normalized frequency of epistemic stance devices (hedges, boosters, attitude markers) following Hyland's (2005) metadiscourse taxonomy.
- **First-Person Expressions:** Self-mention markers ($I, me, my, we, our$).
- **Authenticity Gap Proxy:** Composite standardized deviation metric capturing voice divergence.

---

## 3. Statistical Modeling Specifications

### A. Linear Mixed-Effects Models (LMEM)
To account for the repeated-measures / paired hierarchical structure:
$$\text{Linguistic\_Feature}_{ij} = \beta_0 + \beta_1 (\text{Condition}_j) + u_{0i} + \epsilon_{ij}$$
- **Fixed Effect:** Condition ($0 = \text{Raw}, 1 = \text{AI\_Voice}$)
- **Random Intercept:** Subject/Participant ID ($u_{0i} \sim \mathcal{N}(0, \sigma_u^2)$)
- **Estimation Engine:** Restricted Maximum Likelihood (REML) via `statsmodels.formula.api.mixedlm`.

### B. Agency Path Analysis (OLS)
To examine whether baseline learner agency moderates voice change:
1. $\Delta \text{Grammar} \sim \text{AGS\_Total}$
2. $\Delta \text{Stance} \sim \text{AGS\_Total} + \Delta \text{Grammar}$

Where $\Delta \text{Feature} = \text{Score}_{\text{AI\_Voice}} - \text{Score}_{\text{Raw}}$.
