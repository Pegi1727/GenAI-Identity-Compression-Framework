# ==============================================================================
# Script: LME Analysis for Independent vs. AI-Revised Writing
# ==============================================================================

# 1. Package Installation and Loading
required_pkgs <- c("lme4", "lmerTest", "emmeans", "MuMIn", "performance", "ggplot2", "dplyr", "see")
new_pkgs <- required_pkgs[!(required_pkgs %in% installed.packages()[, "Package"])]
if(length(new_pkgs)) install.packages(new_pkgs)

library(lme4)
library(lmerTest)
library(emmeans)
library(MuMIn)
library(performance)
library(ggplot2)
library(dplyr)

# ------------------------------------------------------------------------------
# 2. Data Loading & Preprocessing
# ------------------------------------------------------------------------------
# Expected columns: StudentID (factor), Condition (factor), Proficiency (factor or numeric),
#                   LexicalDiversity (numeric, e.g., MTLD/VOCD), GrammaticalErrors (count/numeric)

data <- read.csv("writing_data.csv")

data <- data %>%
  mutate(
    StudentID = as.factor(StudentID),
    Condition = factor(Condition, levels = c("Independent", "AI-Revised")),
    Proficiency = as.factor(Proficiency) # یا scale(Proficiency) اگر پیوسته باشد
  )

# ------------------------------------------------------------------------------
# 3. Model Specification: Random Intercept & Random Slope
# ------------------------------------------------------------------------------
# Model A: Main Effects + Interaction (Random Intercept)
model_lme_int <- lmer(LexicalDiversity ~ Condition * Proficiency + (1 | StudentID), 
                      data = data, 
                      REML = TRUE)

# Model B: Adding Random Slope for Condition (if convergence allows)
model_lme_slope <- lmer(LexicalDiversity ~ Condition * Proficiency + (1 + Condition | StudentID), 
                        data = data, 
                        REML = TRUE)

# Model Comparison (Likelihood Ratio Test)
# Note: For testing fixed effects with anova, use REML=FALSE, but for random slopes REML=TRUE is fine
anova(model_lme_int, model_lme_slope, refit = FALSE)

# Select the best model (e.g., model_lme_int or model_lme_slope)
final_model <- model_lme_int
summary(final_model)

# ------------------------------------------------------------------------------
# 4. Effect Size & Model Diagnostics
# ------------------------------------------------------------------------------
# Marginal R2 (variance explained by fixed effects) 
# Conditional R2 (variance explained by fixed + random effects)
r.squaredGLMM(final_model)

# Check model assumptions (Normality of residuals, Homoscedasticity, Multicollinearity)
check_model(final_model)

# ------------------------------------------------------------------------------
# 5. Post-hoc Comparisons (Estimated Marginal Means)
# ------------------------------------------------------------------------------
# Pairwise comparisons between conditions across proficiency levels
emm_interaction <- emmeans(final_model, pairwise ~ Condition | Proficiency, adjust = "bonferroni")
print(emm_interaction$contrasts)

# ------------------------------------------------------------------------------
# 6. Publication-Ready Visualization (Interaction / Distribution)
# ------------------------------------------------------------------------------
# Boxplot with individual trajectories (Paired visual)
ggplot(data, aes(x = Condition, y = LexicalDiversity, fill = Condition)) +
  geom_boxplot(alpha = 0.6, outlier.shape = NA, width = 0.4) +
  geom_line(aes(group = StudentID), color = "gray60", alpha = 0.4) +
  geom_jitter(aes(color = Condition), width = 0.08, size = 2, alpha = 0.7) +
  facet_wrap(~ Proficiency) +
  scale_fill_manual(values = c("#4A90E2", "#50E3C2")) +
  scale_color_manual(values = c("#2A70C2", "#30B3A2")) +
  theme_minimal(base_size = 13) +
  theme(
    legend.position = "none",
    strip.text = element_text(face = "bold", size = 12),
    panel.grid.minor = element_blank(),
    axis.title = element_text(face = "bold")
  ) +
  labs(
    title = "Effect of AI Revision on Lexical Diversity across Proficiency Levels",
    subtitle = "Connected lines represent within-subject shifts across conditions",
    y = "Lexical Diversity (MTLD)",
    x = "Writing Condition"
  )
