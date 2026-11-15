# Decisions log: development modelling (NHANES 2011–2016)

Script: `scripts/fit_development_models.py` · Run: 14 October 2026, 21.7 minutes · Seed 20261014
Registration: OSF Secondary Data Preregistration, submitted 14 October 2026 (embargoed to 14 April 2028).

Only the development file was read. The validation files (2017–March 2020, 2021–2023) were not opened by this step.

## Implementation details the registration did not specify

These are choices made while writing the code. None changes a registered decision; each should be reported in the methods.

1. **Imputation method.** Chained equations written in Python. Continuous items (BMI, waist, activity minutes) use linear regression plus predictive mean matching with 5 donors; activity minutes are modelled on the log scale. Binary items use logistic regression plus a Bernoulli draw. Each model is fitted on a bootstrap sample of the observed values, so the 10 imputations reflect parameter uncertainty. 10 cycles per imputation.
2. **Outcome in imputation.** When imputing training data, the outcome is a predictor (as registered). To impute held-out folds and the validation samples without their outcomes (also as registered), a second set of imputation models without the outcome is fitted on the completed training data and used for all new data.
3. **Gestational diabetes** is imputed among women only; men are 0 by definition. After imputation, blood-pressure medication is set to 0 where hypertension is 0, and the derived items (ADA hypertension, high-glucose history) and both scores are recomputed.
4. **Hyperparameter tuning** uses the first imputed dataset of each training set; the chosen configuration is then fitted to all 10 imputed datasets.
5. **Search space:** trees 100–800, learning rate 0.01–0.3 (log), depth 2–6, minimum child weight 1–20 (log), subsample 0.5–1, column subsample 0.5–1, L1 0.001–10 (log), L2 0.01–100 (log).
6. **Range-extension rule** (registered: "if the best configuration is at a search boundary, the range is extended once"). Implemented as: if the best value of any parameter lies within 5% of an edge of its range, that edge is extended once (log-scale parameters by a factor of 10; others by half the range, within hard limits such as depth ≥1), 50 further configurations are searched, and the overall best is kept. This was triggered in 3 of 5 outer folds and in the final model.
7. **Platt recalibration** is fitted on the 5-fold out-of-fold predictions of the chosen configuration from the tuning step.
8. **Logistic regression:** unpenalised; spline knots at the 10th, 50th and 90th percentiles of the first imputed training dataset. The fallback for non-convergence was never needed (unpenalised fit converged in every fold and in the final model).
9. **H2 thresholds** are set on the nested cross-validation out-of-fold predictions. For the two integer scores, the cutoff whose sensitivity (averaged over imputations) is nearest 90% is chosen.
10. **FINDRISC high-glucose item** uses `high_glucose_history` (told prediabetes, borderline, or gestational diabetes), as defined in the registered codebook. The model predictor list uses `told_prediabetes`, as registered.

## Results of the development step (internal; no validation data used)

| Model | Nested-CV AUC | Brier |
|---|---|---|
| Logistic regression (splines) | 0.785 | 0.0290 |
| Gradient boosting | 0.776 | 0.0291 |
| ADA Risk Test | 0.754 | – |
| Adapted FINDRISC | 0.752 | – |

Final XGBoost configuration: 800 trees, learning rate 0.018, **depth 1**, min child weight 3.5, subsample 0.96, column subsample 0.53, L1 0.53, L2 0.0016. Depth 1 means the boosted model is additive (no interactions). The search range for depth and L2 was extended once, as registered.

Locked H2 thresholds (nearest 90% development sensitivity):

| Strategy | Threshold | Development sensitivity |
|---|---|---|
| Gradient boosting | predicted risk ≥ 0.01642 | 90.2% |
| Logistic regression | predicted risk ≥ 0.01706 | 90.2% |
| ADA Risk Test | score ≥ 4 | 93.3% |
| Adapted FINDRISC | score ≥ 9 | 89.6% |

## Deviations from the registration

None so far.

---

# Decisions log: validation evaluation (one-time)

Script: `scripts/evaluate_validation.py` · Run: 14 October 2026, about 4 minutes · Seed 20261015 · Bootstrap 2,000 replicates (1,000 for calibration)
The script was run once. No analysis was re-run or changed after the validation results were seen.

## Implementation details the registration did not specify

11. **Integer scores across imputations.** Each participant's ADA and FINDRISC scores are averaged over the 10 imputations, and the average is compared with the cutoff. Model predictions are averaged in the same way.
12. **Matched referral rate (H1).** r = share of validation participants with ADA ≥5 (unweighted, 49.6%); k = round(r × n) = 3,296. Both models refer their top k. Ties are broken at random with a fixed seed. The weighted ADA referral rate (43.5%, build log) is the population figure; the matched rate is defined on the sample, because the comparison is paired.
13. **FINDRISC at the matched rate.** An integer score cannot hit r exactly, so the cutoff with the nearest referral rate was used (≥11, referral 50.3%).
14. **Bootstrap.** Stratified by outcome; each replicate draws one imputation at random and recomputes r, k and the referrals inside the replicate.
15. **Weighted robustness analysis.** Referral is classified as in the main analysis (unweighted); the weighted FNR among cases and its difference are estimated with MEC weights and Taylor linearisation (strata and PSU).
16. **Threshold decision table.** F1, Youden and cost-based (FN:FP = 10:1) thresholds were chosen on development out-of-fold predictions (grid of 400 quantiles) and applied unchanged to validation. For both models the F1 and cost-based rules happened to select the same threshold.
17. **E3 (2021–2023).** Family history and gestational diabetes are not collected in 2021–2023, so reduced models without these two items were refitted on development with the same pipeline (including tuning and the range-extension rule) and evaluated at the primary-validation referral rate. Total activity minutes in 2021–2023 come from the new PAD items.

18. **Imputation model covariates** (noted 14 October 2026 while writing the methods, applies to both stages). Each item's imputation model used all other predictors and, in training data, the outcome. Survey-design variables (strata, PSU, weights) and auxiliary variables were not added; the registration included them only "where appropriate". Report this in the methods.

## Deviations from the registration

- The registered fallback for non-convergence of logistic regression was Firth's correction; the code uses ridge (C = 100) instead. It was never triggered, so no result is affected. Report as a minor deviation.
- No other deviations.

## Correction after the run (figure only)

The smoothed lines in `fig_calibration.png` were drawn with lowess robustness iterations, which treat the rare cases as outliers and flatten the curve to zero. The figure was redrawn from the saved predictions with `it=0` (no robustness weighting), and the script was corrected to match. No number in the results changed.

## Reporting additions (14 October 2026, after the validation run)

- **Table A.6 (logistic regression equation).** To report the locked model (TRIPOD+AI item 22), the coefficients of the 10 locked logistic regressions were averaged into one equation (`scripts/lr_coef.py`). This is a presentation of the locked model, not a new analysis. As a reporting check, the averaged equation was compared with the study's averaged predictions in primary validation: AUC 0.782 for both, largest difference in predicted risk 0.0025. No registered result changed.
- **Fig. A.1** redrawn at 300 dpi from the locked model (`scripts/fig_a1.py`), plotting observed values only. Mean absolute SHAP values matched the reported values.
- **Not done (post hoc, not registered):** the Asian–White false-negative-rate comparison with the HbA1c-or-glucose outcome. The HbA1c measurement issue is reported as a limitation instead.
- **Table A.3 confidence intervals** (added 14 October 2026). Wilson 95% intervals for the registered exploratory subgroup false-negative rates, computed from the saved counts. Reporting of precision only; no estimate changed. The intervals do not account for imputation.

---

# Post hoc analyses after co-author review (14 November 2026)

Script: `scripts/posthoc_analyses.py` · Seed 20261114 · Output: `results/posthoc_results.json`, `results/posthoc_run_log.txt`
These analyses are NOT registered. They were run after the registered one-time validation, in response to a co-author's review, and are reported as exploratory (Appendix Table A.7). The script first reproduces the registered predictions (largest difference 1e-16) and the registered H1 counts (29, 30, 36, 40 missed).

- **P1. Without told prediabetes.** Both models refitted on development without the item (same pipeline, tuning included). Missed of 237: gradient boosting 41, logistic regression 34, ADA 36, FINDRISC 40.
- **P2. Paired DeLong tests of AUC.** Both models higher than both scores (p <= 0.012); gradient boosting vs logistic regression p = 0.44.
- **P3. 2021-2023 without the changed items** (activity, antihypertensive medication). Missed of 128: gradient boosting 10, logistic regression 8 (p = 0.50).
- **P4. Asian-White gap with glucose-based outcomes, fasting subsample.** Near zero for both models under all three outcome definitions; 13 to 21 points for the ADA Risk Test. This reverses the earlier decision (above) not to run this comparison.

Not run: retuning with a wider tree range or deeper trees (would build a new model after the validation results were known).
