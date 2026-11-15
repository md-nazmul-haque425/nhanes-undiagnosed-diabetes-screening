# Detecting undiagnosed diabetes at a matched referral rate (NHANES): analysis code

Code, locked models and results for the paper:

> Haque MN, Ismail S, Husin HS, Talukder MGM, Ahmed AU, Noor NU, Hossain MA. *Gradient boosting, logistic regression and risk scores for detecting undiagnosed diabetes at a matched referral rate: development and preregistered temporal validation in a US national survey.* Submitted to *Informatics and Health*.

Preregistration: Open Science Framework, https://osf.io/w2hsa

## What the study does

Short questionnaires decide who is offered a diabetes test. We compared four ways of choosing who to test: gradient boosting (XGBoost), spline logistic regression, the ADA Risk Test and an adapted FINDRISC. All four used the same 11 questionnaire and body-size items, and no blood test. We developed the models on NHANES 2011–2016, locked them, and evaluated them once on NHANES 2017–March 2020. The primary outcome was the share of undiagnosed diabetes cases (HbA1c ≥6.5%) each strategy missed when all of them referred the same share of adults.

## Repository contents

| Folder or file | What it holds |
|---|---|
| `nhanes_file_list.csv` | The 50 NHANES files used, with download links |
| `download_nhanes.ps1`, `Download NHANES files.bat` | Download the NHANES files on Windows (double-click the .bat file) |
| `scripts/` | All analysis code (see the steps below) |
| `analysis/` | Analysis files built from NHANES: development, primary validation and secondary validation samples, codebook, variable mapping and build log |
| `models/` | Locked models (`locked_models.joblib`), development report, tuning results, out-of-fold predictions, run log and decisions log |
| `results/` | Validation results (`validation_results.json`), post hoc results (`posthoc_results.json`), tables, figures and run logs |
| `requirements.txt` | Python package versions |

The raw NHANES `.xpt` files are not included. They are public and can be downloaded with the script above, or from the links in `nhanes_file_list.csv` on any system.

## How to reproduce

Run every command from the repository root, with Python 3.13 and the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

| Step | Command | Output | Paper |
|---|---|---|---|
| 1. Download NHANES files | `download_nhanes.ps1` (Windows) | one folder of `.xpt` files per cycle | |
| 2. Build analysis files | `python scripts/build_analysis_dataset.py` | `analysis/` | Fig. 1, Table 1 |
| 3. Develop and lock models (development data only) | `python scripts/fit_development_models.py analysis/development_2011_2016.csv models` | `models/` | Appendix Tables A.1, A.2 |
| 4. One-time validation | `python scripts/evaluate_validation.py analysis/development_2011_2016.csv analysis/validation_2017_2020.csv analysis/validation2_2021_2023.csv models results` | `results/` | Tables 2–5, Figs. 2–4, Appendix Tables A.4–A.6 |
| 5. Logistic regression equation | `python scripts/lr_coef.py` | `results/lr_coefficients.csv`, `results/lr_knots.json` | Appendix Table A.3 |
| 6. Post hoc analyses | `python scripts/posthoc_analyses.py analysis/development_2011_2016.csv analysis/validation_2017_2020.csv analysis/validation2_2021_2023.csv models results` | `results/posthoc_results.json` | Appendix Table A.7 |
| 7. Manuscript figures | `python scripts/make_figures.py` and `python scripts/fig_a1.py` | `figures/` (300 dpi PNG and PDF) | Figs. 1–4, Fig. A.1 |

Steps 2 to 4 take about 30 minutes on a laptop. The analysis files, locked models and results are already included, so steps 5 to 7 can run without repeating steps 1 to 4. Re-running step 3 or 4 overwrites the files in `models/` or `results/`.

## Notes

- **Locked models.** `models/locked_models.joblib` holds the imputation models, spline knots, gradient-boosting models, Platt recalibration and logistic regressions for all 10 imputations. Loading it needs `scripts/fit_development_models.py` on the Python path, as the other scripts do.
- **Registered and post hoc analyses.** Steps 3 and 4 follow the OSF preregistration and were each run once. Step 6 was added after a co-author's review. It is exploratory and not registered. `models/decisions_log.md` records every implementation choice, the one deviation and the post hoc additions.
- **Path edits.** In `lr_coef.py`, `make_figures.py` and `fig_a1.py`, only the input and output paths were changed so that the scripts run from the repository root. The analysis code is unchanged.
- **Random seeds.** 20261014 (development), 20261015 (validation), 20261114 (post hoc).

## Data source and licence

NHANES data are produced by the US National Center for Health Statistics and are in the public domain. The NHANES protocols were approved by the NCHS Ethics Review Board, and all participants gave written informed consent. The code in this repository is released under the MIT Licence (see `LICENSE`).

## Contact

Md Nazmul Haque, Malaysian Institute of Information Technology, Universiti Kuala Lumpur. Email: md.haque@s.unikl.edu.my. ORCID: 0009-0001-1086-0624.
