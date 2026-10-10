# FPL Top-5 Recommender

🔗 **Live demo:** https://premier-league-fpl-suggestion-model-fccybpydfqgkdmufvxm4hy.streamlit.app/

## What it does
Given a budget, position, and gameweek, returns the top 5 recommended
Fantasy Premier League players, ranked by a machine learning model.

## Data
Historical FPL player-gameweek data (41 features per row, ~30k rows).

## Approach
- Built lagged form features (e.g., points_last3_avg, minutes_last5_avg) using
  a shift+rolling window per player, avoiding target leakage.
- Temporal train/test split (GW1–30 train, GW31–38 test).
- XGBoost regressor predicting `total_points`, tuned with TimeSeriesSplit CV.

## Leakage hunting (the interesting part)
Three separate leakage sources were found and removed:
1. `influence`, `creativity`, `threat` — post-match stats that leak the target.
2. `expected_goals`, `expected_assists`, `expected_goal_involvements` — per-match, not rolling.
3. Re-checking after each removal until metrics matched realistic FPL benchmarks.

A label permutation test confirmed no leakage: shuffled-target R² = −0.04.

## Results
- MAE: 1.23 (baseline: 1.41) — 15% lift
- R²: 0.12
- Mean per-GW Spearman: 0.41
- Precision@5: 0.03 (low — top-5 scorers are outliers)

## Known limitations
- Regression-to-the-mean models miss GW outliers (top-5 has low precision).
- Early-season predictions (GW1–5) rely on NaN lag features; the model falls back
  on non-lagged signals.
- Model is trained on a static snapshot; no live retraining pipeline yet.

## How to run locally
pip install -r requirements.txt
streamlit run app.py