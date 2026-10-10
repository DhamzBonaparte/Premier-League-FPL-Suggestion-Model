# ⚽ FPL Top-5 Player Recommender

Predicts Fantasy Premier League player points and recommends the top 5 players
for a given budget, position, and gameweek.

🔗 **Live app:** https://premier-league-fpl-suggestion-model-fccybpydfqgkdmufvxm4hy.streamlit.app/

---

## What it does

User picks:
- Budget (e.g., £8.0m)
- Position (GK / DEF / MID / FWD)
- Gameweek

The app returns the top 5 matching players, ranked by a trained XGBoost model.

---

## Approach

- **Features:** lagged form features built with per-player rolling windows
  (`points_last3_avg`, `minutes_last5_avg`, `goals_last5_avg`, etc.) so every
  feature is known *before* kickoff.
- **Model:** XGBoost regressor predicting `total_points`, tuned with
  GridSearchCV + TimeSeriesSplit.
- **Split:** GW1–30 train, GW31–38 test (temporal, not random).
- **Leakage:** three separate leakage sources found and removed; verified with
  a label permutation test (shuffled-target R² = −0.04).
- **Deployment:** Streamlit, with the model, OneHotEncoder, and feature list
  pickled for serving.

---

## Results

| Metric | Value | Baseline |
|---|---|---|
| MAE | **1.23** | 1.41 (predict-mean) |
| R² | 0.12 | — |
| Mean per-GW Spearman | 0.41 | — |

Beats the naive baseline by ~15%.

---

## Run locally

```bash
git clone https://github.com/DhamzBonaparte/Premier-League-FPL-Suggestion-Model
cd Premier-League-FPL-Suggestion-Model
pip install -r requirements.txt
streamlit run app.py