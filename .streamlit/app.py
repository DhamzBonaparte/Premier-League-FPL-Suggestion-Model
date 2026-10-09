import streamlit as st
import pandas as pd
import joblib
import numpy as np


@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    ohe = joblib.load("transform.pkl")
    features = joblib.load("features.pkl")
    return model, ohe, features


@st.cache_data
def load_Data():
    return pd.read_parquet("data.parquet")


model, ohe, features = load_artifacts()
data = load_Data()


st.title("Fantasy Premier League Top 5 Player Recommender")
st.write(
    "Pick your budget, gameweek and position to get the top 5 recommended players for your fantasy football team."
)

col1, col2, col3 = st.columns(3)

with col1:
    budget = st.slider(
        "Budget (in £m)", min_value=4.0, max_value=15.0, value=8.0, step=0.5
    )

with col2:
    position = st.selectbox("Position", ["GK", "DEF", "FWD", "MID"])

with col3:
    gw = st.selectbox("Gameweek", sorted(data["GW"].unique()))


def recommend(budget, position, gw, data, model):
    budget_tenth = int(budget * 10)

    candidates = data[
        (data["GW"] == gw)
        & (data["position"] == position)
        & (data["value"] <= budget_tenth)
    ].copy()

    if candidates.empty:
        return None

    pos_encoded = ohe.transform(data[["position"]])
    pos_data = pd.DataFrame(
        pos_encoded,
        columns=pos_encoded.get_feature_names_out(["position"]),
        index=candidates.index,
    )

    X = pd.concat(candidates.drop(columns=["position", "name"]), pos_data, axis=1)

    candidates["pred"] = model.predict(X)

    return candidates.nlargest(5, "pred")[["name", "value","pred"]]

