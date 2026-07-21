# Machine Learning
# Aim - Predict Power Plant Energy Output (PE)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

st.set_page_config(page_title="Power Plant PE", layout="wide")

@st.cache_resource
def train_models():
    df = pd.read_csv("power_plant_regression.csv").drop_duplicates().reset_index(drop=True)

    X = df[["AT", "V", "AP", "RH"]]
    Y = df["PE"]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
        "SVR": SVR(),
        "KNN": KNeighborsRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }

    scores = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        scores[name] = r2_score(y_test, model.predict(X_test))

    return models, scores, scaler

models, scores, scaler = train_models()

if "history" not in st.session_state:
    st.session_state.history = []

st.title("Power Plant Energy Output Prediction")

c1, c2, c3, c4 = st.columns(4)

AT = c1.number_input("AT", 1.81, 37.11, 15.00, step=0.01)
V = c2.number_input("V", 25.36, 81.56, 40.00, step=0.01)
AP = c3.number_input("AP", 992.89, 1033.30, 1010.00, step=0.01)
RH = c4.number_input("RH", 25.56, 100.16, 75.00, step=0.01)

b1, b2 = st.columns([1, 1])

predict_button = b1.button("Predict", use_container_width=True)
clear_button = b2.button("Clear History", use_container_width=True)

if clear_button:
    st.session_state.history = []
    st.rerun()

def make_chart(prediction_number, predictions):
    best = max(scores, key=scores.get)
    values = list(predictions.values())

    fig, ax = plt.subplots(figsize=(5.8, 2.7))

    colors = ["#2ca02c" if name == best else "#1f77b4" for name in predictions]

    ax.bar(range(len(predictions)), values, color=colors)

    ax.set_xticks(range(len(predictions)))
    ax.set_xticklabels([name[:3] for name in predictions], rotation=45, ha="right", fontsize=7)

    ax.set_title(f"Prediction {prediction_number} | Best: {best}", fontsize=8)
    ax.set_ylim(min(values) - 5, max(values) + 18)

    for i, (name, value) in enumerate(predictions.items()):
        ax.text(i, value, f"{value:.1f}\nR2 {scores[name]:.2f}", ha="center", va="bottom", fontsize=6)

    fig.tight_layout()
    return fig

if predict_button:
    input_df = pd.DataFrame([[AT, V, AP, RH]], columns=["AT", "V", "AP", "RH"])
    input_scaled = scaler.transform(input_df)

    predictions = {}

    for name, model in models.items():
        predictions[name] = model.predict(input_scaled)[0]

    st.session_state.history.append(predictions)

if st.session_state.history:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Latest Prediction")
        latest = st.session_state.history[-1]
        fig = make_chart(len(st.session_state.history), latest)
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Stacked History")

        with st.container(height=300):
            for i, predictions in enumerate(st.session_state.history, start=1):
                fig = make_chart(i, predictions)
                st.pyplot(fig)
                plt.close(fig)

else:
    st.info("Enter inputs and click Predict.")