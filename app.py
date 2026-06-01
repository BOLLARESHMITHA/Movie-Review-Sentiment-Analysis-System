import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tf_keras.models import load_model
from tf_keras.datasets import imdb
from tf_keras.preprocessing.sequence import pad_sequences

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Movie Review Sentiment Analysis System",
    page_icon="🎬",
    layout="wide"
)

# ==================================================
# LOAD MODELS
# ==================================================

@st.cache_resource
def load_models():
    rnn_model = load_model("simple_rnn_model.keras")
    lstm_model = load_model("lstm_model.keras")
    gru_model = load_model("gru_model.keras")
    return rnn_model, lstm_model, gru_model

rnn_model, lstm_model, gru_model = load_models()

# ==================================================
# IMDB WORD INDEX
# ==================================================

word_index = imdb.get_word_index()

VOCAB_SIZE = 10000
MAX_LENGTH = 200

# ==================================================
# TEXT TO SEQUENCE
# ==================================================

def text_to_sequence(text):
    text = text.lower()
    words = text.split()
    sequence = []
    for word in words:
        idx = word_index.get(word)
        if idx is not None and idx < VOCAB_SIZE:
            sequence.append(idx + 3)
    padded = pad_sequences(
        [sequence],
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )
    return padded

# ==================================================
# PREDICTION FUNCTION
# ==================================================

def predict_sentiment(model, text):
    sequence = text_to_sequence(text)
    positive_prob = float(model.predict(sequence, verbose=0)[0][0])
    negative_prob = 1 - positive_prob
    sentiment = "Positive 😊" if positive_prob > 0.5 else "Negative 😞"
    confidence = max(positive_prob, negative_prob) * 100
    return sentiment, confidence, positive_prob, negative_prob

# ==================================================
# HEADER
# ==================================================

st.title("🎬 Movie Review Sentiment Analysis System")
st.subheader("Deep Learning Based Sentiment Classification")
st.markdown("---")

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.header("📚 Project Information")
st.sidebar.write("""
Dataset : IMDB Movie Reviews

Models:
- SimpleRNN
- LSTM
- GRU

Classes:
- Positive
- Negative
""")

# ==================================================
# INPUT AREA
# ==================================================

review = st.text_area("Enter your movie review here...", height=180)
selected_model = st.radio("Select Model", ["SimpleRNN", "LSTM", "GRU"])
predict_btn = st.button("Analyze Review")

# ==================================================
# PREDICTION
# ==================================================

if predict_btn:

    if review.strip() == "":
        st.warning("Please enter a review.")
        st.stop()

    if selected_model == "SimpleRNN":
        current_model = rnn_model
    elif selected_model == "LSTM":
        current_model = lstm_model
    else:
        current_model = gru_model

    sentiment, confidence, positive_prob, negative_prob = predict_sentiment(current_model, review)

    # ==================================================
    # OUTPUT AREA
    # ==================================================

    st.markdown("## Prediction Result")
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"Sentiment: {sentiment}")
    with col2:
        st.info(f"Confidence: {confidence:.2f}%")

    # ==================================================
    # VISUALIZATION AREA
    # ==================================================

    st.markdown("## Probability Distribution")
    chart_df = pd.DataFrame({
        "Class": ["Positive", "Negative"],
        "Probability": [positive_prob, negative_prob]
    })

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(chart_df["Class"], chart_df["Probability"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Positive vs Negative Probability")
    st.pyplot(fig)

    # ==================================================
    # PIE CHART
    # ==================================================

    st.markdown("## Confidence Chart")
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.pie(chart_df["Probability"], labels=chart_df["Class"], autopct="%1.1f%%")
    ax2.set_title("Probability Distribution")
    st.pyplot(fig2)

    # ==================================================
    # MODEL COMPARISON
    # ==================================================

    st.markdown("## Model Comparison")
    rnn_result = predict_sentiment(rnn_model, review)
    lstm_result = predict_sentiment(lstm_model, review)
    gru_result = predict_sentiment(gru_model, review)

    comparison_df = pd.DataFrame({
        "Model": ["SimpleRNN", "LSTM", "GRU"],
        "Sentiment": [rnn_result[0], lstm_result[0], gru_result[0]],
        "Confidence (%)": [
            round(rnn_result[1], 2),
            round(lstm_result[1], 2),
            round(gru_result[1], 2)
        ]
    })

    st.dataframe(comparison_df, use_container_width=True)

    # ==================================================
    # CONFIDENCE COMPARISON GRAPH
    # ==================================================

    st.markdown("## Confidence Comparison Chart")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.bar(comparison_df["Model"], comparison_df["Confidence (%)"])
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("Confidence (%)")
    ax3.set_title("Model Confidence Comparison")
    st.pyplot(fig3)

    # ==================================================
    # BEST MODEL
    # ==================================================

    best_model = comparison_df.loc[comparison_df["Confidence (%)"].idxmax()]
    st.success(
        f"🏆 Best Performing Model: {best_model['Model']} ({best_model['Confidence (%)']}%)"
    )

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")
st.caption("Developed using TensorFlow, Keras, Streamlit and IMDB Dataset")
