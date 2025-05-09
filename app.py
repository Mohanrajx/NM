import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

import streamlit as st
from nrclex import NRCLex
import matplotlib.pyplot as plt
import matplotlib

# Use a nicer font for charts
matplotlib.rcParams.update({'font.size': 12, 'axes.titlesize': 16, 'axes.labelsize': 14})

# App configuration
st.set_page_config(page_title="Emotion Detector", layout="centered", page_icon="😊")

# App title and description
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>🎭 Emotion Detector using NRCLex</h1>
    <p style='text-align: center; font-size: 18px;'>Analyze the emotional tone of your text in real-time</p>
    <hr style='border: 1px solid #4CAF50;'>
""", unsafe_allow_html=True)

# Text input
user_input = st.text_area("✏️ Enter Text Below", "I am excited and hopeful about the future!", height=150)

# Button to analyze
if st.button("🔍 Analyze Emotions"):
    if user_input.strip():
        emotion = NRCLex(user_input)
        scores = emotion.raw_emotion_scores
        top = emotion.top_emotions

        st.success("✅ Analysis Complete")

        # Display top emotions
        st.subheader("🔥 Top Detected Emotions")
        if top:
            for em, val in top:
                st.markdown(f"<span style='font-size:18px;'>💡 <b>{em.capitalize()}</b>: {val*100:.1f}%</span>", unsafe_allow_html=True)
        else:
            st.info("No dominant emotions found.")

        # Display chart
        if scores:
            st.subheader("📊 Emotion Score Chart")
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(scores.keys(), scores.values(), color='#4CAF50')
            ax.set_title("Emotion Distribution", fontsize=16)
            ax.set_ylabel("Frequency")
            ax.set_xlabel("Emotions")
            ax.set_xticks(range(len(scores)))
            ax.set_xticklabels(scores.keys(), rotation=45)
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
            st.pyplot(fig)
        else:
            st.warning("No emotion scores to display.")
    else:
        st.warning("⚠️ Please enter some text to analyze.")

# Footer
st.markdown("""
    <hr>
    <p style='text-align: center;'>Created by <b>G Mohanraj</b> | Department of AI & DS | Kingston Engineering College</p>
""", unsafe_allow_html=True)
