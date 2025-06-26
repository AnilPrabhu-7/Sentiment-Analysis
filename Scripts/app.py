# app.py

import streamlit as st

# ✅ Only imports predict_sentiment (model/tokenizer cached inside)
from Sentiment_Analysis import predict_sentiment

st.set_page_config(page_title="Text Sentiment Analyzer", layout="centered")
st.title("\U0001F4AC Text Sentiment Analyzer")

# Option 1: Web Scraping Section
# st.subheader("Option 1: 🌐 Scrape Content from a Webpage")
# url = st.text_input("Enter a webpage URL:", "https://en.wikipedia.org/wiki/Sentiment_analysis")

# if st.button("Fetch Web Content"):
#     try:
#         import requests
#         from bs4 import BeautifulSoup

#         response = requests.get(url)
#         soup = BeautifulSoup(response.text, 'html.parser')
#         paragraphs = soup.find_all('p')
#         content = ' '.join([para.get_text() for para in paragraphs])

#         if content:
#             st.success("Web content fetched successfully!")
#             st.text_area("Fetched Content", content, height=200)
#             result = predict_sentiment(content)
#             st.markdown(f"### Sentiment: :blue[**{result}**]")
#         else:
#             st.warning("No content found on the webpage.")
#     except Exception as e:
#         st.error(f"Failed to fetch content: {e}")

# Option 2: Manual Text Input
st.subheader("📝 Type a Sentence")
manual_text = st.text_area("Type or paste a sentence here for sentiment analysis:", height=150)

if st.button("Analyze Text Sentiment"):
    if manual_text.strip():
        result = predict_sentiment(manual_text)
        color = "green" if "positive" in result.lower() else "red" if "negative" in result.lower() else "orange"
        st.markdown(f"<div style='background-color:{color};padding:10px;border-radius:5px;color:white;'>\n<b>Sentiment {result}</b></div>", unsafe_allow_html=True)
    else:
        st.warning("Please enter some text to analyze.")
