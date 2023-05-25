import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("articles.csv", encoding="latin1")

articles = df["Article"].tolist()
tfidf = TfidfVectorizer(stop_words="english")
uni_matrix = tfidf.fit_transform(articles)
cos_sim = cosine_similarity(uni_matrix)

def recommended_articles(title, n=5):
	article_index = df[df["Title"] == title].index[0]
	article_similarities = cos_sim[article_index]
	top_indices = article_similarities.argsort()[::-1][1:n+1]
	recommended_titles = [df["Title"].loc[i] for i in top_indices]
	return recommended_titles

st.title("Article Recommendation System")

user_title = st.text_input("Title")
n = st.number_input("Recommend Count")

if st.button("Recommend"):
	recommended_titles = recommended_articles(user_title, int(n))
	st.write("Onerilen Makaleler:")
	for title in recommended_titles:
		st.success("- " + title)

