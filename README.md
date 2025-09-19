# Student Survey Sentiment & Emotion Analysis

This project applies Natural Language Processing (NLP) techniques to analyze student exit survey responses. The goal is to extract actionable insights from open-ended feedback, enabling administrators to identify strengths, weaknesses, and emotional drivers behind the student experience.

The analysis feeds into an interactive dashboard that visualizes trends in sentiment and emotions across student cohorts.

Project Workflow

Data Preparation
- Student exit survey responses (open-text fields) are collected.
- Data is cleaned (removing stopwords, punctuation, and duplicates).
- Placeholder/mock data is provided in this repo to demonstrate the workflow.

Keyword Extraction & Word Cloud

- Used frequency analysis and keyword extraction methods.
- Generated word clouds to visualize the most common themes.

Sentiment Analysis

Applied two methods:

- VADER (lexicon-based)
- RoBERTa transformer model (deep learning)

Finding: RoBERTa produced more accurate classifications of Positive, Neutral, Negative sentiment compared to VADER.

Emotion Analysis

Applied two approaches:

- Text2Emotion (Python library for emotion tagging)
- NRC Emotion Lexicon (dictionary-based method)

Finding: Text2Emotion provided more reliable emotion ratings (Joy, Anger, Sadness, Surprise, Fear).

Dashboard Integration

- Results were designed for visualization in a Power BI dashboard.
- Dashboard enables interactive filtering by sentiment, emotion, or keyword themes.

# Outcomes
- Identified key themes students most frequently mentioned in exit surveys.
- Classified responses into positive, neutral, and negative categories.
- Mapped emotions (joy, anger, sadness, etc.) to better understand the emotional drivers of student experience.
- Produced a workflow that can be reused for other survey or feedback datasets.
