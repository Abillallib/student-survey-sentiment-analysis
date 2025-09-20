import pandas as pd
from openpyxl import load_workbook
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import text2emotion as te  # For multi-emotion extraction

# Load Excel file
file_path = r"path\exit_data_sentiments.xlsx"  # Update path as needed
sheet_name = "sheet1"

# Read the Excel sheet
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# List of columns to analyze
sentiment_columns = [
    "CAMPUS_ENVIRONMENT",
    "MORE_DETAILS",
    "RESIDENCE_HALLS",
    "DINING_SERVICES",
    "SOCIAL_LIFE",
    "CAMPUS_ACTIVITIES",
    "ADVISING_TUTORING",
    "BELONGING"
]

# Analyze each comment for sentiments and emotions
for col in sentiment_columns:
    #sentiment_scores = []
    emotion_lists = []

    for comment in df[col].fillna(''):  # Handle NaNs as empty strings
        # Sentiment scores (VADER)
        #sentiment = analyzer.polarity_scores(comment)
        #sentiment_scores.append(sentiment)

        # Emotion detection (text2emotion)
        emotions = te.get_emotion(comment)
        # Convert emotion dict to list format: ['Emotion: score', ...]
        emotion_list = [f"{emotion}: {score:.2f}" for emotion, score in emotions.items() if score > 0]
        emotion_lists.append(emotion_list)

    # Store sentiment and emotion outputs in new columns
    #df[f"{col}_SentimentScores"] = sentiment_scores
    df[f"{col}_Emotions"] = emotion_lists

# Save the updated dataframe back to Excel
with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    df.to_excel(writer, sheet_name=sheet_name, index=False)

print("Sentiment and emotion analysis complete. Data saved to Excel.")


