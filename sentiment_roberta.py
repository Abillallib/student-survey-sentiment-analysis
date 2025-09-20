import pandas as pd
from transformers import pipeline

# Load Excel file
file_path = r"C:\path\exit_data_sentiments.xlsx"  # Update path if needed
sheet_name = "sheet1"
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Load RoBERTa sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# Columns to analyze
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
]

# Function to get sentiment score and type
def analyze_sentiment_roberta(text):
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return pd.Series([None, None])

    result = sentiment_pipeline(text)[0]
    return pd.Series([result["score"], result["label"]])

# Apply sentiment analysis to each column
for col in sentiment_columns:
    df[[f"{col}_compound", f"{col}_sentiment"]] = df[col].apply(analyze_sentiment_roberta)

# Save results back to the same Excel file
with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Sentiment analysis completed. Results saved to {file_path}")

# LABEL_0 → Negative
# LABEL_1 → Neutral
# LABEL_2 → Positive

