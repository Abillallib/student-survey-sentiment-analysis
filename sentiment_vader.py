import pandas as pd
from openpyxl import load_workbook
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



# Load Excel file
file_path = r"C:\path\exit_data_sentiments.xlsx"  # Change this to your actual file
sheet_name = "d3"

# Read the Excel table
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

# Function to analyze sentiment
def analyze_sentiment(text):
    if pd.isna(text) or not isinstance(text, str):  # Handle NaN values
        return pd.Series([ None, None])  # Return None for all

    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]

    # Classification based on compound score
    if compound >= 0.05:
        sentiment_class = "Positive"
    elif compound <= -0.05:
        sentiment_class = "Negative"
    else:
        sentiment_class = "Neutral"

    return pd.Series([ compound, sentiment_class]) #scores["pos"], scores["neu"], scores["neg"],


# Function to extract lexicon sentiment words for a given text
def extract_lexicon_words(text):
    if pd.isna(text) or not isinstance(text, str):  # Handle NaN values
        return {} #pd.Series([None, None, None])  # Return None for all
    
    words_sentiment = {}
    words = text.lower().split()  # Split text into words
    for word in words:
        if word in analyzer.lexicon:  # Check if word exists in the lexicon
            words_sentiment[word] = analyzer.lexicon[word]

    return words_sentiment

# Apply sentiment analysis to each column and extract lexicon words
all_lexicon_words = []



# Apply sentiment analysis to each column
for col in sentiment_columns:
    df[[ f"{col}_compound", f"{col}_sentiment"]] = df[col].apply(analyze_sentiment) #f"{col}_pos", f"{col}_neu", f"{col}_neg",

    # Extract lexicon words and their sentiment scores
    for index, row in df.iterrows():
        words_sentiment = extract_lexicon_words(row[col])
        if len(words_sentiment) > 0:  # Check if the list is not empty
            all_lexicon_words.append([index, row[col], words_sentiment])


# Convert lexicon word results to a DataFrame
lexicon_df = pd.DataFrame(all_lexicon_words, columns=["Index", "Word", "SentimentScore"])

# Merge the lexicon data back into the main DataFrame, aligning it with the corresponding rows
df_merged = pd.merge(df, lexicon_df, left_index=True, right_on="Index", how="left")

# Load the existing Excel file to append data
book = load_workbook(file_path)

# Save results back to the same file
with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    df_merged.to_excel(writer, sheet_name=sheet_name, index=False)
    #lexicon_df.to_excel(writer, sheet_name="LexiconWords", index=False)  # Save the lexicon words table

print(f"Sentiment analysis completed. Results saved back to {file_path}")

