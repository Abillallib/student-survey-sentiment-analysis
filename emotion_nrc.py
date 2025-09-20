import pandas as pd
from nrclex import NRCLex
import nltk

# Download tokenizer for NRCLex
nltk.download('punkt')

# Load Excel file
file_path = r"C:\path\exit_data_sentiments.xlsx"  # Update path as needed
sheet_name = "sheet1"

# Read the Excel sheet
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Define columns for sentiment analysis
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

# Process each column for emotion analysis
for col in sentiment_columns:
    # Create a new column to store emotion results
    df[f"{col}_Emotions"] = None
    
    # Process each comment
    for index, row in df.iterrows():
        comment = str(row[col]) if pd.notna(row[col]) else ""
        
        if comment.strip():
            # Analyze emotions using NRCLex
            emotion_analyzer = NRCLex(comment)
            
            # Get emotion frequencies
            emotions = emotion_analyzer.affect_frequencies

            # Filter out positive and negative if you only want the 8 basic emotions
            emotions = {k: v for k, v in emotions.items() if k not in ['positive', 'negative']}
            
            # Format the emotion list as "emotion: score"
            emotion_list = [f"{emotion}: {round(score, 3)}" for emotion, score in emotions.items() if score > 0]
            
            # Store in dataframe
            df.at[index, f"{col}_Emotions"] = ', '.join(emotion_list)
        else:
            df.at[index, f"{col}_Emotions"] = ""


# Save the updated dataframe back to Excel
with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    df.to_excel(writer, sheet_name=sheet_name, index=False)

print("Sentiment and emotion analysis complete. Data saved to Excel.")

