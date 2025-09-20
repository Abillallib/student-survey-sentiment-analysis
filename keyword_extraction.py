import pandas as pd
import re
import string
import yake
import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from textblob import TextBlob
from rake_nltk import Rake
from sklearn.feature_extraction.text import TfidfVectorizer

# Load spaCy model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")


# Load NLTK stopwords
stop_words = set(stopwords.words('english'))

# Load Excel file
file_path = r"C:\path\exit_data_sentiments.xlsx"  # Change to your file 
sheet_name = "sheet1"  # Change to your sheet name
df = pd.read_excel(file_path, sheet_name=sheet_name)

# List of text columns to process
text_columns = [
    "CAMPUS_ENVIRONMENT",
    "MORE_DETAILS",
    "RESIDENCE_HALLS",
    "DINING_SERVICES",
    "SOCIAL_LIFE",
    "CAMPUS_ACTIVITIES",
    "ADVISING_TUTORING",
    "BELONGING"
]


# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Function to handle negations using TextBlob
def handle_negation_spacy(text):
    doc = nlp(text)
    processed_words = []
    negation = False  # Track if negation is active

    for token in doc:
        # Check if the token is a negation word
        if token.dep_ == 'neg':
            negation = True
            continue

        # Prefix "not_" if negation is active and reset after punctuation or conjunctions
        if negation:
            processed_words.append(f"not_{token.lemma_}")
            if token.dep_ in ['punct', 'cc', 'conj']:
                negation = False
        else:
            processed_words.append(token.lemma_)

    return " ".join(processed_words)

# Function 1: Text Preprocessing
def clean_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""

    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    
    text = handle_negation_spacy(text)  # Preserve negations
    words = word_tokenize(text)  # Tokenize words
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatize & remove stopwords
    words = [stemmer.stem(word) for word in words]  # Stemming
    return ' '.join(words)

# Duplicate columns before applying changes
for col in text_columns:
    df[col + "_Original"] = df[col]  # Keep a copy of the original text
    df[col] = df[col].astype(str).apply(clean_text)  # Apply text cleaning to the duplicate column

# Function 2: Extract keywords using TF-IDF with a minimum threshold
def extract_tfidf_keywords(df, column):
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=50)
    vectorizer.fit(df[column].fillna("").astype(str))

    def get_keywords(text):
        tfidf_vec = vectorizer.transform([text]).toarray().flatten()
        feature_names = vectorizer.get_feature_names_out()
        indices = [i for i in tfidf_vec.nonzero()[0]]
        return ", ".join([feature_names[i] for i in indices])

    return df[column].astype(str).apply(get_keywords)

# Function 3: Extract Keywords with YAKE
def extract_yake_keywords(text, top_n=10):
    kw_extractor = yake.KeywordExtractor(top=top_n)
    keywords = kw_extractor.extract_keywords(text)
    return [kw[0] for kw in keywords]  # Only keep keyword text

# Function 4: Extract Named Entities (NER)
def extract_named_entities(text):
    if not isinstance(text, str) or pd.isna(text):  # Ensure text is a valid string
        return []  # Return empty list if text is invalid
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PERSON", "GPE"]]  # Extract relevant entities


# Function 5: Extract Rake-NLTK Keywords
def extract_rake_keywords(text):
    rake = Rake()
    text = str(text)
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()[:10]  # Get top 10 ranked phrases

# Function 6: Extract Sentiment-Weighted Words
def extract_sentiment_words(text):
    text = str(text)
    blob = TextBlob(text)
    return [word for word in blob.words if TextBlob(word).sentiment.polarity != 0]  # Keep words with sentiment

# Function 7: Combine and Deduplicate Keywords
def combine_keywords(text, df, column):
    tfidf_kw = extract_tfidf_keywords(df, column).iloc[0].split(", ")
    yake_kw = extract_yake_keywords(text)
    rake_kw = extract_rake_keywords(text)
    ner_kw = extract_named_entities(text)
    sentiment_kw = extract_sentiment_words(text)

    all_keywords = set(tfidf_kw + yake_kw + rake_kw + ner_kw + sentiment_kw)  # Remove duplicates
    return ", ".join(all_keywords)


# Create new keyword columns for each text column
for col in text_columns:
    df[col + "_Keywords"] = df.apply(lambda row: combine_keywords(row[col], df, col), axis=1)


# Save updated data back to the same Excel file
with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    df.to_excel(writer, sheet_name=sheet_name, index=False)

print("Updated Excel file with keyword columns saved successfully.")





