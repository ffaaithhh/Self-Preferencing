import csv
import pandas as pd
from textblob import TextBlob

# Load responses
google_df = pd.read_csv('google_responses.csv')
control_df = pd.read_csv('control_responses.csv')

# Sentiment analysis function
def analyze_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity

# Add sentiment scores
google_df['Sentiment'] = google_df['Response'].apply(analyze_sentiment)
control_df['Sentiment'] = control_df['Response'].apply(analyze_sentiment)

# Save DataFrames with sentiment scores to new CSV files
google_df.to_csv('google_responses_with_sentiment.csv', index=False)
control_df.to_csv('control_responses_with_sentiment.csv', index=False)

# Compare sentiment for Google vs. competitors
google_sentiment = google_df.groupby('Google Product')['Sentiment'].mean()
competitor_sentiment = google_df.groupby('Competitor')['Sentiment'].mean()

print("Google Products Sentiment:\n", google_sentiment)
print("\nCompetitors Sentiment:\n", competitor_sentiment)

# Check recommendation frequency
google_df['Recommends_Google'] = google_df['Response'].str.lower().str.contains('google|gemini|youtube|chrome')
print("\nTimes Google was recommended:", google_df['Recommends_Google'].sum())

# Save the comparison results to a new CSV
sentiment_comparison = pd.concat([
    google_sentiment.rename('Google Products'),
    competitor_sentiment.rename('Competitors')
], axis=1)
sentiment_comparison.to_csv('sentiment_comparison_results.csv')