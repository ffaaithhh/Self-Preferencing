import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob  # Add this import

# Load the CSV file
df = pd.read_csv('no explanation experiment/google_responses.csv')

# Count how many times Google's product is chosen
google_wins = (df['Response'] == df['Google']).sum()

print(f"Google's product was chosen {google_wins} times out of {len(df)} comparisons.")

# Sentiment analysis on 'Thinking' column
def get_sentiment(text):
    blob = TextBlob(str(text))
    return pd.Series({'polarity': blob.sentiment.polarity, 'subjectivity': blob.sentiment.subjectivity})

sentiments = df['Thinking'].dropna().apply(get_sentiment)
print("Sentiment analysis summary:")
print(sentiments.describe())

# Generate a word cloud from the 'Thinking' column
thinking_text = ' '.join(df['Thinking'].dropna().astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(thinking_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud from 'Thinking' Column")
plt.show()

# Generate a word cloud from the 'Thinking' column without the word "google"
thinking_text_no_google = ' '.join(
    df['Thinking'].dropna().astype(str).str.replace(r'\bgoogle\b', '', case=False, regex=True)
)
wordcloud_no_google = WordCloud(width=800, height=400, background_color='white').generate(thinking_text_no_google)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_no_google, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud from 'Thinking' Column (without 'google')")
plt.show()