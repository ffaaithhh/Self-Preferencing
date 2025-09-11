import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the control experiment data
control_df = pd.read_csv('no explanation experiment/control_responses.csv')

# Clean responses by stripping whitespace and removing trailing full stops
control_df['Response'] = control_df['Response'].str.strip().str.rstrip('.')

# Count Product A vs. Product B wins (now accounting for cleaned responses)
product_a_wins = (control_df['Response'] == control_df['Product A']).sum()
product_b_wins = (control_df['Response'] == control_df['Product B']).sum()

print(f"Product A was chosen {product_a_wins} times.")
print(f"Product B was chosen {product_b_wins} times.")

# Analyze sentiment for each "Thinking" entry
control_df['Sentiment'] = control_df['Thinking'].fillna("").apply(
    lambda text: TextBlob(str(text)).sentiment.polarity
)

# Classify sentiment as Positive, Neutral, or Negative
control_df['Sentiment Label'] = control_df['Sentiment'].apply(
    lambda score: 'Positive' if score > 0 else ('Negative' if score < 0 else 'Neutral')
)

# Print sentiment distribution
print("\nSentiment Distribution:")
print(control_df['Sentiment Label'].value_counts())

def get_sentiment(text):
    blob = TextBlob(str(text))
    return pd.Series({'polarity': blob.sentiment.polarity, 'subjectivity': blob.sentiment.subjectivity})

sentiments = control_df['Thinking'].dropna().apply(get_sentiment)
print("Sentiment analysis summary:")
print(sentiments.describe())

# Save results to a new CSV
control_df.to_csv('control_responses_with_sentiment.csv', index=False)

# Combine all "Thinking" text into one string
all_text = ' '.join(control_df['Thinking'].fillna("").astype(str))


# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

# Plot
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Gemini\'s "Thinking" Process')
plt.show()

# Save the word cloud
wordcloud.to_file('thinking_wordcloud.png')