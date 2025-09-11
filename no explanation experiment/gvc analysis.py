import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
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

class GeminiBiasAnalyzer:
    def __init__(self, csv_file_path):
            """Initialize the analyzer with CSV data"""
            self.data = pd.read_csv(csv_file_path)
            self.setup_styling()
    
    def setup_styling(self):
        """Set up matplotlib styling"""
        plt.style.use('default')
        sns.set_palette("husl")

    def analyze_by_category(self):
            """Analyze preferences by product category"""
            if 'Category' not in self.data.columns:
                print("No 'Category' column found in data")
                return
            
            print("\n" + "="*50)
            print("CATEGORY-SPECIFIC ANALYSIS")
            print("="*50)
            
            results = []
            for category, group in self.data.groupby('Category'):
                total = len(group)
                google_chosen = sum(1 for _, row in group.iterrows() 
                                if row['Response'] == row['Google'])
                perc = google_chosen / total * 100
                results.append((category, google_chosen, total, perc))
            
            # Sort by Google preference rate
            results.sort(key=lambda x: x[3], reverse=True)
            
            # Print results
            for category, google, total, perc in results:
                print(f"{category}: Google chosen {google}/{total} times ({perc:.1f}%)")
            
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            categories = [x[0] for x in results]
            percentages = [x[3] for x in results]
            
            colors = ['#4285F4' if p >= 50 else '#FF6B6B' for p in percentages]
            bars = ax.barh(categories, percentages, color=colors)
            
            ax.set_xlabel('Google Preference Rate (%)')
            ax.set_title('Google Preference by Category', fontweight='bold')
            ax.bar_label(bars, fmt='%.1f%%')
            
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    # Initialize analyzer with your CSV file
    analyzer = GeminiBiasAnalyzer('no explanation experiment/google_responses.csv')
    
    # Run the complete analysis
    analyzer.analyze_by_category()