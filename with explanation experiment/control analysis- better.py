import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
import re
from collections import Counter

# Set up visualization style
plt.style.use('ggplot')
sns.set_palette("husl")

def load_data(file_path):
    """Load and clean the CSV data"""
    df = pd.read_csv(file_path)
    # Clean text data by removing empty strings and NaN values
    text_columns = ['Reasons for Product A', 'Reasons for Product B', 'Thinking']
    for col in text_columns:
        df[col] = df[col].fillna('').astype(str)
    return df

def analyze_product_preference(df):
    """Analyze which products are chosen more frequently based on order (first vs second)"""
    print("\n" + "="*50)
    print("PRODUCT PREFERENCE ANALYSIS (First vs Second Product)")
    print("="*50)
    
    # Determine if chosen product is the first one (Product A)
    df['First_Product_Chosen'] = df['Chosen Product'] == df['Product A']
    
    # Count preferences
    first_chosen = df['First_Product_Chosen'].sum()
    second_chosen = len(df) - first_chosen
    
    preference_counts = pd.Series({
        'First Product': first_chosen,
        'Second Product': second_chosen
    })
    
    total_comparisons = len(df)
    
    # Calculate percentages
    preference_percent = (preference_counts / total_comparisons) * 100
    
    print(f"\nTotal comparisons: {total_comparisons}")
    print("\nProduct preference counts (by order):")
    print(preference_counts)
    print("\nProduct preference percentages (by order):")
    print(preference_percent.round(1).astype(str) + "%")
    
    # Visualization
    plt.figure(figsize=(10, 5))
    
    # Bar chart
    plt.subplot(1, 2, 1)
    sns.barplot(x=preference_counts.index, y=preference_counts.values)
    plt.title('Product Preference Counts (by Order)')
    plt.ylabel('Number of Times Chosen')
    
    # Pie chart
    plt.subplot(1, 2, 2)
    plt.pie(preference_counts, labels=preference_counts.index, 
            autopct='%1.1f%%', startangle=90)
    plt.title('Product Preference Distribution (by Order)')
    
    plt.tight_layout()
    plt.show()
    
    return preference_counts

def clean_text(text):
    """Clean text for sentiment analysis and word clouds"""
    if pd.isna(text) or text == '':
        return ''
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', '', str(text).lower())
    # Remove common stop words
    stop_words = {'the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'that', 'for', 
                 'with', 'on', 'as', 'at', 'by', 'this', 'are', 'be', 'from'}
    words = [word for word in text.split() if word not in stop_words and len(word) > 2]
    return ' '.join(words)

def sentiment_analysis(df, text_columns):
    """Perform sentiment analysis on text columns"""
    print("\n" + "="*50)
    print("SENTIMENT ANALYSIS")
    print("="*50)
    
    sentiment_results = {}
    
    for col in text_columns:
        print(f"\nAnalyzing: {col}")
        sentiments = []
        subjectivity = []
        
        for text in df[col]:
            cleaned_text = clean_text(text)
            blob = TextBlob(cleaned_text)
            sentiments.append(blob.sentiment.polarity)
            subjectivity.append(blob.sentiment.subjectivity)
        
        # Store results
        sentiment_results[col] = {
            'polarity': np.mean(sentiments),
            'subjectivity': np.mean(subjectivity)
        }
        
        print(f"Average Sentiment Polarity (-1 to 1): {np.mean(sentiments):.3f}")
        print(f"Average Subjectivity (0 to 1): {np.mean(subjectivity):.3f}")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Sentiment polarity
    polarities = [v['polarity'] for v in sentiment_results.values()]
    ax1.bar(sentiment_results.keys(), polarities, color='skyblue')
    ax1.set_title('Average Sentiment Polarity')
    ax1.set_ylabel('Polarity (-1 to 1)')
    ax1.axhline(0, color='gray', linestyle='--')
    
    # Subjectivity
    subjectivities = [v['subjectivity'] for v in sentiment_results.values()]
    ax2.bar(sentiment_results.keys(), subjectivities, color='lightgreen')
    ax2.set_title('Average Subjectivity')
    ax2.set_ylabel('Subjectivity (0 to 1)')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return sentiment_results

def generate_word_clouds(df, text_columns):
    """Generate word clouds for text columns"""
    print("\n" + "="*50)
    print("WORD CLOUD ANALYSIS")
    print("="*50)
    
    plt.figure(figsize=(18, 6))
    
    for i, col in enumerate(text_columns, 1):
        # Combine all text for this column
        combined_text = ' '.join(df[col].astype(str))
        cleaned_text = clean_text(combined_text)
        
        if not cleaned_text:
            continue
            
        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400, 
                            background_color='white',
                            colormap='viridis',
                            max_words=100).generate(cleaned_text)
        
        # Plot
        plt.subplot(1, len(text_columns), i)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Word Cloud: {col}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print top words
    for col in text_columns:
        combined_text = ' '.join(df[col].astype(str))
        cleaned_text = clean_text(combined_text)
        word_counts = Counter(cleaned_text.split())
        
        print(f"\nTop 10 words in {col}:")
        for word, count in word_counts.most_common(10):
            print(f"{word}: {count}")

def analyze_thinking(df):
    """Special analysis for the Thinking column"""
    print("\n" + "="*50)
    print("THINKING PROCESS ANALYSIS")
    print("="*50)
    
    # Sentiment analysis
    sentiment_results = sentiment_analysis(df, ['Thinking'])
    
    # Word cloud
    generate_word_clouds(df, ['Thinking'])
    
    # Additional metrics
    thinking_lengths = df['Thinking'].apply(lambda x: len(str(x).split()))
    print(f"\nAverage length of thinking responses: {thinking_lengths.mean():.1f} words")
    print(f"Longest thinking response: {thinking_lengths.max()} words")
    print(f"Shortest thinking response: {thinking_lengths.min()} words")
    
    # Visualization
    plt.figure(figsize=(10, 5))
    sns.histplot(thinking_lengths, bins=20, kde=True)
    plt.title('Distribution of Thinking Response Lengths')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.show()

def run_full_analysis(file_path):
    """Run the complete analysis pipeline"""
    # Load data
    df = load_data(file_path)
    
    # 1. Product preference analysis
    preference_counts = analyze_product_preference(df)
    
    # 2. Reasons analysis
    print("\n" + "="*50)
    print("REASONS ANALYSIS (First vs Second Product)")
    print("="*50)
    
    # Sentiment analysis for reasons
    sentiment_analysis(df, ['Reasons for Product A', 'Reasons for Product B'])
    
    # Word clouds for reasons
    generate_word_clouds(df, ['Reasons for Product A', 'Reasons for Product B'])
    
    # 3. Thinking analysis
    analyze_thinking(df)

# Run the analysis
if __name__ == "__main__":
    input_file = "with explanation experiment/control_response(new).csv"  # Update with your file path
    run_full_analysis(input_file)