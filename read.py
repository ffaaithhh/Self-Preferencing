import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Load your CSV file
df = pd.read_csv("google_vs_competitors_analysis_results.csv")

# Ensure we have the category column (add if missing)
if 'Category' not in df.columns:
    # If your CSV doesn't have categories, merge with the reference you provided
    categories = pd.read_csv("google_vs_competitors.csv")  # Save your category table as a separate CSV
    df = pd.merge(df, categories[['Prompt', 'Category']], on='Prompt', how='left')

# 1. Category Analysis
def analyze_by_category(df):
    results = df.groupby('Category').agg({
        'google_net_sentiment': ['mean', 'count'],
        'competitor_net_sentiment': 'mean',
        'google_pro_count': 'sum',
        'google_con_count': 'sum',
        'competitor_pro_count': 'sum',
        'competitor_con_count': 'sum'
    })
    
    # Flatten multi-index columns
    results.columns = ['_'.join(col).strip() for col in results.columns.values]
    
    # Calculate differences and ratios
    results['sentiment_diff'] = results['google_net_sentiment_mean'] - results['competitor_net_sentiment_mean']
    results['google_arg_ratio'] = results['google_pro_count_sum'] / results['google_con_count_sum'].replace(0, 1)
    results['competitor_arg_ratio'] = results['competitor_pro_count_sum'] / results['competitor_con_count_sum'].replace(0, 1)
    
    return results.sort_values('sentiment_diff', ascending=False)

category_results = analyze_by_category(df)
print("\nCATEGORY ANALYSIS:")
print(category_results[['google_net_sentiment_mean', 'competitor_net_sentiment_mean', 'sentiment_diff', 
                        'google_net_sentiment_count', 'google_arg_ratio', 'competitor_arg_ratio']])

# 2. Visualization by Category
plt.figure(figsize=(14, 8))
category_results.reset_index(inplace=True)
plt.barh(category_results['Category'], category_results['sentiment_diff'], color='#4285F4')
plt.axvline(0, color='black', linestyle='--')
plt.title("Google vs Competitor Sentiment Difference by Category")
plt.xlabel("Sentiment Difference (Google - Competitor)")
plt.ylabel("Category")
plt.tight_layout()
plt.show()

# 3. Statistical Tests per Category
print("\nSTATISTICAL SIGNIFICANCE BY CATEGORY:")
for category in df['Category'].unique():
    subset = df[df['Category'] == category]
    if len(subset) > 1:  # Need at least 2 samples for t-test
        t_stat, p_value = stats.ttest_rel(subset['google_net_sentiment'], subset['competitor_net_sentiment'])
        print(f"{category:<20} p-value: {p_value:.4f} ({'Significant' if p_value < 0.05 else 'Not significant'})")

# 4. Argument Analysis by Category
def plot_argument_ratios(category_results):
    fig, ax = plt.subplots(figsize=(14, 8))
    width = 0.35
    x = range(len(category_results))
    
    ax.barh(x, category_results['google_arg_ratio'], width, label='Google', color='#4285F4')
    ax.barh([i + width for i in x], category_results['competitor_arg_ratio'], width, label='Competitor', color='#EA4335')
    
    ax.set_yticks([i + width/2 for i in x])
    ax.set_yticklabels(category_results['Category'])
    ax.set_title("Pro/Con Argument Ratios by Category")
    ax.set_xlabel("Pro Arguments per Con Argument")
    ax.legend()
    plt.tight_layout()
    plt.show()

plot_argument_ratios(category_results)

# 5. Detailed Category Drill-Down
def analyze_specific_category(category_name):
    subset = df[df['Category'] == category_name]
    print(f"\nDETAILED ANALYSIS FOR: {category_name}")
    
    # Top 3 most biased comparisons
    subset['sentiment_diff'] = subset['google_net_sentiment'] - subset['competitor_net_sentiment']
    top_biased = subset.nlargest(3, 'sentiment_diff')[['Prompt', 'google_net_sentiment', 'competitor_net_sentiment']]
    print("\nTop 3 Pro-Google Comparisons:")
    print(top_biased)
    
    # Top 3 competitor-favored comparisons
    top_competitor = subset.nsmallest(3, 'sentiment_diff')[['Prompt', 'google_net_sentiment', 'competitor_net_sentiment']]
    print("\nTop 3 Pro-Competitor Comparisons:")
    print(top_competitor)

# Example for a specific category
analyze_specific_category('Productivity')
analyze_specific_category('Browsers')