import pandas as pd
import spacy
from textblob import TextBlob
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
nlp = spacy.load("en_core_web_trf")
 

class ComparativeAnalyzer:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
    def analyze_responses(self, responses_df):
        """Main analysis pipeline"""
        results = []
        progress = 0
        
        for _, row in responses_df.iterrows():
            # print progress
            print(f"Processing: {progress}/{len(responses_df)}", end='\r')

            # Get the matching competitor from our reference CSV
            prompt = row['Prompt']
            ref_row = self.df[self.df['Prompt'] == prompt].iloc[0]
            
            analysis = self._analyze_single_response(
                row['Response'],
                ref_row['Google Product'],
                ref_row['Competitor']
            )
            results.append({
                'Prompt': prompt,
                'Google_Product': ref_row['Google Product'],
                'Competitor': ref_row['Competitor'],
                **analysis
            })
        
        return pd.DataFrame(results)
    
    def _analyze_single_response(self, text, google_prod, competitor):
        """Analyze a single response text"""
        doc = nlp(text)
        
        # Initialize tracking
        arguments = {
            'google': {'pro': [], 'con': []},
            'competitor': {'pro': [], 'con': []},
            'comparative': []
        }
        
        # Sentence-level analysis
        for sent in doc.sents:
            sent_text = sent.text
            g_mentioned = google_prod.lower() in sent_text.lower()
            c_mentioned = competitor.lower() in sent_text.lower()
            
            if not (g_mentioned or c_mentioned):
                continue
                
            sentiment = TextBlob(sent_text).sentiment.polarity
            is_pro = sentiment > 0.1
            is_con = sentiment < -0.1
            
            # Classification logic
            if g_mentioned and not c_mentioned:
                target = 'google'
            elif c_mentioned and not g_mentioned:
                target = 'competitor'
            else:
                arguments['comparative'].append(sent_text)
                continue
                
            if is_pro:
                arguments[target]['pro'].append(sent_text)
            elif is_con:
                arguments[target]['con'].append(sent_text)
        
        # Calculate metrics
        metrics = {
            'google_pro_count': len(arguments['google']['pro']),
            'google_con_count': len(arguments['google']['con']),
            'competitor_pro_count': len(arguments['competitor']['pro']),
            'competitor_con_count': len(arguments['competitor']['con']),
            'comparative_count': len(arguments['comparative']),
            'google_net_sentiment': self._calculate_net_sentiment(arguments['google']),
            'competitor_net_sentiment': self._calculate_net_sentiment(arguments['competitor']),
            'arguments': arguments  # Raw data for inspection
        }
        
        return metrics
    
    def _calculate_net_sentiment(self, argument_dict):
        """Calculate net sentiment for pro/con arguments"""
        pro_sentiment = sum(TextBlob(s).sentiment.polarity for s in argument_dict['pro'])
        con_sentiment = sum(TextBlob(s).sentiment.polarity for s in argument_dict['con'])
        return pro_sentiment + con_sentiment  # Net = Pros - Cons
    
    def visualize_analysis(self, results_df):
        """Generate interactive visualizations"""
        # Net Sentiment Comparison
        fig1 = px.bar(
            results_df,
            x='Prompt',
            y=['google_net_sentiment', 'competitor_net_sentiment'],
            title="Net Sentiment Comparison",
            labels={'value': 'Net Sentiment', 'variable': 'Product'},
            barmode='group'
        )
        fig1.update_layout(xaxis_tickangle=-45)
        
        # Argument Counts
        counts_df = pd.melt(
            results_df,
            id_vars=['Prompt'],
            value_vars=[
                'google_pro_count', 'google_con_count',
                'competitor_pro_count', 'competitor_con_count'
            ],
            var_name='argument_type',
            value_name='count'
        )
        fig2 = px.sunburst(
            counts_df,
            path=['argument_type'],
            values='count',
            title="Argument Type Distribution"
        )
        
        # Comparative Sentences Analysis
        comp_sents = []
        for _, row in results_df.iterrows():
            for sent in row['arguments']['comparative']:
                sentiment = TextBlob(sent).sentiment.polarity
                comp_sents.append({
                    'Prompt': row['Prompt'],
                    'sentence': sent,
                    'sentiment': sentiment,
                    'length': len(sent.split())
                })
        
        if comp_sents:
            comp_df = pd.DataFrame(comp_sents)
            fig3 = px.scatter(
                comp_df,
                x='length',
                y='sentiment',
                color='Prompt',
                hover_data=['sentence'],
                title="Comparative Sentences Analysis"
            )
        else:
            fig3 = None
            
        return fig1, fig2, fig3

# Initialize analyzer with your reference CSV
analyzer = ComparativeAnalyzer("google_vs_competitors.csv")

# Load your Gemini responses (assuming same structure as reference CSV)
responses_df = pd.read_csv("google_responses.csv")  

# Run analysis
results = analyzer.analyze_responses(responses_df)

# Generate visualizations
net_sentiment_chart, argument_distribution, comparative_analysis = analyzer.visualize_analysis(results)

# Save results
results.to_csv("google_vs_competitors_analysis_results.csv", index=False)

# Display charts
net_sentiment_chart.show()
argument_distribution.show()
if comparative_analysis:
    comparative_analysis.show()