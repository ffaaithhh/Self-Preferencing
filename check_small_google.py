import pandas as pd
import spacy
from textblob import TextBlob
from multiprocessing import Pool, cpu_count
import re
from collections import defaultdict

class OptimizedComparativeAnalyzer:
    def __init__(self, reference_csv, 
                 prompt_col='Prompt',
                 google_product_col='Product1',
                 competitor_col='Product2'):
        """
        Initialize the analyzer with configurable column names
        
        Args:
            reference_csv: Path to reference CSV file
            prompt_col: Name of prompt column
            google_product_col: Name of Google product column
            competitor_col: Name of competitor column
        """
        self.reference_df = pd.read_csv(reference_csv)
        self.prompt_col = prompt_col
        self.google_product_col = google_product_col
        self.competitor_col = competitor_col
        
        # Verify required columns exist
        self._verify_columns(self.reference_df, "Reference CSV")
        
        self.sentiment_cache = {}
        # Load the large model once
        self.nlp = spacy.load("en_core_web_trf", disable=["ner", "parser"])
        
    def _verify_columns(self, df, df_name):
        """Verify required columns exist in dataframe"""
        required_columns = {
            self.prompt_col,
            self.google_product_col,
            self.competitor_col
        }
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"{df_name} is missing required columns: {missing}\n"
                f"Available columns: {df.columns.tolist()}"
            )
    
    def analyze_responses(self, responses_csv, response_col='Response'):
        """
        Analyze responses from CSV file
        
        Args:
            responses_csv: Path to responses CSV file
            response_col: Name of response text column
        """
        responses_df = pd.read_csv(responses_csv)
        
        # Verify response CSV has required columns
        if self.prompt_col not in responses_df.columns:
            raise ValueError(
                f"Responses CSV must contain '{self.prompt_col}' column\n"
                f"Available columns: {responses_df.columns.tolist()}"
            )
        if response_col not in responses_df.columns:
            raise ValueError(
                f"Responses CSV must contain '{response_col}' column\n"
                f"Available columns: {responses_df.columns.tolist()}"
            )
        
        merged_df = pd.merge(
            responses_df,
            self.reference_df,
            on=self.prompt_col,
            how="left"
        )
        
        # Prepare data for parallel processing
        analysis_data = []
        for _, row in merged_df.iterrows():
            analysis_data.append({
                'text': row[response_col],
                'google_product': row[self.google_product_col],
                'competitor': row[self.competitor_col],
                'prompt': row[self.prompt_col]
            })
        
        # Process in parallel
        with Pool(cpu_count()) as pool:
            results = pool.map(self._analyze_single, analysis_data)
        
        return pd.DataFrame(results)
    
    # [Rest of your methods remain the same...]
    def _analyze_single(self, data):
        """Analyze a single response with optimized sentiment analysis"""
        doc = self.nlp(data['text'])
        google_lower = data['google_product'].lower()
        competitor_lower = data['competitor'].lower()
        
        arguments = {
            'google': {'pro': [], 'con': []},
            'competitor': {'pro': [], 'con': []},
            'comparative': []
        }
        
        # Extract all sentences first
        sentences = [sent.text for sent in doc.sents]
        
        # Batch process sentiment analysis
        sentiments = self._batch_sentiment_analysis(sentences)
        
        for sent_text, sentiment in zip(sentences, sentiments):
            g_mentioned = google_lower in sent_text.lower()
            c_mentioned = competitor_lower in sent_text.lower()
            
            if not (g_mentioned or c_mentioned):
                continue
                
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
            'Prompt': data['prompt'],
            'Google_Product': data['google_product'],
            'Competitor': data['competitor'],
            'google_pro_count': len(arguments['google']['pro']),
            'google_con_count': len(arguments['google']['con']),
            'competitor_pro_count': len(arguments['competitor']['pro']),
            'competitor_con_count': len(arguments['competitor']['con']),
            'comparative_count': len(arguments['comparative']),
            'google_net_sentiment': self._calculate_net_sentiment(arguments['google']),
            'competitor_net_sentiment': self._calculate_net_sentiment(arguments['competitor']),
            'arguments': arguments
        }
        
        return metrics
    
    def _batch_sentiment_analysis(self, sentences):
        """Process sentiment analysis in batches with caching"""
        results = []
        uncached = []
        
        # Check cache first
        for sent in sentences:
            clean_sent = self._clean_text(sent)
            if clean_sent in self.sentiment_cache:
                results.append(self.sentiment_cache[clean_sent])
            else:
                uncached.append((len(results), clean_sent))
                results.append(None)
        
        # Process uncached sentences
        if uncached:
            sentiments = []
            for _, sent in uncached:
                sentiments.append(TextBlob(sent).sentiment.polarity)
            
            # Update cache and results
            for (idx, sent), sentiment in zip(uncached, sentiments):
                self.sentiment_cache[sent] = sentiment
                results[idx] = sentiment
        
        return results
    
    def _clean_text(self, text):
        """Basic text cleaning for cache keys"""
        return re.sub(r'\s+', ' ', text).strip().lower()
    
    def _calculate_net_sentiment(self, argument_dict):
        """Calculate net sentiment for pro/con arguments"""
        pro_sentiment = sum(TextBlob(s).sentiment.polarity for s in argument_dict['pro'])
        con_sentiment = sum(TextBlob(s).sentiment.polarity for s in argument_dict['con'])
        return pro_sentiment + con_sentiment

    def print_categorized_sentences(self, results_df):
        """Print categorized sentences from analysis results"""
        for _, row in results_df.iterrows():
            print(f"\n\n=== Analysis for Prompt: '{row['Prompt']}' ===")
            print(f"Google Product: {row['Google_Product']}")
            print(f"Competitor: {row['Competitor']}\n")
            
            arguments = row['arguments']
            
            def print_section(title, sentences):
                print(f"\n=== {title} ===")
                if not sentences:
                    print("(None found)")
                    return
                for i, sent in enumerate(sentences, 1):
                    sentiment = TextBlob(sent).sentiment.polarity
                    print(f"{i}. [Sentiment: {sentiment:.2f}] {sent}")
            
            print_section("Google Pro Arguments", arguments['google']['pro'])
            print_section("Google Con Arguments", arguments['google']['con'])
            print_section("Competitor Pro Arguments", arguments['competitor']['pro'])
            print_section("Competitor Con Arguments", arguments['competitor']['con'])
            print_section("Comparative Sentences", arguments['comparative'])

if __name__ == "__main__":
    # Initialize with your actual column names
    analyzer = OptimizedComparativeAnalyzer(
        "small_google_prompt.csv",
        prompt_col='Prompt',          # Change if your column has different name
        google_product_col='Google',  # Change to your Google product column name
        competitor_col='Competitor'   # Change to your competitor column name
    )
    
    # First print the columns to verify
    print("Reference CSV columns:", analyzer.reference_df.columns.tolist())
    
    # Then analyze
    results = analyzer.analyze_responses(
        "small_google_responses.csv",
        response_col='Response'  # Change if your response column has different name
    )
    
    analyzer.print_categorized_sentences(results)
    results.to_csv("small_test.csv", index=False)