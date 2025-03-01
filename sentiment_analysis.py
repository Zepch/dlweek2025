# sentiment_analysis.py
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

class SentimentAnalyzer:
    def __init__(self):
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        self.sid = SentimentIntensityAnalyzer()
        
    def fetch_news(self, symbol, start_date, end_date, api_key='MN8ZH85QBP7XSDVO'):
        """
        Fetch news for a given symbol from Alpha Vantage API
        """
        if api_key:
            # Implement API call to fetch real news
            try:
                # Convert dates to required format if they're strings
                if isinstance(start_date, str):
                    start_date = pd.to_datetime(start_date).strftime('%Y%m%dT0000')
                else:
                    start_date = start_date.strftime('%Y%m%dT0000')
                
                if isinstance(end_date, str):
                    end_date = pd.to_datetime(end_date).strftime('%Y%m%dT0000')
                else:
                    end_date = end_date.strftime('%Y%m%dT0000')
                
                # Build the API URL
                base_url = 'https://www.alphavantage.co/query'
                params = {
                    'function': 'NEWS_SENTIMENT',
                    'tickers': symbol,
                    'apikey': api_key,
                    'time_from': start_date,
                    'time_to': end_date,
                    'limit': 1000  # Get maximum available news items
                }
                
                print(f"Fetching news data for {symbol}...")
                response = requests.get(base_url, params=params)
                response.raise_for_status()  # Raise error for bad status codes
                
                data = response.json()
                
                if 'feed' not in data:
                    print(f"No news data available or API error: {data}")
                    return self.generate_simulated_news(symbol, start_date, end_date)
                
                # Process the news feed
                news_items = []
                for item in data['feed']:
                    # Extract relevant fields
                    news_items.append({
                        'date': pd.to_datetime(item.get('time_published', '')),
                        'headline': item.get('title', ''),
                        'summary': item.get('summary', ''),
                        'source': item.get('source', ''),
                        'url': item.get('url', ''),
                        'overall_sentiment_score': item.get('overall_sentiment_score', 0),
                        'overall_sentiment_label': item.get('overall_sentiment_label', '')
                    })
                
                # Create DataFrame
                news_df = pd.DataFrame(news_items)
                print(f"Retrieved {len(news_df)} news items for {symbol}")
                return news_df
                
            except Exception as e:
                print(f"Error fetching news data: {e}")
                print("Falling back to simulated news data")
                return self.generate_simulated_news(symbol, start_date, end_date)
        else:
            return self.generate_simulated_news(symbol, start_date, end_date)
        
    def generate_simulated_news(self, symbol, start_date, end_date):
        """Helper method to generate simulated news data"""
        print("Using simulated news data")
        
        # Convert string dates to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        date_range = pd.date_range(start=start_date, end=end_date)
        
        # Create random sentiment data
        np.random.seed(42)  # For reproducibility
        sentiment_data = pd.DataFrame({
            'date': date_range,
            'headline': [f"News about {symbol} on {d.strftime('%Y-%m-%d')}" for d in date_range],
            'source': ['Simulated'] * len(date_range),
        })
        
        return sentiment_data
    
    def analyze_sentiment(self, news_df):
        """
        Calculate sentiment scores for news headlines
        """
        sentiment_scores = []
        
        for headline in news_df['headline']:
            score = self.sid.polarity_scores(headline)
            sentiment_scores.append(score)
        
        # Extract compound score as the main sentiment indicator
        news_df['sentiment'] = [score['compound'] for score in sentiment_scores]
        
        # Aggregate by date
        daily_sentiment = news_df.groupby(news_df['date'].dt.date).agg({
            'sentiment': ['mean', 'std', 'count']
        })
        
        # Flatten multi-level columns
        daily_sentiment.columns = ['sentiment_mean', 'sentiment_std', 'sentiment_count']
        daily_sentiment = daily_sentiment.reset_index()
        
        return daily_sentiment