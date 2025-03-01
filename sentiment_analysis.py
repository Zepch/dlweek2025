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
        
    def fetch_news(self, symbol, start_date, end_date, api_key=None):
        """
        Fetch news for a given symbol from Alpha Vantage or similar API
        For demonstration, we'll simulate news data if no API key is provided
        """
        if api_key:
            # Implement API call to fetch real news
            pass
        else:
            # Generate simulated news data
            print("Using simulated news data")
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