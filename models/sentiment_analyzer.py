#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sentiment Analysis module for ATFNet.
Implements methods to analyze sentiment from news and social media data.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
import joblib
import re
import string
import requests
from collections import defaultdict
import time
import json

logger = logging.getLogger('atfnet.models.sentiment')


class SentimentAnalyzer:
    """
    Sentiment Analysis for ATFNet.
    Implements methods to analyze sentiment from news and social media data.
    """
    
    def __init__(self, config: Dict[str, Any] = None, output_dir: str = 'sentiment_analysis'):
        """
        Initialize the sentiment analyzer.
        
        Args:
            config: Configuration dictionary
            output_dir: Directory to save outputs
        """
        self.config = config or {}
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load default parameters
        self.api_keys = self.config.get('api_keys', {})
        self.default_model = self.config.get('default_model', 'vader')
        self.cache_dir = os.path.join(output_dir, 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize state variables
        self.sentiment_cache = {}
        self.news_cache = {}
        self.social_cache = {}
        self.sentiment_history = []
        
        # Initialize sentiment models
        self._initialize_models()
        
        logger.info(f"Sentiment Analyzer initialized with {self.default_model} model")
    
    def _initialize_models(self) -> None:
        """
        Initialize sentiment analysis models.
        """
        self.models = {}
        
        # Initialize VADER
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            self.models['vader'] = SentimentIntensityAnalyzer()
            logger.info("VADER sentiment model initialized")
        except ImportError:
            logger.warning("NLTK VADER not available. Install with: pip install nltk")
            logger.warning("Then run: import nltk; nltk.download('vader_lexicon')")
        
        # Initialize TextBlob
        try:
            from textblob import TextBlob
            self.models['textblob'] = TextBlob
            logger.info("TextBlob sentiment model initialized")
        except ImportError:
            logger.warning("TextBlob not available. Install with: pip install textblob")
        
        # Initialize Transformers (if available)
        try:
            from transformers import pipeline
            self.models['transformers'] = pipeline('sentiment-analysis')
            logger.info("Transformers sentiment model initialized")
        except ImportError:
            logger.warning("Transformers not available. Install with: pip install transformers")
        
        # Check if any models are available
        if not self.models:
            logger.error("No sentiment models available. Install at least one of: nltk, textblob, transformers")
    
    def analyze_text(self, text: str, model: str = None) -> Dict[str, float]:
        """
        Analyze sentiment of text.
        
        Args:
            text: Text to analyze
            model: Model to use (default: from config)
            
        Returns:
            Dictionary with sentiment scores
        """
        if not text:
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
        
        # Set default model
        if model is None:
            model = self.default_model
        
        # Check if model is available
        if model not in self.models:
            available_models = list(self.models.keys())
            if not available_models:
                logger.error("No sentiment models available")
                return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
            
            logger.warning(f"Model {model} not available. Using {available_models[0]} instead")
            model = available_models[0]
        
        # Check cache
        cache_key = f"{model}_{hash(text)}"
        if cache_key in self.sentiment_cache:
            return self.sentiment_cache[cache_key]
        
        # Analyze sentiment based on model
        if model == 'vader':
            sentiment = self._analyze_vader(text)
        elif model == 'textblob':
            sentiment = self._analyze_textblob(text)
        elif model == 'transformers':
            sentiment = self._analyze_transformers(text)
        else:
            logger.error(f"Unknown model: {model}")
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
        
        # Cache result
        self.sentiment_cache[cache_key] = sentiment
        
        return sentiment
    
    def _analyze_vader(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using VADER.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        try:
            scores = self.models['vader'].polarity_scores(text)
            return scores
        except Exception as e:
            logger.error(f"Error analyzing with VADER: {e}")
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
    
    def _analyze_textblob(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using TextBlob.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        try:
            blob = self.models['textblob'](text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Convert to VADER-like format for consistency
            if polarity > 0:
                positive = polarity
                negative = 0.0
                neutral = 1.0 - positive
            elif polarity < 0:
                positive = 0.0
                negative = -polarity
                neutral = 1.0 - negative
            else:
                positive = 0.0
                negative = 0.0
                neutral = 1.0
            
            return {
                'compound': polarity,
                'positive': positive,
                'negative': negative,
                'neutral': neutral,
                'subjectivity': subjectivity
            }
        except Exception as e:
            logger.error(f"Error analyzing with TextBlob: {e}")
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
    
    def _analyze_transformers(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using Transformers.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        try:
            # Truncate text if too long
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
            
            result = self.models['transformers'](text)[0]
            label = result['label']
            score = result['score']
            
            # Convert to VADER-like format for consistency
            if 'POSITIVE' in label:
                positive = score
                negative = 1.0 - score
                neutral = 0.0
                compound = score
            elif 'NEGATIVE' in label:
                positive = 1.0 - score
                negative = score
                neutral = 0.0
                compound = -score
            else:
                positive = 0.0
                negative = 0.0
                neutral = score
                compound = 0.0
            
            return {
                'compound': compound,
                'positive': positive,
                'negative': negative,
                'neutral': neutral,
                'label': label,
                'score': score
            }
        except Exception as e:
            logger.error(f"Error analyzing with Transformers: {e}")
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
    
    def fetch_news(self, symbol: str, days: int = 7, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch news articles for a symbol.
        
        Args:
            symbol: Symbol to fetch news for
            days: Number of days to look back
            limit: Maximum number of articles to fetch
            
        Returns:
            List of news articles
        """
        # Check cache
        cache_key = f"{symbol}_{days}_{limit}"
        if cache_key in self.news_cache:
            # Check if cache is still valid (1 hour)
            cache_time = self.news_cache[cache_key]['timestamp']
            if datetime.now() - cache_time < timedelta(hours=1):
                return self.news_cache[cache_key]['articles']
        
        # Try different news APIs
        articles = []
        
        # Try Alpha Vantage
        if 'alpha_vantage' in self.api_keys:
            articles = self._fetch_alpha_vantage_news(symbol, days, limit)
        
        # Try News API if no articles found
        if not articles and 'news_api' in self.api_keys:
            articles = self._fetch_news_api(symbol, days, limit)
        
        # Try Financial Modeling Prep if no articles found
        if not articles and 'fmp' in self.api_keys:
            articles = self._fetch_fmp_news(symbol, days, limit)
        
        # Cache result
        self.news_cache[cache_key] = {
            'articles': articles,
            'timestamp': datetime.now()
        }
        
        logger.info(f"Fetched {len(articles)} news articles for {symbol}")
        
        return articles
    
    def _fetch_alpha_vantage_news(self, symbol: str, days: int, limit: int) -> List[Dict[str, Any]]:
        """
        Fetch news from Alpha Vantage.
        
        Args:
            symbol: Symbol to fetch news for
            days: Number of days to look back
            limit: Maximum number of articles to fetch
            
        Returns:
            List of news articles
        """
        try:
            api_key = self.api_keys['alpha_vantage']
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={api_key}&limit={limit}"
            
            response = requests.get(url)
            data = response.json()
            
            if 'feed' not in data:
                logger.warning(f"No news data from Alpha Vantage for {symbol}")
                return []
            
            # Process articles
            articles = []
            
            # Filter articles by date
            cutoff_date = datetime.now() - timedelta(days=days)
            
            for article in data['feed']:
                # Parse date
                time_published = article.get('time_published', '')
                if time_published:
                    try:
                        # Format: YYYYMMDDTHHMMSS
                        date = datetime.strptime(time_published, '%Y%m%dT%H%M%S')
                        if date < cutoff_date:
                            continue
                    except ValueError:
                        # If date parsing fails, include the article anyway
                        pass
                
                # Extract sentiment
                sentiment = 0.0
                if 'overall_sentiment_score' in article:
                    sentiment = float(article['overall_sentiment_score'])
                
                # Create article object
                articles.append({
                    'title': article.get('title', ''),
                    'url': article.get('url', ''),
                    'published_at': time_published,
                    'source': article.get('source', ''),
                    'summary': article.get('summary', ''),
                    'sentiment': sentiment
                })
                
                # Limit number of articles
                if len(articles) >= limit:
                    break
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching news from Alpha Vantage: {e}")
            return []
    
    def _fetch_news_api(self, symbol: str, days: int, limit: int) -> List[Dict[str, Any]]:
        """
        Fetch news from News API.
        
        Args:
            symbol: Symbol to fetch news for
            days: Number of days to look back
            limit: Maximum number of articles to fetch
            
        Returns:
            List of news articles
        """
        try:
            api_key = self.api_keys['news_api']
            
            # Calculate date range
            to_date = datetime.now().strftime('%Y-%m-%d')
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            # Create query
            query = f"{symbol} stock OR {symbol} market OR {symbol} finance"
            
            url = f"https://newsapi.org/v2/everything?q={query}&from={from_date}&to={to_date}&sortBy=popularity&apiKey={api_key}&pageSize={limit}"
            
            response = requests.get(url)
            data = response.json()
            
            if data.get('status') != 'ok' or 'articles' not in data:
                logger.warning(f"No news data from News API for {symbol}")
                return []
            
            # Process articles
            articles = []
            
            for article in data['articles']:
                # Create article object
                articles.append({
                    'title': article.get('title', ''),
                    'url': article.get('url', ''),
                    'published_at': article.get('publishedAt', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'summary': article.get('description', ''),
                    'sentiment': 0.0  # Will be analyzed later
                })
                
                # Limit number of articles
                if len(articles) >= limit:
                    break
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching news from News API: {e}")
            return []
    
    def _fetch_fmp_news(self, symbol: str, days: int, limit: int) -> List[Dict[str, Any]]:
        """
        Fetch news from Financial Modeling Prep.
        
        Args:
            symbol: Symbol to fetch news for
            days: Number of days to look back
            limit: Maximum number of articles to fetch
            
        Returns:
            List of news articles
        """
        try:
            api_key = self.api_keys['fmp']
            url = f"https://financialmodelingprep.com/api/v3/stock_news?tickers={symbol}&limit={limit}&apikey={api_key}"
            
            response = requests.get(url)
            data = response.json()
            
            if not data:
                logger.warning(f"No news data from Financial Modeling Prep for {symbol}")
                return []
            
            # Process articles
            articles = []
            
            # Filter articles by date
            cutoff_date = datetime.now() - timedelta(days=days)
            
            for article in data:
                # Parse date
                published_at = article.get('publishedDate', '')
                if published_at:
                    try:
                        date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                        if date < cutoff_date:
                            continue
                    except ValueError:
                        # If date parsing fails, include the article anyway
                        pass
                
                # Create article object
                articles.append({
                    'title': article.get('title', ''),
                    'url': article.get('url', ''),
                    'published_at': published_at,
                    'source': article.get('site', ''),
                    'summary': article.get('text', ''),
                    'sentiment': 0.0  # Will be analyzed later
                })
                
                # Limit number of articles
                if len(articles) >= limit:
                    break
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching news from Financial Modeling Prep: {e}")
            return []
    
    def fetch_social_media(self, symbol: str, days: int = 7, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch social media posts for a symbol.
        
        Args:
            symbol: Symbol to fetch posts for
            days: Number of days to look back
            limit: Maximum number of posts to fetch
            
        Returns:
            List of social media posts
        """
        # Check cache
        cache_key = f"{symbol}_{days}_{limit}"
        if cache_key in self.social_cache:
            # Check if cache is still valid (1 hour)
            cache_time = self.social_cache[cache_key]['timestamp']
            if datetime.now() - cache_time < timedelta(hours=1):
                return self.social_cache[cache_key]['posts']
        
        # Try different social media APIs
        posts = []
        
        # Try Twitter (now X) API
        if 'twitter' in self.api_keys:
            posts = self._fetch_twitter_posts(symbol, days, limit)
        
        # Try Reddit API if no posts found
        if not posts and 'reddit' in self.api_keys:
            posts = self._fetch_reddit_posts(symbol, days, limit)
        
        # Try StockTwits API if no posts found
        if not posts and 'stocktwits' in self.api_keys:
            posts = self._fetch_stocktwits_posts(symbol, days, limit)
        
        # Cache result
        self.social_cache[cache_key] = {
            'posts': posts,
            'timestamp': datetime.now()
        }
        
        logger.info(f"Fetched {len(posts)} social media posts for {symbol}")
        
        return posts
    
    def _fetch_twitter_posts(self, symbol: str, days: int, limit: int) -> List[Dict[str, Any]]:
        """
        Fetch posts from Twitter (X).
        
        Args:
            symbol: Symbol to fetch posts for
            days: Number of days to look back
            limit: Maximum number of posts to fetch
            
        Returns:
            List of posts
        """
        try:
            api_key = self.api_keys['twitter']
            
            # Calculate date range
            end_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
            start_time = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%dT%H:%M:%SZ')
            
            # Create query
            query = f"${symbol} OR #{symbol}stock OR #{symbol} -is:retweet"
            
            url = f"https://api.twitter.com/2/tweets/search/recent?query={query}&start_time={start_time}&end_time={end_time}&max_results={min(limit, 100)}&tweet.fields=created_at,public_metrics"
            
            headers = {
                'Authorization': f"Bearer {api_key}"
            }
            
            response = requests.get(url, headers=headers)
            data = response.json()
            
            if 'data' not in data:
                logger.warning(f"No Twitter data for {symbol}")
                return []
            
            # Process posts
            posts = []
            
            for tweet in data['data']:
                # Create post object
                posts.append({
                    'text': tweet.get('text', ''),
                    'created_at': tweet.get('created_at', ''),
                    'source': 'Twitter',
                    'likes': tweet.get('public_metrics', {}).get('like_count', 0),
                    'retweets': tweet.get('public_metrics', {}).get('retweet_count', 0),
                    'sentiment': 0.0  # Will be analyzed later
                })
                
                # Limit number of posts
                if len(posts) >= limit:
                    break
            
            return posts
            
        except Exception as e:
            logger.error(f"Error fetching posts from Twitter: {e}")
            return []
    
    def _fetch_reddit_posts(self, symbol: str, days: int, limit: int) -> List[Dict[str, Any]]:
        """
        Fetch posts from Reddit.
        
        Args:
            symbol: Symbol to fetch posts for
            days: Number of days to look back
            limit: Maximum number of posts to fetch
            
        Returns:
            List of posts
        """
        try:
            api_key = self.api_keys['reddit']
            client_id = self.api_keys.get('reddit_client_id', '')
            client_secret = self.api_keys.get('reddit_client_secret', '')
            
            if not client_id or not client_secret:
                logger.warning("Reddit client ID or secret not provided")
                return []
            
            # Get OAuth token
            auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
            data = {
                'grant_type': 'client_credentials',
                'username': self.api_keys.get('reddit_username', ''),
                'password': self.api_keys.get('reddit_password', '')
            }
            headers = {'User-Agent': 'ATFNet/0.1'}
            
            response = requests.post('https://www.reddit.com/api/v1/access_token', auth=auth, data=data, headers=headers)
            token = response.json().get('access_token', '')
            
            if not token:
                logger.warning("Failed to get Reddit token")
                return []
            
            # Search for posts
            headers = {
                'Authorization': f"bearer {token}",
                'User-Agent': 'ATFNet/0.1'
            }
            
            # Search in relevant subreddits
            subreddits = ['wallstreetbets', 'stocks', 'investing', 'stockmarket']
            posts = []
            
            for subreddit in subreddits:
                url = f"https://oauth.reddit.com/r/{subreddit}/search?q={symbol}&restrict_sr=on&sort=new&t=week&limit={min(limit, 100)}"
                
                response = requests.get(url, headers=headers)
                data = response.json()
                
                if 'data' not in data or 'children' not in data['data']:
                    continue
                
                # Process posts
                for post in data['data']['children']:
                    post_data = post['data']
                    
                    # Check if post is within time range
                    created_utc = post_data.get('created_utc', 0)
                    post_date = datetime.fromtimestamp(created_utc)
                    
                    if (datetime.now() - post_date).days > days:
                        continue
                    
                    # Create post object
                    posts.append({
                        'title': post_data.get('title', ''),
                        'text': post_data.get('selftext', ''),
                        'created_at': post_date.isoformat(),
                        'source': f"Reddit/r/{subreddit}",
                        'likes': post_data.get('score', 0),
                        'comments': post_data.get('num_comments', 0),
                        'sentiment': 0.0  # Will be analyzed later
                    })
                    
                    # Limit number of posts
                    if len(posts) >= limit:
                        break
                
                if len(posts) >= limit:
                    break
            
            return posts
            
        except Exception as e:
            logger.error(f"Error fetching posts from Reddit: {e}")
            return []
    
    def _fetch_stocktwits_posts(self, symbol: str, days: int, limit: int) -> List[Dict[str, Any]]:
        """
        Fetch posts from StockTwits.
        
        Args:
            symbol: Symbol to fetch posts for
            days: Number of days to look back
            limit: Maximum number of posts to fetch
            
        Returns:
            List of posts
        """
        try:
            api_key = self.api_keys['stocktwits']
            url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json?access_token={api_key}&limit={min(limit, 30)}"
            
            response = requests.get(url)
            data = response.json()
            
            if 'messages' not in data:
                logger.warning(f"No StockTwits data for {symbol}")
                return []
            
            # Process posts
            posts = []
            
            # Filter posts by date
            cutoff_date = datetime.now() - timedelta(days=days)
            
            for message in data['messages']:
                # Parse date
                created_at = message.get('created_at', '')
                if created_at:
                    try:
                        # Format: 2023-01-01T12:00:00Z
                        date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        if date < cutoff_date:
                            continue
                    except ValueError:
                        # If date parsing fails, include the post anyway
                        pass
                
                # Extract sentiment
                sentiment = 0.0
                entities = message.get('entities', {})
                if 'sentiment' in entities:
                    sentiment_data = entities['sentiment']
                    if sentiment_data.get('basic') == 'Bullish':
                        sentiment = 1.0
                    elif sentiment_data.get('basic') == 'Bearish':
                        sentiment = -1.0
                
                # Create post object
                posts.append({
                    'text': message.get('body', ''),
                    'created_at': created_at,
                    'source': 'StockTwits',
                    'likes': message.get('likes', {}).get('total', 0),
                    'sentiment': sentiment
                })
                
                # Limit number of posts
                if len(posts) >= limit:
                    break
            
            return posts
            
        except Exception as e:
            logger.error(f"Error fetching posts from StockTwits: {e}")
            return []
    
    def analyze_news_sentiment(self, articles: List[Dict[str, Any]], model: str = None) -> Dict[str, Any]:
        """
        Analyze sentiment of news articles.
        
        Args:
            articles: List of news articles
            model: Model to use (default: from config)
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not articles:
            return {
                'compound': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 0.0,
                'count': 0
            }
        
        # Set default model
        if model is None:
            model = self.default_model
        
        # Analyze sentiment of each article
        for article in articles:
            # Skip if already analyzed
            if article.get('sentiment', 0.0) != 0.0:
                continue
            
            # Combine title and summary for analysis
            text = f"{article.get('title', '')} {article.get('summary', '')}"
            
            # Analyze sentiment
            sentiment = self.analyze_text(text, model)
            
            # Update article with sentiment
            article['sentiment'] = sentiment.get('compound', 0.0)
        
        # Calculate aggregate sentiment
        compound_sum = sum(article.get('sentiment', 0.0) for article in articles)
        compound_avg = compound_sum / len(articles)
        
        positive_count = sum(1 for article in articles if article.get('sentiment', 0.0) > 0.05)
        negative_count = sum(1 for article in articles if article.get('sentiment', 0.0) < -0.05)
        neutral_count = len(articles) - positive_count - negative_count
        
        positive_ratio = positive_count / len(articles)
        negative_ratio = negative_count / len(articles)
        neutral_ratio = neutral_count / len(articles)
        
        # Create result
        result = {
            'compound': compound_avg,
            'positive': positive_ratio,
            'negative': negative_ratio,
            'neutral': neutral_ratio,
            'count': len(articles),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count
        }
        
        # Store result
        self.sentiment_history.append({
            'timestamp': datetime.now(),
            'type': 'news',
            'result': result
        })
        
        logger.info(f"Analyzed sentiment of {len(articles)} news articles: {compound_avg:.4f}")
        
        return result
    
    def analyze_social_sentiment(self, posts: List[Dict[str, Any]], model: str = None) -> Dict[str, Any]:
        """
        Analyze sentiment of social media posts.
        
        Args:
            posts: List of social media posts
            model: Model to use (default: from config)
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not posts:
            return {
                'compound': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 0.0,
                'count': 0
            }
        
        # Set default model
        if model is None:
            model = self.default_model
        
        # Analyze sentiment of each post
        for post in posts:
            # Skip if already analyzed
            if post.get('sentiment', 0.0) != 0.0:
                continue
            
            # Get text for analysis
            text = post.get('text', '')
            
            # Analyze sentiment
            sentiment = self.analyze_text(text, model)
            
            # Update post with sentiment
            post['sentiment'] = sentiment.get('compound', 0.0)
        
        # Calculate aggregate sentiment
        compound_sum = sum(post.get('sentiment', 0.0) for post in posts)
        compound_avg = compound_sum / len(posts)
        
        positive_count = sum(1 for post in posts if post.get('sentiment', 0.0) > 0.05)
        negative_count = sum(1 for post in posts if post.get('sentiment', 0.0) < -0.05)
        neutral_count = len(posts) - positive_count - negative_count
        
        positive_ratio = positive_count / len(posts)
        negative_ratio = negative_count / len(posts)
        neutral_ratio = neutral_count / len(posts)
        
        # Create result
        result = {
            'compound': compound_avg,
            'positive': positive_ratio,
            'negative': negative_ratio,
            'neutral': neutral_ratio,
            'count': len(posts),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count
        }
        
        # Store result
        self.sentiment_history.append({
            'timestamp': datetime.now(),
            'type': 'social',
            'result': result
        })
        
        logger.info(f"Analyzed sentiment of {len(posts)} social media posts: {compound_avg:.4f}")
        
        return result
    
    def get_combined_sentiment(self, news_sentiment: Dict[str, Any], social_sentiment: Dict[str, Any],
                              news_weight: float = 0.6, social_weight: float = 0.4) -> Dict[str, Any]:
        """
        Get combined sentiment from news and social media.
        
        Args:
            news_sentiment: News sentiment results
            social_sentiment: Social media sentiment results
            news_weight: Weight for news sentiment
            social_weight: Weight for social media sentiment
            
        Returns:
            Dictionary with combined sentiment results
        """
        # Normalize weights
        total_weight = news_weight + social_weight
        news_weight = news_weight / total_weight
        social_weight = social_weight / total_weight
        
        # Calculate combined sentiment
        compound = news_sentiment.get('compound', 0.0) * news_weight + social_sentiment.get('compound', 0.0) * social_weight
        positive = news_sentiment.get('positive', 0.0) * news_weight + social_sentiment.get('positive', 0.0) * social_weight
        negative = news_sentiment.get('negative', 0.0) * news_weight + social_sentiment.get('negative', 0.0) * social_weight
        neutral = news_sentiment.get('neutral', 0.0) * news_weight + social_sentiment.get('neutral', 0.0) * social_weight
        
        # Create result
        result = {
            'compound': compound,
            'positive': positive,
            'negative': negative,
            'neutral': neutral,
            'news_count': news_sentiment.get('count', 0),
            'social_count': social_sentiment.get('count', 0),
            'news_weight': news_weight,
            'social_weight': social_weight
        }
        
        # Store result
        self.sentiment_history.append({
            'timestamp': datetime.now(),
            'type': 'combined',
            'result': result
        })
        
        logger.info(f"Combined sentiment: {compound:.4f}")
        
        return result
    
    def get_sentiment_features(self, sentiment: Dict[str, Any]) -> Dict[str, float]:
        """
        Get sentiment features for machine learning.
        
        Args:
            sentiment: Sentiment analysis results
            
        Returns:
            Dictionary with sentiment features
        """
        features = {
            'sentiment_compound': sentiment.get('compound', 0.0),
            'sentiment_positive': sentiment.get('positive', 0.0),
            'sentiment_negative': sentiment.get('negative', 0.0),
            'sentiment_neutral': sentiment.get('neutral', 0.0),
            'sentiment_ratio': sentiment.get('positive', 0.0) / max(0.01, sentiment.get('negative', 0.0))
        }
        
        return features
    
    def plot_sentiment_history(self, days: int = 30, save_plot: bool = True,
                              filename: str = 'sentiment_history.png') -> None:
        """
        Plot sentiment history.
        
        Args:
            days: Number of days to plot
            save_plot: Whether to save the plot
            filename: Filename for saved plot
        """
        if not self.sentiment_history:
            logger.warning("No sentiment history to plot")
            return
        
        try:
            # Filter history by date
            cutoff_date = datetime.now() - timedelta(days=days)
            history = [entry for entry in self.sentiment_history if entry['timestamp'] > cutoff_date]
            
            if not history:
                logger.warning(f"No sentiment history within the last {days} days")
                return
            
            # Group by type
            news_history = [entry for entry in history if entry['type'] == 'news']
            social_history = [entry for entry in history if entry['type'] == 'social']
            combined_history = [entry for entry in history if entry['type'] == 'combined']
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot news sentiment
            if news_history:
                timestamps = [entry['timestamp'] for entry in news_history]
                compounds = [entry['result']['compound'] for entry in news_history]
                ax.plot(timestamps, compounds, 'b-', label='News Sentiment')
            
            # Plot social sentiment
            if social_history:
                timestamps = [entry['timestamp'] for entry in social_history]
                compounds = [entry['result']['compound'] for entry in social_history]
                ax.plot(timestamps, compounds, 'g-', label='Social Sentiment')
            
            # Plot combined sentiment
            if combined_history:
                timestamps = [entry['timestamp'] for entry in combined_history]
                compounds = [entry['result']['compound'] for entry in combined_history]
                ax.plot(timestamps, compounds, 'r-', label='Combined Sentiment')
            
            # Add neutral line
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Add labels
            ax.set_xlabel('Date')
            ax.set_ylabel('Sentiment (Compound)')
            ax.set_title('Sentiment History')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Format x-axis
            fig.autofmt_xdate()
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            if save_plot:
                plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight', dpi=300)
                logger.info(f"Sentiment history plot saved to {self.output_dir}/{filename}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting sentiment history: {e}")
            raise
    
    def plot_sentiment_distribution(self, articles: List[Dict[str, Any]] = None,
                                   posts: List[Dict[str, Any]] = None,
                                   save_plot: bool = True,
                                   filename: str = 'sentiment_distribution.png') -> None:
        """
        Plot sentiment distribution.
        
        Args:
            articles: List of news articles
            posts: List of social media posts
            save_plot: Whether to save the plot
            filename: Filename for saved plot
        """
        try:
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot news sentiment distribution
            if articles:
                sentiments = [article.get('sentiment', 0.0) for article in articles]
                ax1.hist(sentiments, bins=20, alpha=0.7, color='blue')
                ax1.set_xlabel('Sentiment')
                ax1.set_ylabel('Count')
                ax1.set_title(f'News Sentiment Distribution (n={len(articles)})')
                ax1.grid(True, linestyle='--', alpha=0.7)
                
                # Add mean line
                mean_sentiment = sum(sentiments) / len(sentiments)
                ax1.axvline(x=mean_sentiment, color='red', linestyle='--', label=f'Mean: {mean_sentiment:.2f}')
                ax1.legend()
            else:
                ax1.set_title('No News Data Available')
            
            # Plot social sentiment distribution
            if posts:
                sentiments = [post.get('sentiment', 0.0) for post in posts]
                ax2.hist(sentiments, bins=20, alpha=0.7, color='green')
                ax2.set_xlabel('Sentiment')
                ax2.set_ylabel('Count')
                ax2.set_title(f'Social Media Sentiment Distribution (n={len(posts)})')
                ax2.grid(True, linestyle='--', alpha=0.7)
                
                # Add mean line
                mean_sentiment = sum(sentiments) / len(sentiments)
                ax2.axvline(x=mean_sentiment, color='red', linestyle='--', label=f'Mean: {mean_sentiment:.2f}')
                ax2.legend()
            else:
                ax2.set_title('No Social Media Data Available')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            if save_plot:
                plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight', dpi=300)
                logger.info(f"Sentiment distribution plot saved to {self.output_dir}/{filename}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting sentiment distribution: {e}")
            raise
    
    def save_cache(self, filepath: str = None) -> str:
        """
        Save cache to file.
        
        Args:
            filepath: Path to save cache (default: cache_dir/sentiment_cache.pkl)
            
        Returns:
            Path to saved cache
        """
        if filepath is None:
            filepath = os.path.join(self.cache_dir, 'sentiment_cache.pkl')
        
        # Prepare cache to save
        cache = {
            'sentiment_cache': self.sentiment_cache,
            'news_cache': self.news_cache,
            'social_cache': self.social_cache,
            'sentiment_history': self.sentiment_history
        }
        
        # Save cache
        joblib.dump(cache, filepath)
        logger.info(f"Sentiment cache saved to {filepath}")
        
        return filepath
    
    def load_cache(self, filepath: str = None) -> None:
        """
        Load cache from file.
        
        Args:
            filepath: Path to load cache from (default: cache_dir/sentiment_cache.pkl)
        """
        if filepath is None:
            filepath = os.path.join(self.cache_dir, 'sentiment_cache.pkl')
        
        if not os.path.exists(filepath):
            logger.warning(f"Cache file {filepath} not found")
            return
        
        # Load cache
        cache = joblib.load(filepath)
        
        # Restore cache
        self.sentiment_cache = cache.get('sentiment_cache', {})
        self.news_cache = cache.get('news_cache', {})
        self.social_cache = cache.get('social_cache', {})
        self.sentiment_history = cache.get('sentiment_history', [])
        
        logger.info(f"Sentiment cache loaded from {filepath}")
