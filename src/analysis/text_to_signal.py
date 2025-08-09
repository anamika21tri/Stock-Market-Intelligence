"""
Text-to-signal conversion module for transforming tweet content into trading signals
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
import re

from ..utils import get_logger, get_config

logger = get_logger(__name__)


class MarketSentimentAnalyzer:
    """Analyzes market sentiment from tweet content"""
    
    def __init__(self):
        # Market-specific sentiment words
        self.bullish_words = {
            'strong': ['buy', 'bull', 'bullish', 'long', 'rise', 'up', 'gain', 'profit', 'green', 'rocket', 'moon', 'surge', 'breakout', 'rally'],
            'moderate': ['positive', 'good', 'better', 'growth', 'increase', 'support', 'strong', 'confident', 'optimistic'],
            'hindi': ['तेजी', 'बुल', 'खरीद', 'फायदा', 'बढ़ोतरी']
        }
        
        self.bearish_words = {
            'strong': ['sell', 'bear', 'bearish', 'short', 'fall', 'down', 'loss', 'red', 'crash', 'dump', 'drop', 'breakdown'],
            'moderate': ['negative', 'bad', 'worse', 'decline', 'decrease', 'resistance', 'weak', 'worried', 'pessimistic'],
            'hindi': ['मंदी', 'बेयर', 'बेच', 'नुकसान', 'गिरावट']
        }
        
        # Market action words with weights
        self.action_words = {
            'buy': 2.0, 'sell': -2.0, 'hold': 0.0,
            'long': 1.5, 'short': -1.5,
            'call': 1.0, 'put': -1.0,
            'bullish': 1.8, 'bearish': -1.8,
            'breakout': 2.5, 'breakdown': -2.5
        }
        
        logger.info("Market sentiment analyzer initialized")
    
    def calculate_sentiment_score(self, text: str) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate sentiment score with market-specific context
        
        Returns:
            Tuple of (sentiment_score, sentiment_details)
        """
        if not isinstance(text, str) or not text.strip():
            return 0.0, {}
        
        text_lower = text.lower()
        
        # Basic sentiment using TextBlob
        blob = TextBlob(text)
        basic_sentiment = blob.sentiment.polarity
        
        # Market-specific sentiment
        market_sentiment = self._calculate_market_sentiment(text_lower)
        
        # Action words sentiment
        action_sentiment = self._calculate_action_sentiment(text_lower)
        
        # Price movement indicators
        price_sentiment = self._analyze_price_indicators(text)
        
        # Combine sentiments with weights
        combined_sentiment = (
            basic_sentiment * 0.3 +
            market_sentiment * 0.4 +
            action_sentiment * 0.2 +
            price_sentiment * 0.1
        )
        
        # Normalize to [-1, 1] range
        final_sentiment = max(-1.0, min(1.0, combined_sentiment))
        
        details = {
            'basic_sentiment': basic_sentiment,
            'market_sentiment': market_sentiment,
            'action_sentiment': action_sentiment,
            'price_sentiment': price_sentiment,
            'final_sentiment': final_sentiment,
            'confidence': blob.sentiment.subjectivity
        }
        
        return final_sentiment, details
    
    def _calculate_market_sentiment(self, text: str) -> float:
        """Calculate sentiment based on market-specific words"""
        bullish_score = 0
        bearish_score = 0
        
        # Check bullish words
        for category, words in self.bullish_words.items():
            weight = 2.0 if category == 'strong' else 1.0
            for word in words:
                count = len(re.findall(rf'\b{word}\b', text))
                bullish_score += count * weight
        
        # Check bearish words
        for category, words in self.bearish_words.items():
            weight = 2.0 if category == 'strong' else 1.0
            for word in words:
                count = len(re.findall(rf'\b{word}\b', text))
                bearish_score += count * weight
        
        # Calculate net sentiment
        if bullish_score + bearish_score == 0:
            return 0.0
        
        net_sentiment = (bullish_score - bearish_score) / (bullish_score + bearish_score)
        return net_sentiment
    
    def _calculate_action_sentiment(self, text: str) -> float:
        """Calculate sentiment based on trading actions"""
        total_weight = 0
        word_count = 0
        
        for word, weight in self.action_words.items():
            count = len(re.findall(rf'\b{word}\b', text))
            if count > 0:
                total_weight += weight * count
                word_count += count
        
        return total_weight / word_count if word_count > 0 else 0.0
    
    def _analyze_price_indicators(self, text: str) -> float:
        """Analyze price movement indicators"""
        # Look for percentage changes
        percentage_pattern = r'[+-]?\d+(?:\.\d+)?%'
        percentages = re.findall(percentage_pattern, text)
        
        if not percentages:
            return 0.0
        
        total_change = 0
        for pct_str in percentages:
            try:
                pct_value = float(pct_str.rstrip('%'))
                # Normalize large percentages
                normalized_pct = max(-10, min(10, pct_value)) / 10
                total_change += normalized_pct
            except ValueError:
                continue
        
        return total_change / len(percentages) if percentages else 0.0


class TextToSignalConverter:
    """
    Converts tweet text into quantitative trading signals
    """
    
    def __init__(self):
        self.config = get_config().analysis.text_processing
        self.signal_config = get_config().analysis.signal_generation
        
        # Initialize components
        self.sentiment_analyzer = MarketSentimentAnalyzer()
        self.vectorizer = self._initialize_vectorizer()
        self.scaler = StandardScaler()
        
        # Feature weights for signal generation
        self.feature_weights = {
            'sentiment': self.signal_config.get('sentiment_weight', 0.4),
            'volume': self.signal_config.get('volume_weight', 0.3),
            'engagement': self.signal_config.get('engagement_weight', 0.3)
        }
        
        logger.info("Text-to-signal converter initialized")
    
    def _initialize_vectorizer(self) -> TfidfVectorizer:
        """Initialize text vectorizer based on configuration"""
        vectorization_method = self.config.get('vectorization_method', 'tfidf')
        max_features = self.config.get('max_features', 10000)
        ngram_range = tuple(self.config.get('ngram_range', [1, 2]))
        
        if vectorization_method == 'tfidf':
            return TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words='english',
                lowercase=True,
                token_pattern=r'\b\w+\b'
            )
        else:
            return CountVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words='english',
                lowercase=True
            )
    
    def extract_text_features(self, tweets_df: pd.DataFrame) -> pd.DataFrame:
        """Extract comprehensive text features from tweets"""
        logger.info(f"Extracting text features from {len(tweets_df)} tweets")
        
        features_df = tweets_df.copy()
        
        # Sentiment features
        sentiment_data = []
        for content in tweets_df['content']:
            sentiment_score, sentiment_details = self.sentiment_analyzer.calculate_sentiment_score(content)
            sentiment_data.append({
                'sentiment_score': sentiment_score,
                'sentiment_confidence': sentiment_details.get('confidence', 0),
                'basic_sentiment': sentiment_details.get('basic_sentiment', 0),
                'market_sentiment': sentiment_details.get('market_sentiment', 0)
            })
        
        sentiment_df = pd.DataFrame(sentiment_data)
        features_df = pd.concat([features_df, sentiment_df], axis=1)
        
        # Text statistics
        features_df['content_length'] = features_df['content'].str.len()
        features_df['word_count'] = features_df['content'].str.split().str.len()
        features_df['hashtag_count'] = features_df['hashtags'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        features_df['mention_count'] = features_df['mentions'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        
        # Market-specific features
        features_df['has_price_mention'] = features_df['content'].str.contains(r'₹|\$|\d+%', regex=True, na=False)
        features_df['has_target_hashtag'] = features_df['hashtags'].apply(self._contains_target_hashtag)
        features_df['urgency_score'] = features_df['content'].apply(self._calculate_urgency_score)
        
        # Engagement features (normalized)
        engagement_cols = ['replies_count', 'retweets_count', 'likes_count', 'views_count']
        for col in engagement_cols:
            if col in features_df.columns:
                # Log transform to handle skewed distributions
                features_df[f'{col}_log'] = np.log1p(features_df[col].fillna(0))
        
        # Time-based features
        if 'timestamp' in features_df.columns:
            features_df['hour'] = pd.to_datetime(features_df['timestamp']).dt.hour
            features_df['day_of_week'] = pd.to_datetime(features_df['timestamp']).dt.dayofweek
            features_df['is_market_hours'] = features_df['hour'].apply(self._is_market_hours)
        
        logger.info(f"Extracted {features_df.shape[1]} features")
        return features_df
    
    def _contains_target_hashtag(self, hashtags: List[str]) -> bool:
        """Check if tweet contains target market hashtags"""
        if not isinstance(hashtags, list):
            return False
        
        target_hashtags = ['#nifty50', '#sensex', '#banknifty', '#finnifty', '#intraday']
        hashtags_lower = [tag.lower() for tag in hashtags]
        
        return any(target in hashtags_lower for target in target_hashtags)
    
    def _calculate_urgency_score(self, text: str) -> float:
        """Calculate urgency score based on text patterns"""
        if not isinstance(text, str):
            return 0.0
        
        text_lower = text.lower()
        urgency_indicators = {
            'urgent': 3.0,
            'breaking': 3.0,
            'alert': 2.5,
            'now': 2.0,
            'immediate': 2.5,
            'quickly': 2.0,
            'asap': 3.0,
            '!!!': 2.0,
            'emergency': 3.0
        }
        
        urgency_score = 0
        for indicator, score in urgency_indicators.items():
            if indicator in text_lower:
                urgency_score += score
        
        # Normalize to [0, 1] range
        return min(1.0, urgency_score / 5.0)
    
    def _is_market_hours(self, hour: int) -> bool:
        """Check if time is within market hours (9:15 AM to 3:30 PM IST)"""
        return 9 <= hour <= 15
    
    def create_vectorized_features(self, tweets_df: pd.DataFrame) -> np.ndarray:
        """Create vectorized features from tweet content"""
        logger.info("Creating vectorized features")
        
        # Combine content with hashtags for better vectorization
        combined_text = tweets_df.apply(
            lambda row: f"{row['content']} {' '.join(row.get('hashtags', []))}", 
            axis=1
        )
        
        # Fit and transform
        tfidf_features = self.vectorizer.fit_transform(combined_text)
        
        # Reduce dimensionality if too many features
        if tfidf_features.shape[1] > 1000:
            logger.info("Reducing dimensionality with PCA")
            pca = PCA(n_components=min(1000, tfidf_features.shape[0] - 1))
            tfidf_features = pca.fit_transform(tfidf_features.toarray())
        
        return tfidf_features
    
    def generate_trading_signals(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from extracted features
        
        Returns:
            DataFrame with trading signals and confidence scores
        """
        logger.info(f"Generating trading signals from {len(features_df)} tweets")
        
        signals_df = features_df.copy()
        
        # Calculate base signal from sentiment
        sentiment_signal = signals_df['sentiment_score'].fillna(0)
        
        # Calculate volume signal from engagement
        engagement_cols = ['replies_count_log', 'retweets_count_log', 'likes_count_log']
        available_engagement_cols = [col for col in engagement_cols if col in signals_df.columns]
        
        if available_engagement_cols:
            # Normalize engagement metrics
            engagement_normalized = signals_df[available_engagement_cols].fillna(0)
            engagement_normalized = (engagement_normalized - engagement_normalized.mean()) / (engagement_normalized.std() + 1e-8)
            volume_signal = engagement_normalized.mean(axis=1)
        else:
            volume_signal = pd.Series(0, index=signals_df.index)
        
        # Calculate urgency signal
        urgency_signal = signals_df.get('urgency_score', pd.Series(0, index=signals_df.index))
        
        # Combine signals with weights
        composite_signal = (
            sentiment_signal * self.feature_weights['sentiment'] +
            volume_signal * self.feature_weights['volume'] +
            urgency_signal * self.feature_weights['engagement']
        )
        
        # Normalize composite signal to [-1, 1]
        if composite_signal.std() > 0:
            composite_signal = (composite_signal - composite_signal.mean()) / composite_signal.std()
            composite_signal = np.tanh(composite_signal)  # Smooth normalization
        
        # Calculate confidence based on multiple factors
        confidence = self._calculate_signal_confidence(signals_df, composite_signal)
        
        # Create signal categories
        confidence_threshold = self.signal_config.get('confidence_threshold', 0.6)
        
        signals_df['signal_strength'] = composite_signal
        signals_df['signal_confidence'] = confidence
        signals_df['signal_category'] = pd.cut(
            composite_signal,
            bins=[-np.inf, -0.3, -0.1, 0.1, 0.3, np.inf],
            labels=['strong_bearish', 'weak_bearish', 'neutral', 'weak_bullish', 'strong_bullish']
        )
        
        # Filter by confidence
        signals_df['high_confidence'] = confidence >= confidence_threshold
        
        # Add timestamp-based features for signal aggregation
        if 'timestamp' in signals_df.columns:
            signals_df['signal_timestamp'] = pd.to_datetime(signals_df['timestamp'])
            signals_df['signal_hour'] = signals_df['signal_timestamp'].dt.hour
        
        logger.info(f"Generated signals: {len(signals_df[signals_df['high_confidence']])} high-confidence signals")
        
        return signals_df
    
    def _calculate_signal_confidence(self, features_df: pd.DataFrame, signal: pd.Series) -> pd.Series:
        """Calculate confidence score for signals"""
        confidence_factors = []
        
        # Sentiment confidence
        if 'sentiment_confidence' in features_df.columns:
            confidence_factors.append(features_df['sentiment_confidence'].fillna(0))
        
        # Engagement confidence (higher engagement = higher confidence)
        engagement_cols = ['replies_count_log', 'retweets_count_log', 'likes_count_log']
        available_engagement = [col for col in engagement_cols if col in features_df.columns]
        
        if available_engagement:
            engagement_sum = features_df[available_engagement].sum(axis=1)
            # Normalize to [0, 1]
            if engagement_sum.max() > 0:
                engagement_confidence = engagement_sum / engagement_sum.max()
                confidence_factors.append(engagement_confidence)
        
        # Target hashtag confidence
        if 'has_target_hashtag' in features_df.columns:
            hashtag_confidence = features_df['has_target_hashtag'].astype(float) * 0.5 + 0.5
            confidence_factors.append(hashtag_confidence)
        
        # Signal strength confidence (stronger signals are more confident)
        signal_strength_confidence = 1 - np.exp(-np.abs(signal) * 2)
        confidence_factors.append(signal_strength_confidence)
        
        # Combine confidence factors
        if confidence_factors:
            combined_confidence = pd.concat(confidence_factors, axis=1).mean(axis=1)
        else:
            combined_confidence = pd.Series(0.5, index=features_df.index)
        
        return combined_confidence.clip(0, 1)
    
    def process_tweets_to_signals(self, tweets: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Complete pipeline from tweets to trading signals
        
        Args:
            tweets: List of tweet dictionaries
            
        Returns:
            DataFrame with extracted features and trading signals
        """
        logger.info(f"Processing {len(tweets)} tweets to trading signals")
        
        # Convert to DataFrame
        tweets_df = pd.DataFrame(tweets)
        
        if tweets_df.empty:
            logger.warning("No tweets to process")
            return pd.DataFrame()
        
        # Extract features
        features_df = self.extract_text_features(tweets_df)
        
        # Generate signals
        signals_df = self.generate_trading_signals(features_df)
        
        logger.info(f"Signal processing completed: {len(signals_df)} tweets processed")
        
        return signals_df


def create_text_to_signal_converter() -> TextToSignalConverter:
    """Factory function to create text-to-signal converter"""
    return TextToSignalConverter()
