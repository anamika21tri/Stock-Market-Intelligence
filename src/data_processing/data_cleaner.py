"""
Data cleaning and normalization module for tweet data
"""
import re
import unicodedata
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import html
import emoji

from ..utils import get_logger, get_config

logger = get_logger(__name__)


class DataCleaner:
    """
    Comprehensive data cleaning for tweet data with focus on Indian market content
    """
    
    def __init__(self):
        self.config = get_config().data_processing.cleaning
        
        # Market-specific patterns
        self.indian_market_patterns = {
            'stock_symbols': r'\b(?:NIFTY|SENSEX|BANKNIFTY|FINNIFTY|RELIANCE|TCS|INFY|HDFC|ICICI)\b',
            'price_patterns': r'₹?\d+(?:[.,]\d+)*(?:\s*(?:cr|crore|lakh|k|million|billion))?',
            'percentage_patterns': r'[+-]?\d+(?:\.\d+)?%',
            'market_timings': r'(?:9:15|9\.15|15:30|15\.30|pre-market|post-market|opening|closing)',
        }
        
        # Common Indian language terms that should be preserved
        self.preserve_terms = {
            'hindi': ['बुल', 'बेयर', 'तेजी', 'मंदी', 'निफ्टी', 'सेंसेक्स'],
            'market_terms': ['bull', 'bear', 'bullish', 'bearish', 'long', 'short', 'calls', 'puts']
        }
        
        logger.info("Data cleaner initialized")
    
    def clean_tweet_text(self, text: str) -> str:
        """
        Clean and normalize tweet text while preserving market-relevant content
        """
        if not isinstance(text, str):
            return ""
        
        original_text = text
        
        try:
            # HTML decode
            text = html.unescape(text)
            
            # Normalize unicode
            if self.config.get('normalize_unicode', True):
                text = unicodedata.normalize('NFKC', text)
            
            # Remove excessive whitespace but preserve structure
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            # Remove URLs if configured
            if self.config.get('remove_urls', True):
                text = self._remove_urls(text)
            
            # Clean mentions and hashtags based on config
            if self.config.get('remove_mentions', False):
                text = re.sub(r'@\w+', '', text)
            
            if self.config.get('remove_hashtags', False):
                text = re.sub(r'#\w+', '', text)
            
            # Remove excessive punctuation but preserve market indicators
            text = self._clean_punctuation(text)
            
            # Handle emojis (convert to text or remove)
            text = self._handle_emojis(text)
            
            # Preserve important market terms
            text = self._preserve_market_terms(text)
            
            # Remove very short content
            min_length = self.config.get('min_content_length', 10)
            if len(text.strip()) < min_length:
                return ""
            
            return text.strip()
            
        except Exception as e:
            logger.warning(f"Error cleaning text '{original_text[:50]}...': {e}")
            return original_text
    
    def _remove_urls(self, text: str) -> str:
        """Remove various URL patterns"""
        url_patterns = [
            r'https?://\S+',
            r'www\.\S+',
            r't\.co/\w+',
            r'pic\.twitter\.com/\w+',
            r'tinyurl\.com/\w+',
            r'bit\.ly/\w+'
        ]
        
        for pattern in url_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text
    
    def _clean_punctuation(self, text: str) -> str:
        """Clean excessive punctuation while preserving market indicators"""
        # Preserve important punctuation patterns
        text = re.sub(r'\.{3,}', '...', text)  # Multiple dots to ellipsis
        text = re.sub(r'!{2,}', '!', text)     # Multiple exclamations
        text = re.sub(r'\?{2,}', '?', text)    # Multiple questions
        
        # Remove excessive special characters but keep market-relevant ones
        text = re.sub(r'[^\w\s\-+%₹$#@.,!?():/]', ' ', text)
        
        return text
    
    def _handle_emojis(self, text: str) -> str:
        """Handle emojis by converting to text description"""
        try:
            # Convert emojis to text description
            text = emoji.demojize(text, delimiters=(' ', ' '))
            
            # Clean up emoji descriptions
            text = re.sub(r'\s+', ' ', text)
            
            return text
        except Exception:
            # If emoji package fails, just remove emojis
            return re.sub(r'[^\w\s\-+%₹$#@.,!?():/]', ' ', text)
    
    def _preserve_market_terms(self, text: str) -> str:
        """Ensure important market terms are preserved and standardized"""
        # Standardize common market terms
        replacements = {
            r'\bnse\b': 'NSE',
            r'\bbse\b': 'BSE',
            r'\bnifty\s*50\b': 'NIFTY50',
            r'\bbank\s*nifty\b': 'BANKNIFTY',
            r'\bfin\s*nifty\b': 'FINNIFTY',
            r'\brupees?\b': '₹',
            r'\brs\.?\s*(\d+)': r'₹\1',
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def clean_username(self, username: str) -> str:
        """Clean and validate username"""
        if not isinstance(username, str):
            return ""
        
        # Remove @ symbol and normalize
        username = username.replace('@', '').strip()
        
        # Remove non-alphanumeric characters except underscore
        username = re.sub(r'[^\w]', '', username)
        
        return username.lower()
    
    def normalize_hashtags(self, hashtags: List[str]) -> List[str]:
        """Normalize hashtag format"""
        if not isinstance(hashtags, list):
            return []
        
        normalized = []
        for tag in hashtags:
            if isinstance(tag, str):
                # Remove # symbol and normalize
                clean_tag = tag.replace('#', '').strip().lower()
                if clean_tag and len(clean_tag) > 1:
                    normalized.append(f"#{clean_tag}")
        
        return list(set(normalized))  # Remove duplicates
    
    def normalize_mentions(self, mentions: List[str]) -> List[str]:
        """Normalize mention format"""
        if not isinstance(mentions, list):
            return []
        
        normalized = []
        for mention in mentions:
            if isinstance(mention, str):
                clean_mention = self.clean_username(mention)
                if clean_mention:
                    normalized.append(f"@{clean_mention}")
        
        return list(set(normalized))  # Remove duplicates
    
    def validate_engagement_metrics(self, metrics: Dict[str, Any]) -> Dict[str, int]:
        """Validate and clean engagement metrics"""
        validated = {}
        
        metric_fields = ['replies_count', 'retweets_count', 'likes_count', 'views_count']
        
        for field in metric_fields:
            value = metrics.get(field, 0)
            
            # Convert to integer, default to 0 if invalid
            try:
                validated[field] = max(0, int(value))
            except (ValueError, TypeError):
                validated[field] = 0
        
        return validated
    
    def clean_timestamp(self, timestamp: Any) -> Optional[datetime]:
        """Clean and validate timestamp"""
        if isinstance(timestamp, datetime):
            return timestamp
        
        if isinstance(timestamp, str):
            # Try various timestamp formats
            formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%S.%f',
                '%Y-%m-%d',
                '%d/%m/%Y %H:%M:%S',
                '%d-%m-%Y %H:%M:%S'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(timestamp, fmt)
                except ValueError:
                    continue
        
        # If all parsing fails, return None
        logger.warning(f"Could not parse timestamp: {timestamp}")
        return None
    
    def clean_tweet_data(self, tweet_data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean complete tweet data structure"""
        cleaned = {}
        
        try:
            # Clean text content
            cleaned['content'] = self.clean_tweet_text(tweet_data.get('content', ''))
            
            # Skip if content is too short after cleaning
            if len(cleaned['content']) < self.config.get('min_content_length', 10):
                return {}
            
            # Clean username
            cleaned['username'] = self.clean_username(tweet_data.get('username', ''))
            cleaned['display_name'] = tweet_data.get('display_name', cleaned['username'])
            
            # Clean timestamp
            cleaned['timestamp'] = self.clean_timestamp(tweet_data.get('timestamp'))
            if not cleaned['timestamp']:
                cleaned['timestamp'] = datetime.now()
            
            # Clean hashtags and mentions
            cleaned['hashtags'] = self.normalize_hashtags(tweet_data.get('hashtags', []))
            cleaned['mentions'] = self.normalize_mentions(tweet_data.get('mentions', []))
            
            # Clean engagement metrics
            engagement_metrics = self.validate_engagement_metrics(tweet_data)
            cleaned.update(engagement_metrics)
            
            # Preserve other important fields
            cleaned['id'] = tweet_data.get('id', '')
            cleaned['url'] = tweet_data.get('url', '')
            cleaned['is_retweet'] = bool(tweet_data.get('is_retweet', False))
            cleaned['language'] = tweet_data.get('language', 'en')
            
            # Add cleaning metadata
            cleaned['cleaned_at'] = datetime.now()
            cleaned['original_length'] = len(str(tweet_data.get('content', '')))
            cleaned['cleaned_length'] = len(cleaned['content'])
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error cleaning tweet data: {e}")
            return {}
    
    def clean_dataset(self, tweets: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Clean entire dataset and return cleaned data with statistics
        
        Returns:
            Tuple of (cleaned_tweets, cleaning_stats)
        """
        logger.info(f"Starting to clean {len(tweets)} tweets")
        
        cleaned_tweets = []
        stats = {
            'original_count': len(tweets),
            'cleaned_count': 0,
            'removed_count': 0,
            'avg_length_reduction': 0,
            'error_count': 0
        }
        
        length_reductions = []
        
        for i, tweet in enumerate(tweets):
            try:
                cleaned_tweet = self.clean_tweet_data(tweet)
                
                if cleaned_tweet:
                    cleaned_tweets.append(cleaned_tweet)
                    stats['cleaned_count'] += 1
                    
                    # Track length reduction
                    original_len = cleaned_tweet.get('original_length', 0)
                    cleaned_len = cleaned_tweet.get('cleaned_length', 0)
                    if original_len > 0:
                        reduction = (original_len - cleaned_len) / original_len
                        length_reductions.append(reduction)
                else:
                    stats['removed_count'] += 1
                
                # Log progress
                if (i + 1) % 100 == 0:
                    logger.info(f"Cleaned {i + 1}/{len(tweets)} tweets")
                    
            except Exception as e:
                logger.error(f"Error cleaning tweet {i}: {e}")
                stats['error_count'] += 1
        
        # Calculate final statistics
        if length_reductions:
            stats['avg_length_reduction'] = np.mean(length_reductions)
        
        stats['success_rate'] = stats['cleaned_count'] / stats['original_count'] if stats['original_count'] > 0 else 0
        
        logger.info(f"Cleaning completed: {stats['cleaned_count']} tweets cleaned, "
                   f"{stats['removed_count']} removed, {stats['error_count']} errors")
        
        return cleaned_tweets, stats


def create_data_cleaner() -> DataCleaner:
    """Factory function to create data cleaner"""
    return DataCleaner()
