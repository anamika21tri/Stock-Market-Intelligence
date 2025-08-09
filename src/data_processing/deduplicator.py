"""
Data deduplication module for removing duplicate tweets efficiently
"""
import hashlib
import xxhash
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Set, Tuple, Optional
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

from ..utils import get_logger, get_config

logger = get_logger(__name__)


class Deduplicator:
    """
    Advanced deduplication using multiple strategies for accurate duplicate detection
    """
    
    def __init__(self):
        self.config = get_config().data_processing.deduplication
        self.hash_algorithm = self.config.get('hash_algorithm', 'xxhash')
        self.similarity_threshold = self.config.get('similarity_threshold', 0.95)
        
        # Initialize TF-IDF vectorizer for content similarity
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        
        logger.info("Deduplicator initialized")
    
    def _normalize_text_for_comparison(self, text: str) -> str:
        """Normalize text for comparison purposes"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs, mentions, hashtags for comparison
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        
        # Remove special characters and normalize whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate hash for tweet content"""
        normalized_content = self._normalize_text_for_comparison(content)
        
        if self.hash_algorithm == 'xxhash':
            return xxhash.xxh64(normalized_content.encode('utf-8')).hexdigest()
        else:
            return hashlib.md5(normalized_content.encode('utf-8')).hexdigest()
    
    def _generate_user_content_hash(self, username: str, content: str) -> str:
        """Generate hash combining user and content for stricter deduplication"""
        combined = f"{username.lower()}:{self._normalize_text_for_comparison(content)}"
        
        if self.hash_algorithm == 'xxhash':
            return xxhash.xxh64(combined.encode('utf-8')).hexdigest()
        else:
            return hashlib.md5(combined.encode('utf-8')).hexdigest()
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using TF-IDF and cosine similarity"""
        try:
            normalized_texts = [
                self._normalize_text_for_comparison(text1),
                self._normalize_text_for_comparison(text2)
            ]
            
            if not normalized_texts[0] or not normalized_texts[1]:
                return 0.0
            
            # Use TF-IDF vectorization
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(normalized_texts)
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            return similarity_matrix[0][1]
            
        except Exception as e:
            logger.debug(f"Error calculating text similarity: {e}")
            return 0.0
    
    def _is_retweet_variation(self, tweet1: Dict[str, Any], tweet2: Dict[str, Any]) -> bool:
        """Check if tweets are variations of the same retweet"""
        content1 = tweet1.get('content', '')
        content2 = tweet2.get('content', '')
        
        # Check for RT patterns
        rt_patterns = [r'^RT @\w+:', r'RT:', r'Retweet:', r'🔄']
        
        is_rt1 = any(re.search(pattern, content1, re.IGNORECASE) for pattern in rt_patterns)
        is_rt2 = any(re.search(pattern, content2, re.IGNORECASE) for pattern in rt_patterns)
        
        if is_rt1 and is_rt2:
            # Extract original content after RT marker
            cleaned1 = re.sub(r'^RT @\w+:\s*', '', content1, flags=re.IGNORECASE)
            cleaned2 = re.sub(r'^RT @\w+:\s*', '', content2, flags=re.IGNORECASE)
            
            return self._calculate_text_similarity(cleaned1, cleaned2) > 0.9
        
        return False
    
    def deduplicate_by_exact_content(self, tweets: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Remove tweets with identical content after normalization"""
        if not tweets:
            return tweets, {}
        
        logger.info(f"Starting exact content deduplication on {len(tweets)} tweets")
        
        seen_hashes = set()
        unique_tweets = []
        duplicates_found = 0
        
        for tweet in tweets:
            content_hash = self._generate_content_hash(tweet.get('content', ''))
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_tweets.append(tweet)
            else:
                duplicates_found += 1
        
        stats = {
            'original_count': len(tweets),
            'unique_count': len(unique_tweets),
            'duplicates_removed': duplicates_found,
            'deduplication_rate': duplicates_found / len(tweets) if tweets else 0
        }
        
        logger.info(f"Exact content deduplication: removed {duplicates_found} duplicates, "
                   f"kept {len(unique_tweets)} unique tweets")
        
        return unique_tweets, stats
    
    def deduplicate_by_user_content(self, tweets: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Remove tweets with same user posting identical content"""
        if not tweets:
            return tweets, {}
        
        logger.info(f"Starting user-content deduplication on {len(tweets)} tweets")
        
        seen_hashes = set()
        unique_tweets = []
        duplicates_found = 0
        
        for tweet in tweets:
            user_content_hash = self._generate_user_content_hash(
                tweet.get('username', ''),
                tweet.get('content', '')
            )
            
            if user_content_hash not in seen_hashes:
                seen_hashes.add(user_content_hash)
                unique_tweets.append(tweet)
            else:
                duplicates_found += 1
        
        stats = {
            'original_count': len(tweets),
            'unique_count': len(unique_tweets),
            'duplicates_removed': duplicates_found,
            'deduplication_rate': duplicates_found / len(tweets) if tweets else 0
        }
        
        logger.info(f"User-content deduplication: removed {duplicates_found} duplicates, "
                   f"kept {len(unique_tweets)} unique tweets")
        
        return unique_tweets, stats
    
    def deduplicate_by_similarity(self, tweets: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Remove tweets with high content similarity using TF-IDF"""
        if not tweets:
            return tweets, {}
        
        logger.info(f"Starting similarity-based deduplication on {len(tweets)} tweets")
        
        unique_tweets = []
        duplicates_found = 0
        processed_count = 0
        
        for i, tweet in enumerate(tweets):
            is_duplicate = False
            current_content = tweet.get('content', '')
            
            # Compare with existing unique tweets
            for unique_tweet in unique_tweets:
                similarity = self._calculate_text_similarity(
                    current_content,
                    unique_tweet.get('content', '')
                )
                
                if similarity > self.similarity_threshold:
                    is_duplicate = True
                    break
                
                # Also check for retweet variations
                if self._is_retweet_variation(tweet, unique_tweet):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_tweets.append(tweet)
            else:
                duplicates_found += 1
            
            processed_count += 1
            
            # Log progress for large datasets
            if processed_count % 100 == 0:
                logger.info(f"Processed {processed_count}/{len(tweets)} tweets for similarity")
        
        stats = {
            'original_count': len(tweets),
            'unique_count': len(unique_tweets),
            'duplicates_removed': duplicates_found,
            'deduplication_rate': duplicates_found / len(tweets) if tweets else 0,
            'similarity_threshold': self.similarity_threshold
        }
        
        logger.info(f"Similarity deduplication: removed {duplicates_found} duplicates, "
                   f"kept {len(unique_tweets)} unique tweets")
        
        return unique_tweets, stats
    
    def deduplicate_by_temporal_clustering(self, tweets: List[Dict[str, Any]], 
                                          time_window_minutes: int = 5) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Remove duplicates considering temporal proximity"""
        if not tweets:
            return tweets, {}
        
        logger.info(f"Starting temporal clustering deduplication on {len(tweets)} tweets")
        
        # Sort tweets by timestamp
        sorted_tweets = sorted(tweets, key=lambda x: x.get('timestamp', datetime.min))
        
        unique_tweets = []
        duplicates_found = 0
        
        for tweet in sorted_tweets:
            is_duplicate = False
            current_time = tweet.get('timestamp', datetime.now())
            current_content = tweet.get('content', '')
            
            # Check tweets within time window
            for unique_tweet in reversed(unique_tweets[-10:]):  # Check last 10 tweets for efficiency
                unique_time = unique_tweet.get('timestamp', datetime.min)
                time_diff = abs((current_time - unique_time).total_seconds() / 60)
                
                if time_diff <= time_window_minutes:
                    similarity = self._calculate_text_similarity(current_content, unique_tweet.get('content', ''))
                    
                    if similarity > 0.8:  # Lower threshold for temporal clustering
                        is_duplicate = True
                        break
                else:
                    break  # No need to check older tweets
            
            if not is_duplicate:
                unique_tweets.append(tweet)
            else:
                duplicates_found += 1
        
        stats = {
            'original_count': len(tweets),
            'unique_count': len(unique_tweets),
            'duplicates_removed': duplicates_found,
            'deduplication_rate': duplicates_found / len(tweets) if tweets else 0,
            'time_window_minutes': time_window_minutes
        }
        
        logger.info(f"Temporal clustering deduplication: removed {duplicates_found} duplicates, "
                   f"kept {len(unique_tweets)} unique tweets")
        
        return unique_tweets, stats
    
    def comprehensive_deduplication(self, tweets: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Perform comprehensive deduplication using multiple strategies
        
        Returns:
            Tuple of (deduplicated_tweets, comprehensive_stats)
        """
        logger.info(f"Starting comprehensive deduplication on {len(tweets)} tweets")
        
        original_count = len(tweets)
        current_tweets = tweets.copy()
        all_stats = {'stages': []}
        
        # Stage 1: Exact content deduplication
        current_tweets, exact_stats = self.deduplicate_by_exact_content(current_tweets)
        exact_stats['stage'] = 'exact_content'
        all_stats['stages'].append(exact_stats)
        
        # Stage 2: User-content deduplication
        current_tweets, user_stats = self.deduplicate_by_user_content(current_tweets)
        user_stats['stage'] = 'user_content'
        all_stats['stages'].append(user_stats)
        
        # Stage 3: Temporal clustering (for very similar tweets posted close in time)
        current_tweets, temporal_stats = self.deduplicate_by_temporal_clustering(current_tweets)
        temporal_stats['stage'] = 'temporal_clustering'
        all_stats['stages'].append(temporal_stats)
        
        # Stage 4: Similarity-based deduplication (most resource-intensive, so last)
        if len(current_tweets) <= 1000:  # Only for smaller datasets due to computational cost
            current_tweets, similarity_stats = self.deduplicate_by_similarity(current_tweets)
            similarity_stats['stage'] = 'similarity'
            all_stats['stages'].append(similarity_stats)
        else:
            logger.info("Skipping similarity deduplication for large dataset (>1000 tweets)")
        
        # Calculate overall statistics
        final_count = len(current_tweets)
        total_removed = original_count - final_count
        
        all_stats['summary'] = {
            'original_count': original_count,
            'final_count': final_count,
            'total_duplicates_removed': total_removed,
            'overall_deduplication_rate': total_removed / original_count if original_count > 0 else 0,
            'data_reduction_percentage': (total_removed / original_count * 100) if original_count > 0 else 0
        }
        
        logger.info(f"Comprehensive deduplication completed: "
                   f"{original_count} → {final_count} tweets "
                   f"({total_removed} duplicates removed, "
                   f"{all_stats['summary']['data_reduction_percentage']:.1f}% reduction)")
        
        return current_tweets, all_stats
    
    def find_potential_duplicates(self, tweets: List[Dict[str, Any]], 
                                 threshold: float = 0.8) -> List[Tuple[int, int, float]]:
        """
        Find potential duplicates without removing them (for manual review)
        
        Returns:
            List of tuples (index1, index2, similarity_score)
        """
        potential_duplicates = []
        
        for i in range(len(tweets)):
            for j in range(i + 1, len(tweets)):
                similarity = self._calculate_text_similarity(
                    tweets[i].get('content', ''),
                    tweets[j].get('content', '')
                )
                
                if similarity >= threshold:
                    potential_duplicates.append((i, j, similarity))
        
        # Sort by similarity score (highest first)
        potential_duplicates.sort(key=lambda x: x[2], reverse=True)
        
        logger.info(f"Found {len(potential_duplicates)} potential duplicate pairs")
        
        return potential_duplicates


def create_deduplicator() -> Deduplicator:
    """Factory function to create deduplicator"""
    return Deduplicator()
