"""
Signal aggregation module for combining individual tweet signals into trading insights
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from ..utils import get_logger, get_config

logger = get_logger(__name__)


class SignalAggregator:
    """
    Aggregates individual tweet signals into actionable trading insights
    """
    
    def __init__(self):
        self.config = get_config().analysis.signal_aggregation
        
        # Aggregation parameters
        self.time_windows = {
            '5min': timedelta(minutes=5),
            '15min': timedelta(minutes=15),
            '30min': timedelta(minutes=30),
            '1hour': timedelta(hours=1),
            '4hour': timedelta(hours=4)
        }
        
        # Signal strength thresholds
        self.signal_thresholds = {
            'weak': 0.2,
            'moderate': 0.5,
            'strong': 0.8
        }
        
        # Confidence requirements
        self.min_confidence = self.config.get('min_confidence', 0.6)
        self.min_tweets_per_signal = self.config.get('min_tweets_per_signal', 3)
        
        logger.info("Signal aggregator initialized")
    
    def aggregate_signals_by_time(self, signals_df: pd.DataFrame, window: str = '15min') -> pd.DataFrame:
        """
        Aggregate signals by time windows
        
        Args:
            signals_df: DataFrame with individual tweet signals
            window: Time window for aggregation ('5min', '15min', '30min', '1hour', '4hour')
            
        Returns:
            DataFrame with aggregated signals by time
        """
        logger.info(f"Aggregating signals by {window} time windows")
        
        if signals_df.empty or 'signal_timestamp' not in signals_df.columns:
            logger.warning("No timestamp data for aggregation")
            return pd.DataFrame()
        
        # Ensure timestamp column is datetime
        signals_df['signal_timestamp'] = pd.to_datetime(signals_df['signal_timestamp'])
        
        # Filter for high confidence signals only
        high_conf_signals = signals_df[signals_df['signal_confidence'] >= self.min_confidence].copy()
        
        if len(high_conf_signals) < self.min_tweets_per_signal:
            logger.warning(f"Insufficient high-confidence signals ({len(high_conf_signals)}) for aggregation")
            return pd.DataFrame()
        
        # Set timestamp as index for resampling
        high_conf_signals.set_index('signal_timestamp', inplace=True)
        
        # Resample by time window
        time_window = window
        if window.endswith('min'):
            time_window = window.replace('min', 'T')
        elif window.endswith('hour'):
            time_window = window.replace('hour', 'H')
        
        aggregated = high_conf_signals.resample(time_window).agg({
            'signal_strength': ['mean', 'std', 'count'],
            'signal_confidence': 'mean',
            'sentiment_score': ['mean', 'std'],
            'replies_count': 'sum' if 'replies_count' in high_conf_signals.columns else lambda x: 0,
            'retweets_count': 'sum' if 'retweets_count' in high_conf_signals.columns else lambda x: 0,
            'likes_count': 'sum' if 'likes_count' in high_conf_signals.columns else lambda x: 0,
            'views_count': 'sum' if 'views_count' in high_conf_signals.columns else lambda x: 0,
            'urgency_score': 'mean' if 'urgency_score' in high_conf_signals.columns else lambda x: 0
        }).round(4)
        
        # Flatten column names
        aggregated.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in aggregated.columns]
        
        # Filter periods with sufficient tweet count
        aggregated = aggregated[aggregated['signal_strength_count'] >= self.min_tweets_per_signal]
        
        # Calculate composite metrics
        aggregated['signal_intensity'] = (
            aggregated['signal_strength_mean'].abs() * 
            aggregated['signal_confidence_mean'] * 
            np.log1p(aggregated['signal_strength_count'])
        )
        
        # Calculate signal direction consistency
        if len(high_conf_signals) > 0:
            direction_consistency = high_conf_signals.resample(time_window)['signal_strength'].apply(
                lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0.5
            )
            aggregated['direction_consistency'] = direction_consistency
        
        # Calculate volatility indicator
        aggregated['signal_volatility'] = aggregated['signal_strength_std'].fillna(0)
        
        # Reset index to make timestamp a column
        aggregated.reset_index(inplace=True)
        aggregated.rename(columns={'signal_timestamp': 'time_window'}, inplace=True)
        
        # Add time window metadata
        aggregated['window_size'] = window
        aggregated['aggregation_timestamp'] = datetime.now()
        
        logger.info(f"Aggregated {len(aggregated)} time windows from {len(high_conf_signals)} signals")
        
        return aggregated
    
    def aggregate_signals_by_hashtag(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate signals by hashtag/topic
        
        Args:
            signals_df: DataFrame with individual tweet signals
            
        Returns:
            DataFrame with aggregated signals by hashtag
        """
        logger.info("Aggregating signals by hashtag")
        
        if signals_df.empty or 'hashtags' not in signals_df.columns:
            logger.warning("No hashtag data for aggregation")
            return pd.DataFrame()
        
        # Expand hashtags (each hashtag gets its own row)
        hashtag_signals = []
        
        for idx, row in signals_df.iterrows():
            if isinstance(row['hashtags'], list) and row['hashtags']:
                for hashtag in row['hashtags']:
                    hashtag_row = row.copy()
                    hashtag_row['hashtag'] = hashtag.lower()
                    hashtag_signals.append(hashtag_row)
        
        if not hashtag_signals:
            logger.warning("No hashtags found in signals")
            return pd.DataFrame()
        
        hashtag_df = pd.DataFrame(hashtag_signals)
        
        # Filter for high confidence signals
        hashtag_df = hashtag_df[hashtag_df['signal_confidence'] >= self.min_confidence]
        
        # Aggregate by hashtag
        hashtag_agg = hashtag_df.groupby('hashtag').agg({
            'signal_strength': ['mean', 'std', 'count'],
            'signal_confidence': 'mean',
            'sentiment_score': 'mean',
            'replies_count': 'sum' if 'replies_count' in hashtag_df.columns else lambda x: 0,
            'retweets_count': 'sum' if 'retweets_count' in hashtag_df.columns else lambda x: 0,
            'likes_count': 'sum' if 'likes_count' in hashtag_df.columns else lambda x: 0,
            'urgency_score': 'mean' if 'urgency_score' in hashtag_df.columns else lambda x: 0
        }).round(4)
        
        # Flatten column names
        hashtag_agg.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in hashtag_agg.columns]
        
        # Filter hashtags with sufficient signals
        hashtag_agg = hashtag_agg[hashtag_agg['signal_strength_count'] >= self.min_tweets_per_signal]
        
        # Calculate hashtag-specific metrics
        hashtag_agg['hashtag_momentum'] = (
            hashtag_agg['signal_strength_mean'] * 
            np.log1p(hashtag_agg['signal_strength_count'])
        )
        
        # Reset index
        hashtag_agg.reset_index(inplace=True)
        
        # Sort by momentum
        hashtag_agg = hashtag_agg.sort_values('hashtag_momentum', ascending=False)
        
        logger.info(f"Aggregated signals for {len(hashtag_agg)} hashtags")
        
        return hashtag_agg
    
    def create_trading_recommendations(self, signals_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create actionable trading recommendations from aggregated signals
        
        Args:
            signals_df: DataFrame with individual tweet signals
            
        Returns:
            Dictionary with trading recommendations
        """
        logger.info("Creating trading recommendations")
        
        if signals_df.empty:
            return {'status': 'no_data', 'recommendations': []}
        
        recommendations = {
            'timestamp': datetime.now(),
            'analysis_period': self._get_analysis_period(signals_df),
            'total_signals_analyzed': len(signals_df),
            'high_confidence_signals': len(signals_df[signals_df['signal_confidence'] >= self.min_confidence]),
            'recommendations': []
        }
        
        # Overall market sentiment
        high_conf_signals = signals_df[signals_df['signal_confidence'] >= self.min_confidence]
        
        if len(high_conf_signals) < self.min_tweets_per_signal:
            recommendations['status'] = 'insufficient_data'
            return recommendations
        
        # Calculate overall metrics
        overall_sentiment = high_conf_signals['signal_strength'].mean()
        sentiment_consistency = self._calculate_sentiment_consistency(high_conf_signals)
        signal_volume = len(high_conf_signals)
        confidence_level = high_conf_signals['signal_confidence'].mean()
        
        # Market direction recommendation
        direction_rec = self._get_direction_recommendation(
            overall_sentiment, sentiment_consistency, signal_volume, confidence_level
        )
        recommendations['recommendations'].append(direction_rec)
        
        # Time-based recommendations
        for window in ['15min', '1hour']:
            time_agg = self.aggregate_signals_by_time(signals_df, window)
            if not time_agg.empty:
                time_rec = self._get_time_based_recommendation(time_agg, window)
                recommendations['recommendations'].append(time_rec)
        
        # Hashtag-based recommendations
        hashtag_agg = self.aggregate_signals_by_hashtag(signals_df)
        if not hashtag_agg.empty:
            hashtag_rec = self._get_hashtag_recommendations(hashtag_agg)
            recommendations['recommendations'].extend(hashtag_rec)
        
        # Risk assessment
        risk_assessment = self._assess_risk(high_conf_signals)
        recommendations['risk_assessment'] = risk_assessment
        
        # Overall recommendation score
        recommendations['overall_score'] = self._calculate_overall_score(
            overall_sentiment, sentiment_consistency, confidence_level, signal_volume
        )
        
        recommendations['status'] = 'success'
        
        logger.info(f"Generated {len(recommendations['recommendations'])} trading recommendations")
        
        return recommendations
    
    def _get_analysis_period(self, signals_df: pd.DataFrame) -> Dict[str, Any]:
        """Get the time period covered by the analysis"""
        if 'signal_timestamp' not in signals_df.columns:
            return {}
        
        timestamps = pd.to_datetime(signals_df['signal_timestamp'])
        return {
            'start_time': timestamps.min(),
            'end_time': timestamps.max(),
            'duration_hours': (timestamps.max() - timestamps.min()).total_seconds() / 3600
        }
    
    def _calculate_sentiment_consistency(self, signals_df: pd.DataFrame) -> float:
        """Calculate how consistent the sentiment direction is"""
        signal_strengths = signals_df['signal_strength']
        
        if len(signal_strengths) == 0:
            return 0.0
        
        positive_signals = (signal_strengths > 0).sum()
        negative_signals = (signal_strengths < 0).sum()
        total_signals = len(signal_strengths)
        
        # Consistency is the proportion of the dominant direction
        consistency = max(positive_signals, negative_signals) / total_signals
        
        return consistency
    
    def _get_direction_recommendation(self, sentiment: float, consistency: float, 
                                    volume: int, confidence: float) -> Dict[str, Any]:
        """Generate overall market direction recommendation"""
        
        # Determine signal strength
        if abs(sentiment) >= self.signal_thresholds['strong']:
            strength = 'strong'
        elif abs(sentiment) >= self.signal_thresholds['moderate']:
            strength = 'moderate'
        elif abs(sentiment) >= self.signal_thresholds['weak']:
            strength = 'weak'
        else:
            strength = 'neutral'
        
        # Determine direction
        if sentiment > 0:
            direction = 'bullish'
        elif sentiment < 0:
            direction = 'bearish'
        else:
            direction = 'neutral'
        
        # Calculate recommendation confidence
        rec_confidence = (confidence * 0.4 + consistency * 0.3 + 
                         min(1.0, volume / 50) * 0.3)
        
        recommendation = {
            'type': 'market_direction',
            'direction': direction,
            'strength': strength,
            'confidence': round(rec_confidence, 3),
            'signal_score': round(sentiment, 3),
            'consistency': round(consistency, 3),
            'supporting_signals': volume,
            'description': f"{strength.title()} {direction} signal with {consistency:.1%} consistency based on {volume} tweets"
        }
        
        return recommendation
    
    def _get_time_based_recommendation(self, time_agg: pd.DataFrame, window: str) -> Dict[str, Any]:
        """Generate time-based recommendations"""
        
        if time_agg.empty:
            return {}
        
        latest_window = time_agg.iloc[-1]
        
        recommendation = {
            'type': f'time_based_{window}',
            'time_window': window,
            'latest_signal': round(latest_window['signal_strength_mean'], 3),
            'signal_intensity': round(latest_window['signal_intensity'], 3),
            'tweet_volume': int(latest_window['signal_strength_count']),
            'consistency': round(latest_window.get('direction_consistency', 0.5), 3),
            'volatility': round(latest_window['signal_volatility'], 3),
            'description': f"Last {window}: {'Bullish' if latest_window['signal_strength_mean'] > 0 else 'Bearish'} sentiment with {int(latest_window['signal_strength_count'])} supporting tweets"
        }
        
        return recommendation
    
    def _get_hashtag_recommendations(self, hashtag_agg: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate hashtag-specific recommendations"""
        
        recommendations = []
        
        # Get top hashtags by momentum
        top_hashtags = hashtag_agg.head(3)
        
        for _, hashtag_data in top_hashtags.iterrows():
            recommendation = {
                'type': 'hashtag_momentum',
                'hashtag': hashtag_data['hashtag'],
                'momentum': round(hashtag_data['hashtag_momentum'], 3),
                'signal_strength': round(hashtag_data['signal_strength_mean'], 3),
                'supporting_tweets': int(hashtag_data['signal_strength_count']),
                'average_confidence': round(hashtag_data['signal_confidence_mean'], 3),
                'description': f"Strong momentum in {hashtag_data['hashtag']} with {int(hashtag_data['signal_strength_count'])} supporting tweets"
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    def _assess_risk(self, signals_df: pd.DataFrame) -> Dict[str, Any]:
        """Assess risk based on signal characteristics"""
        
        if signals_df.empty:
            return {'level': 'unknown', 'factors': []}
        
        risk_factors = []
        risk_score = 0.5  # Base risk
        
        # Signal volatility risk
        signal_volatility = signals_df['signal_strength'].std()
        if signal_volatility > 0.5:
            risk_factors.append("High signal volatility detected")
            risk_score += 0.2
        
        # Consistency risk
        consistency = self._calculate_sentiment_consistency(signals_df)
        if consistency < 0.6:
            risk_factors.append("Low sentiment consistency")
            risk_score += 0.2
        
        # Volume risk
        if len(signals_df) < 10:
            risk_factors.append("Low signal volume")
            risk_score += 0.1
        
        # Confidence risk
        avg_confidence = signals_df['signal_confidence'].mean()
        if avg_confidence < 0.7:
            risk_factors.append("Below average signal confidence")
            risk_score += 0.15
        
        # Determine risk level
        if risk_score >= 0.8:
            risk_level = 'high'
        elif risk_score >= 0.6:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'level': risk_level,
            'score': round(risk_score, 3),
            'factors': risk_factors,
            'description': f"Risk level: {risk_level.upper()} (score: {risk_score:.2f})"
        }
    
    def _calculate_overall_score(self, sentiment: float, consistency: float, 
                               confidence: float, volume: int) -> float:
        """Calculate overall recommendation score"""
        
        # Normalize volume (assume 50 tweets is good volume)
        volume_score = min(1.0, volume / 50)
        
        # Combine factors
        overall_score = (
            abs(sentiment) * 0.3 +     # Signal strength
            consistency * 0.25 +       # Consistency
            confidence * 0.25 +        # Confidence
            volume_score * 0.2         # Volume
        )
        
        return round(overall_score, 3)


def create_signal_aggregator() -> SignalAggregator:
    """Factory function to create signal aggregator"""
    return SignalAggregator()
