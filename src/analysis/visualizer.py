"""
Memory-efficient visualization module for streaming plots and real-time dashboards
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import io
import base64
from collections import deque
import threading
import time

from ..utils import get_logger, get_config

logger = get_logger(__name__)

# Set plotting parameters for memory efficiency
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.max_open_warning'] = 20
plt.rcParams['figure.dpi'] = 100


class MemoryEfficientVisualizer:
    """
    Memory-efficient visualization system for real-time market intelligence
    """
    
    def __init__(self, max_data_points: int = 1000):
        self.config = get_config().analysis.visualization
        self.max_data_points = max_data_points
        
        # Streaming data buffers
        self.signal_buffer = deque(maxlen=max_data_points)
        self.sentiment_buffer = deque(maxlen=max_data_points)
        self.volume_buffer = deque(maxlen=max_data_points)
        self.timestamp_buffer = deque(maxlen=max_data_points)
        
        # Color schemes
        self.colors = {
            'bullish': '#2E8B57',      # Sea Green
            'bearish': '#DC143C',      # Crimson
            'neutral': '#4682B4',      # Steel Blue
            'volume': '#FFD700',       # Gold
            'confidence': '#9370DB'    # Medium Purple
        }
        
        # Figure management
        self.active_figures = {}
        self.max_figures = 5
        
        logger.info(f"Memory-efficient visualizer initialized with {max_data_points} max data points")
    
    def add_data_point(self, signal_strength: float, sentiment: float, 
                       volume: int, timestamp: datetime, confidence: float = 1.0):
        """Add a new data point to streaming buffers"""
        
        self.signal_buffer.append(signal_strength)
        self.sentiment_buffer.append(sentiment)
        self.volume_buffer.append(volume)
        self.timestamp_buffer.append(timestamp)
        
    def create_signal_timeline(self, signals_df: pd.DataFrame, 
                              save_path: Optional[str] = None) -> str:
        """
        Create timeline visualization of signal strength over time
        
        Returns:
            Base64 encoded image string or file path
        """
        logger.info("Creating signal timeline visualization")
        
        # Close any existing figures to save memory
        self._cleanup_figures()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        if signals_df.empty or 'signal_timestamp' not in signals_df.columns:
            # Create empty plot
            ax1.text(0.5, 0.5, 'No Data Available', ha='center', va='center', transform=ax1.transAxes)
            ax2.text(0.5, 0.5, 'No Data Available', ha='center', va='center', transform=ax2.transAxes)
        else:
            # Prepare data
            df = signals_df.copy()
            df['signal_timestamp'] = pd.to_datetime(df['signal_timestamp'])
            df = df.sort_values('signal_timestamp')
            
            # Resample to reduce data points for memory efficiency
            if len(df) > self.max_data_points:
                df = df.set_index('signal_timestamp').resample('5min').agg({
                    'signal_strength': 'mean',
                    'signal_confidence': 'mean',
                    'sentiment_score': 'mean'
                }).dropna().reset_index()
            
            # Signal strength timeline
            colors = [self.colors['bullish'] if x > 0 else self.colors['bearish'] 
                     for x in df['signal_strength']]
            
            ax1.scatter(df['signal_timestamp'], df['signal_strength'], 
                       c=colors, alpha=0.6, s=50)
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax1.set_ylabel('Signal Strength')
            ax1.set_title('Market Signal Timeline', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Add trend line
            if len(df) > 1:
                z = np.polyfit(range(len(df)), df['signal_strength'], 1)
                p = np.poly1d(z)
                ax1.plot(df['signal_timestamp'], p(range(len(df))), 
                        color='red', linestyle='--', alpha=0.8, linewidth=2)
            
            # Confidence timeline
            ax2.fill_between(df['signal_timestamp'], df['signal_confidence'], 
                           alpha=0.4, color=self.colors['confidence'])
            ax2.plot(df['signal_timestamp'], df['signal_confidence'], 
                    color=self.colors['confidence'], linewidth=2)
            ax2.set_ylabel('Confidence Level')
            ax2.set_xlabel('Time')
            ax2.set_title('Signal Confidence Over Time')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save or return base64
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            return save_path
        else:
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close(fig)
            return img_base64
    
    def create_sentiment_distribution(self, signals_df: pd.DataFrame, 
                                    save_path: Optional[str] = None) -> str:
        """
        Create sentiment distribution visualization
        
        Returns:
            Base64 encoded image string or file path
        """
        logger.info("Creating sentiment distribution visualization")
        
        self._cleanup_figures()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        if signals_df.empty:
            for ax in [ax1, ax2, ax3, ax4]:
                ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', transform=ax.transAxes)
        else:
            # Sentiment histogram
            sentiment_data = signals_df['sentiment_score'].dropna()
            ax1.hist(sentiment_data, bins=30, alpha=0.7, color=self.colors['neutral'], edgecolor='black')
            ax1.axvline(sentiment_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {sentiment_data.mean():.3f}')
            ax1.set_xlabel('Sentiment Score')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Sentiment Score Distribution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Signal strength distribution
            signal_data = signals_df['signal_strength'].dropna()
            ax2.hist(signal_data, bins=30, alpha=0.7, color=self.colors['volume'], edgecolor='black')
            ax2.axvline(signal_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {signal_data.mean():.3f}')
            ax2.set_xlabel('Signal Strength')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Signal Strength Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Signal categories pie chart
            if 'signal_category' in signals_df.columns:
                category_counts = signals_df['signal_category'].value_counts()
                colors_pie = [self.colors['bearish'], self.colors['neutral'], self.colors['bullish']][:len(category_counts)]
                ax3.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%', 
                       colors=colors_pie, startangle=90)
                ax3.set_title('Signal Category Distribution')
            else:
                ax3.text(0.5, 0.5, 'No Category Data', ha='center', va='center', transform=ax3.transAxes)
            
            # Confidence vs Signal Strength scatter
            if len(signals_df) > 0:
                scatter = ax4.scatter(signals_df['signal_confidence'], signals_df['signal_strength'], 
                                    alpha=0.6, c=signals_df['sentiment_score'], cmap='RdYlGn', s=50)
                ax4.set_xlabel('Signal Confidence')
                ax4.set_ylabel('Signal Strength')
                ax4.set_title('Confidence vs Signal Strength')
                ax4.grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=ax4, label='Sentiment Score')
            else:
                ax4.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        
        # Save or return base64
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            return save_path
        else:
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close(fig)
            return img_base64
    
    def create_hashtag_analysis(self, hashtag_agg: pd.DataFrame, 
                               save_path: Optional[str] = None) -> str:
        """
        Create hashtag momentum analysis visualization
        
        Returns:
            Base64 encoded image string or file path
        """
        logger.info("Creating hashtag analysis visualization")
        
        self._cleanup_figures()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        if hashtag_agg.empty:
            for ax in [ax1, ax2]:
                ax.text(0.5, 0.5, 'No Hashtag Data Available', ha='center', va='center', transform=ax.transAxes)
        else:
            # Top hashtags by momentum
            top_hashtags = hashtag_agg.head(10)
            
            colors = [self.colors['bullish'] if x > 0 else self.colors['bearish'] 
                     for x in top_hashtags['hashtag_momentum']]
            
            bars = ax1.barh(range(len(top_hashtags)), top_hashtags['hashtag_momentum'], 
                           color=colors, alpha=0.7)
            ax1.set_yticks(range(len(top_hashtags)))
            ax1.set_yticklabels(top_hashtags['hashtag'], fontsize=10)
            ax1.set_xlabel('Hashtag Momentum')
            ax1.set_title('Top Hashtags by Momentum', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='x')
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax1.text(width + 0.01 if width >= 0 else width - 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{width:.2f}', ha='left' if width >= 0 else 'right', va='center', fontsize=9)
            
            # Hashtag tweet volume vs signal strength
            if len(top_hashtags) > 0:
                scatter = ax2.scatter(top_hashtags['signal_strength_count'], 
                                    top_hashtags['signal_strength_mean'],
                                    s=np.abs(top_hashtags['hashtag_momentum']) * 100,
                                    c=top_hashtags['signal_strength_mean'],
                                    cmap='RdYlGn', alpha=0.7)
                
                # Add hashtag labels
                for i, row in top_hashtags.iterrows():
                    ax2.annotate(row['hashtag'], 
                               (row['signal_strength_count'], row['signal_strength_mean']),
                               xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
                
                ax2.set_xlabel('Number of Supporting Tweets')
                ax2.set_ylabel('Average Signal Strength')
                ax2.set_title('Hashtag Volume vs Signal Strength')
                ax2.grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=ax2, label='Signal Strength')
            else:
                ax2.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        
        # Save or return base64
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            return save_path
        else:
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close(fig)
            return img_base64
    
    def create_time_aggregation_plot(self, time_agg_data: Dict[str, pd.DataFrame], 
                                   save_path: Optional[str] = None) -> str:
        """
        Create time aggregation visualization for multiple time windows
        
        Returns:
            Base64 encoded image string or file path
        """
        logger.info("Creating time aggregation visualization")
        
        self._cleanup_figures()
        
        n_windows = len(time_agg_data)
        if n_windows == 0:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            ax.text(0.5, 0.5, 'No Time Aggregation Data Available', 
                   ha='center', va='center', transform=ax.transAxes)
        else:
            fig, axes = plt.subplots(n_windows, 1, figsize=(14, 4 * n_windows), sharex=True)
            if n_windows == 1:
                axes = [axes]
            
            for i, (window, df) in enumerate(time_agg_data.items()):
                ax = axes[i]
                
                if df.empty:
                    ax.text(0.5, 0.5, f'No Data for {window}', ha='center', va='center', transform=ax.transAxes)
                    continue
                
                # Plot signal intensity over time
                time_col = 'time_window' if 'time_window' in df.columns else df.index
                
                # Signal strength
                ax.plot(time_col, df['signal_strength_mean'], 
                       color=self.colors['neutral'], linewidth=2, label='Signal Strength')
                ax.fill_between(time_col, df['signal_strength_mean'], alpha=0.3, color=self.colors['neutral'])
                
                # Add confidence indicators as points
                if 'signal_confidence_mean' in df.columns:
                    ax2 = ax.twinx()
                    ax2.scatter(time_col, df['signal_confidence_mean'], 
                              color=self.colors['confidence'], alpha=0.7, s=30, label='Confidence')
                    ax2.set_ylabel('Confidence', color=self.colors['confidence'])
                    ax2.set_ylim(0, 1)
                
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax.set_ylabel('Signal Strength')
                ax.set_title(f'Signal Trends - {window} Windows', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper left')
                
                if 'signal_confidence_mean' in df.columns:
                    ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        # Save or return base64
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            return save_path
        else:
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close(fig)
            return img_base64
    
    def create_recommendation_summary(self, recommendations: Dict[str, Any], 
                                    save_path: Optional[str] = None) -> str:
        """
        Create visual summary of trading recommendations
        
        Returns:
            Base64 encoded image string or file path
        """
        logger.info("Creating recommendation summary visualization")
        
        self._cleanup_figures()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        if not recommendations or recommendations.get('status') != 'success':
            for ax in [ax1, ax2, ax3, ax4]:
                ax.text(0.5, 0.5, 'No Recommendations Available', ha='center', va='center', transform=ax.transAxes)
        else:
            # Overall score gauge
            overall_score = recommendations.get('overall_score', 0)
            self._create_gauge_chart(ax1, overall_score, 'Overall Signal Score')
            
            # Recommendation types pie chart
            rec_types = [rec.get('type', 'unknown') for rec in recommendations.get('recommendations', [])]
            if rec_types:
                type_counts = pd.Series(rec_types).value_counts()
                ax2.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
                ax2.set_title('Recommendation Types Distribution')
            else:
                ax2.text(0.5, 0.5, 'No Recommendations', ha='center', va='center', transform=ax2.transAxes)
            
            # Signal confidence levels
            confidences = [rec.get('confidence', 0) for rec in recommendations.get('recommendations', []) 
                          if 'confidence' in rec]
            if confidences:
                ax3.hist(confidences, bins=10, alpha=0.7, color=self.colors['confidence'], edgecolor='black')
                ax3.axvline(np.mean(confidences), color='red', linestyle='--', linewidth=2, 
                           label=f'Mean: {np.mean(confidences):.2f}')
                ax3.set_xlabel('Confidence Level')
                ax3.set_ylabel('Frequency')
                ax3.set_title('Recommendation Confidence Distribution')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'No Confidence Data', ha='center', va='center', transform=ax3.transAxes)
            
            # Risk assessment
            risk_data = recommendations.get('risk_assessment', {})
            if risk_data:
                risk_level = risk_data.get('level', 'unknown')
                risk_score = risk_data.get('score', 0.5)
                
                # Risk level visualization
                levels = ['low', 'medium', 'high']
                colors_risk = [self.colors['bullish'], self.colors['volume'], self.colors['bearish']]
                scores = [0.4, 0.6, 0.8]  # Threshold scores for each level
                
                bars = ax4.bar(levels, scores, color=colors_risk, alpha=0.7)
                
                # Highlight current risk level
                if risk_level in levels:
                    idx = levels.index(risk_level)
                    bars[idx].set_alpha(1.0)
                    bars[idx].set_edgecolor('black')
                    bars[idx].set_linewidth(3)
                
                ax4.axhline(y=risk_score, color='red', linestyle='-', linewidth=3, 
                           label=f'Current Risk: {risk_score:.2f}')
                ax4.set_ylabel('Risk Score')
                ax4.set_title('Risk Assessment')
                ax4.legend()
                ax4.grid(True, alpha=0.3, axis='y')
            else:
                ax4.text(0.5, 0.5, 'No Risk Data', ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        
        # Save or return base64
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            return save_path
        else:
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close(fig)
            return img_base64
    
    def _create_gauge_chart(self, ax, value: float, title: str):
        """Create a gauge chart for displaying scores"""
        
        # Gauge parameters
        theta = np.linspace(0, np.pi, 100)
        
        # Background arc
        ax.plot(np.cos(theta), np.sin(theta), color='lightgray', linewidth=20, alpha=0.3)
        
        # Value arc
        value_theta = np.linspace(0, np.pi * value, int(100 * value))
        if len(value_theta) > 0:
            color = self.colors['bullish'] if value > 0.6 else self.colors['volume'] if value > 0.4 else self.colors['bearish']
            ax.plot(np.cos(value_theta), np.sin(value_theta), color=color, linewidth=20)
        
        # Needle
        needle_angle = np.pi * value
        ax.arrow(0, 0, 0.8 * np.cos(needle_angle), 0.8 * np.sin(needle_angle), 
                head_width=0.05, head_length=0.05, fc='black', ec='black')
        
        # Labels
        ax.text(0, -0.3, f'{value:.2f}', ha='center', va='center', fontsize=16, fontweight='bold')
        ax.text(0, -0.5, title, ha='center', va='center', fontsize=12)
        
        # Scale labels
        for i, label_val in enumerate([0, 0.25, 0.5, 0.75, 1.0]):
            angle = np.pi * label_val
            x, y = 1.1 * np.cos(angle), 1.1 * np.sin(angle)
            ax.text(x, y, f'{label_val:.2f}', ha='center', va='center', fontsize=10)
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-0.7, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _cleanup_figures(self):
        """Clean up matplotlib figures to save memory"""
        
        # Close all figures if we have too many
        if len(plt.get_fignums()) >= self.max_figures:
            plt.close('all')
            logger.debug("Closed all figures for memory cleanup")
    
    def create_streaming_dashboard(self, update_interval: int = 30) -> None:
        """
        Create a streaming dashboard that updates in real-time
        Note: This is a template for future implementation
        """
        logger.info(f"Streaming dashboard would update every {update_interval} seconds")
        # This would require a web framework like Dash or Streamlit
        # for actual implementation
        pass


def create_visualizer(max_data_points: int = 1000) -> MemoryEfficientVisualizer:
    """Factory function to create memory-efficient visualizer"""
    return MemoryEfficientVisualizer(max_data_points)
