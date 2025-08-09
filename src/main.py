"""
Main application orchestrator for Market Intelligence System
Coordinates data collection, processing,         # Convert Tweet objects to dictionaries for processing
        tweet_dicts = []
        for tweet in raw_tweets:
            if hasattr(tweet, '__dict__'):
                tweet_dict = tweet.__dict__.copy()
            else:
                tweet_dict = tweet  # Already a dict
            tweet_dicts.append(tweet_dict)
        
        # Clean the data
        logger.info("Cleaning tweet data")
        cleaned_tweets, cleaning_stats = self.data_cleaner.clean_dataset(tweet_dicts)
        logger.info(f"Cleaning completed. Stats: {cleaning_stats}")
        
        # Convert cleaned tweets back to DataFrame
        cleaned_df = pd.DataFrame(cleaned_tweets)is, and visualization
"""
import asyncio
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import concurrent.futures
import os

from .utils import get_logger, get_config
from .data_collection import TwitterScraper, RateLimiter
from .data_processing import StorageManager, DataCleaner, Deduplicator
from .analysis import TextToSignalConverter, SignalAggregator, MemoryEfficientVisualizer

logger = get_logger(__name__)


class MarketIntelligenceOrchestrator:
    """
    Main orchestrator for the market intelligence system
    Coordinates all components for end-to-end processing
    """
    
    def __init__(self):
        self.config = get_config()
        
        # Initialize components
        self.scraper = TwitterScraper()
        self.storage_manager = StorageManager()
        self.data_cleaner = DataCleaner()
        self.deduplicator = Deduplicator()
        self.text_to_signal = TextToSignalConverter()
        self.signal_aggregator = SignalAggregator()
        self.visualizer = MemoryEfficientVisualizer()
        
        # Processing statistics
        self.stats = {
            'tweets_collected': 0,
            'tweets_processed': 0,
            'signals_generated': 0,
            'execution_time': 0,
            'start_time': None,
            'last_run': None
        }
        
        logger.info("Market Intelligence Orchestrator initialized")
    
    async def collect_market_data(self) -> List[Dict[str, Any]]:
        """
        Collect tweets from Twitter/X with market-relevant hashtags
        
        Returns:
            List of collected tweets
        """
        logger.info("Starting market data collection")
        
        collection_config = self.config.data_collection
        hashtags = collection_config.hashtags
        max_tweets = collection_config.max_tweets_per_hashtag
        hours_back = collection_config.hours_back
        
        all_tweets = []
        
        # Collect tweets for each hashtag
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            
            for hashtag in hashtags:
                future = executor.submit(
                    self.scraper.search_tweets,
                    hashtag,
                    max_tweets,
                    hours_back
                )
                futures.append(future)
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    tweets = future.result()
                    all_tweets.extend(tweets)
                    logger.info(f"Collected {len(tweets)} tweets")
                except Exception as e:
                    logger.error(f"Error collecting tweets: {e}")
        
        self.stats['tweets_collected'] = len(all_tweets)
        logger.info(f"Total tweets collected: {len(all_tweets)}")
        
        return all_tweets
    
    def process_raw_data(self, raw_tweets: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Process raw tweets through cleaning and deduplication
        
        Args:
            raw_tweets: List of raw tweet dictionaries
            
        Returns:
            Processed DataFrame
        """
        logger.info(f"Processing {len(raw_tweets)} raw tweets")
        
        if not raw_tweets:
            logger.warning("No tweets to process")
            return pd.DataFrame()
        
        # Convert Tweet objects to dictionaries for processing
        tweet_dicts = []
        for tweet in raw_tweets:
            if hasattr(tweet, '__dict__'):
                tweet_dict = tweet.__dict__.copy()
            else:
                tweet_dict = tweet  # Already a dict
            tweet_dicts.append(tweet_dict)
        
        # Clean the data
        logger.info("Cleaning tweet data")
        cleaned_tweets, cleaning_stats = self.data_cleaner.clean_dataset(tweet_dicts)
        logger.info(f"Cleaning completed. Stats: {cleaning_stats}")
        
        # Convert cleaned tweets back to DataFrame
        cleaned_df = pd.DataFrame(cleaned_tweets)
        
        # Deduplicate
        logger.info("Deduplicating tweets")
        deduplicated_tweets, dedup_stats = self.deduplicator.comprehensive_deduplication(cleaned_tweets)
        logger.info(f"Deduplication completed. Stats: {dedup_stats}")
        
        # Convert back to DataFrame
        deduplicated_df = pd.DataFrame(deduplicated_tweets)
        
        # Store processed data
        try:
            # Convert to format expected by storage manager
            tweet_dicts_for_storage = []
            for tweet in deduplicated_tweets:
                if isinstance(tweet, dict):
                    tweet_dicts_for_storage.append(tweet)
                else:
                    # Convert to dict if it's an object
                    tweet_dicts_for_storage.append(tweet.__dict__ if hasattr(tweet, '__dict__') else tweet)
            
            storage_path = self.storage_manager.save_tweets(tweet_dicts_for_storage)
            logger.info(f"Stored processed tweets to: {storage_path}")
        except Exception as e:
            logger.error(f"Error storing tweets: {e}")
        
        self.stats['tweets_processed'] = len(deduplicated_df)
        logger.info(f"Processing completed: {len(deduplicated_df)} tweets after cleaning and deduplication")
        
        return deduplicated_df
    
    def generate_trading_signals(self, processed_tweets: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from processed tweets
        
        Args:
            processed_tweets: Cleaned and deduplicated tweets DataFrame
            
        Returns:
            DataFrame with trading signals
        """
        logger.info(f"Generating trading signals from {len(processed_tweets)} tweets")
        
        if processed_tweets.empty:
            logger.warning("No tweets available for signal generation")
            return pd.DataFrame()
        
        # Convert DataFrame to list of dictionaries for processing
        tweets_list = processed_tweets.to_dict('records')
        
        # Generate signals
        signals_df = self.text_to_signal.process_tweets_to_signals(tweets_list)
        
        self.stats['signals_generated'] = len(signals_df)
        logger.info(f"Generated signals for {len(signals_df)} tweets")
        
        return signals_df
    
    def create_trading_recommendations(self, signals_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create actionable trading recommendations from signals
        
        Args:
            signals_df: DataFrame with trading signals
            
        Returns:
            Dictionary with recommendations and analysis
        """
        logger.info("Creating trading recommendations")
        
        if signals_df.empty:
            logger.warning("No signals available for recommendations")
            return {'status': 'no_data', 'recommendations': []}
        
        # Generate recommendations
        recommendations = self.signal_aggregator.create_trading_recommendations(signals_df)
        
        # Create time-based aggregations
        time_aggregations = {}
        for window in ['15min', '1hour']:
            time_agg = self.signal_aggregator.aggregate_signals_by_time(signals_df, window)
            if not time_agg.empty:
                time_aggregations[window] = time_agg
        
        # Create hashtag aggregations
        hashtag_agg = self.signal_aggregator.aggregate_signals_by_hashtag(signals_df)
        
        # Package all results
        analysis_results = {
            'recommendations': recommendations,
            'time_aggregations': time_aggregations,
            'hashtag_analysis': hashtag_agg,
            'signal_summary': {
                'total_signals': len(signals_df),
                'high_confidence_signals': len(signals_df[signals_df['signal_confidence'] >= 0.6]),
                'average_signal_strength': signals_df['signal_strength'].mean(),
                'average_confidence': signals_df['signal_confidence'].mean()
            }
        }
        
        logger.info("Trading recommendations created successfully")
        
        return analysis_results
    
    def create_visualizations(self, signals_df: pd.DataFrame, 
                            analysis_results: Dict[str, Any],
                            output_dir: str = "reports") -> Dict[str, str]:
        """
        Create visualizations and save to files
        
        Args:
            signals_df: DataFrame with trading signals
            analysis_results: Analysis results from create_trading_recommendations
            output_dir: Directory to save visualizations
            
        Returns:
            Dictionary with visualization file paths
        """
        logger.info("Creating visualizations")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        visualization_paths = {}
        
        try:
            # Signal timeline
            timeline_path = os.path.join(output_dir, f"signal_timeline_{timestamp}.png")
            self.visualizer.create_signal_timeline(signals_df, timeline_path)
            visualization_paths['timeline'] = timeline_path
            
            # Sentiment distribution
            sentiment_path = os.path.join(output_dir, f"sentiment_distribution_{timestamp}.png")
            self.visualizer.create_sentiment_distribution(signals_df, sentiment_path)
            visualization_paths['sentiment'] = sentiment_path
            
            # Hashtag analysis
            if not analysis_results['hashtag_analysis'].empty:
                hashtag_path = os.path.join(output_dir, f"hashtag_analysis_{timestamp}.png")
                self.visualizer.create_hashtag_analysis(analysis_results['hashtag_analysis'], hashtag_path)
                visualization_paths['hashtags'] = hashtag_path
            
            # Time aggregation
            if analysis_results['time_aggregations']:
                time_agg_path = os.path.join(output_dir, f"time_aggregation_{timestamp}.png")
                self.visualizer.create_time_aggregation_plot(analysis_results['time_aggregations'], time_agg_path)
                visualization_paths['time_aggregation'] = time_agg_path
            
            # Recommendation summary
            rec_summary_path = os.path.join(output_dir, f"recommendation_summary_{timestamp}.png")
            self.visualizer.create_recommendation_summary(analysis_results['recommendations'], rec_summary_path)
            visualization_paths['recommendations'] = rec_summary_path
            
            logger.info(f"Created {len(visualization_paths)} visualizations in {output_dir}")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
        
        return visualization_paths
    
    def generate_report(self, analysis_results: Dict[str, Any], 
                       visualization_paths: Dict[str, str],
                       output_path: str = None) -> str:
        """
        Generate comprehensive text report
        
        Args:
            analysis_results: Analysis results from recommendations
            visualization_paths: Paths to generated visualizations
            output_path: Path to save report file
            
        Returns:
            Report content as string
        """
        logger.info("Generating comprehensive report")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report_lines = [
            "="*80,
            "MARKET INTELLIGENCE ANALYSIS REPORT",
            "="*80,
            f"Generated: {timestamp}",
            f"Analysis Period: {self.stats.get('start_time')} to {timestamp}",
            "",
            "EXECUTIVE SUMMARY",
            "-"*40,
            f"• Total Tweets Collected: {self.stats['tweets_collected']}",
            f"• Tweets Processed: {self.stats['tweets_processed']}",
            f"• Trading Signals Generated: {self.stats['signals_generated']}",
            f"• Processing Time: {self.stats.get('execution_time', 0):.2f} seconds",
            ""
        ]
        
        # Add signal summary
        signal_summary = analysis_results.get('signal_summary', {})
        if signal_summary:
            report_lines.extend([
                "SIGNAL ANALYSIS SUMMARY",
                "-"*40,
                f"• Total Signals: {signal_summary.get('total_signals', 0)}",
                f"• High Confidence Signals: {signal_summary.get('high_confidence_signals', 0)}",
                f"• Average Signal Strength: {signal_summary.get('average_signal_strength', 0):.3f}",
                f"• Average Confidence: {signal_summary.get('average_confidence', 0):.3f}",
                ""
            ])
        
        # Add recommendations
        recommendations = analysis_results.get('recommendations', {})
        if recommendations.get('status') == 'success':
            report_lines.extend([
                "TRADING RECOMMENDATIONS",
                "-"*40,
                f"• Overall Score: {recommendations.get('overall_score', 0):.3f}",
                ""
            ])
            
            for i, rec in enumerate(recommendations.get('recommendations', []), 1):
                report_lines.append(f"{i}. {rec.get('type', 'Unknown').upper()}")
                report_lines.append(f"   Description: {rec.get('description', 'N/A')}")
                if 'confidence' in rec:
                    report_lines.append(f"   Confidence: {rec['confidence']:.3f}")
                report_lines.append("")
            
            # Risk assessment
            risk = recommendations.get('risk_assessment', {})
            if risk:
                report_lines.extend([
                    "RISK ASSESSMENT",
                    "-"*40,
                    f"• Risk Level: {risk.get('level', 'Unknown').upper()}",
                    f"• Risk Score: {risk.get('score', 0):.3f}",
                    f"• Description: {risk.get('description', 'N/A')}",
                    ""
                ])
                
                if risk.get('factors'):
                    report_lines.append("Risk Factors:")
                    for factor in risk['factors']:
                        report_lines.append(f"   - {factor}")
                    report_lines.append("")
        
        # Add visualizations
        if visualization_paths:
            report_lines.extend([
                "GENERATED VISUALIZATIONS",
                "-"*40
            ])
            for viz_type, path in visualization_paths.items():
                report_lines.append(f"• {viz_type.title()}: {path}")
            report_lines.append("")
        
        # Technical details
        report_lines.extend([
            "TECHNICAL DETAILS",
            "-"*40,
            f"• Data Collection Method: Twitter/X Scraping",
            f"• Text Processing: TF-IDF Vectorization + Sentiment Analysis",
            f"• Signal Generation: Multi-factor Weighted Algorithm",
            f"• Aggregation Windows: 15min, 1hour",
            f"• Confidence Threshold: {self.config.analysis.signal_aggregation.get('min_confidence', 0.6)}",
            "",
            "="*80,
            "END OF REPORT",
            "="*80
        ])
        
        report_content = "\n".join(report_lines)
        
        # Save to file if path provided
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                logger.info(f"Report saved to: {output_path}")
            except Exception as e:
                logger.error(f"Error saving report: {e}")
        
        return report_content
    
    async def run_full_analysis(self, output_dir: str = "reports") -> Dict[str, Any]:
        """
        Run complete market intelligence analysis pipeline
        
        Args:
            output_dir: Directory to save outputs
            
        Returns:
            Complete analysis results
        """
        start_time = datetime.now()
        self.stats['start_time'] = start_time
        logger.info("Starting full market intelligence analysis")
        
        try:
            # Step 1: Collect data
            logger.info("Step 1: Collecting market data")
            raw_tweets = await self.collect_market_data()
            
            if not raw_tweets:
                logger.error("No tweets collected - aborting analysis")
                return {'status': 'error', 'message': 'No data collected'}
            
            # Step 2: Process data
            logger.info("Step 2: Processing raw data")
            processed_tweets = self.process_raw_data(raw_tweets)
            
            if processed_tweets.empty:
                logger.error("No tweets after processing - aborting analysis")
                return {'status': 'error', 'message': 'No data after processing'}
            
            # Step 3: Generate signals
            logger.info("Step 3: Generating trading signals")
            signals_df = self.generate_trading_signals(processed_tweets)
            
            if signals_df.empty:
                logger.error("No signals generated - aborting analysis")
                return {'status': 'error', 'message': 'No signals generated'}
            
            # Step 4: Create recommendations
            logger.info("Step 4: Creating trading recommendations")
            analysis_results = self.create_trading_recommendations(signals_df)
            
            # Step 5: Create visualizations
            logger.info("Step 5: Creating visualizations")
            visualization_paths = self.create_visualizations(signals_df, analysis_results, output_dir)
            
            # Step 6: Generate report
            logger.info("Step 6: Generating report")
            report_path = os.path.join(output_dir, f"analysis_report_{start_time.strftime('%Y%m%d_%H%M%S')}.txt")
            report_content = self.generate_report(analysis_results, visualization_paths, report_path)
            
            # Calculate execution time
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            self.stats['execution_time'] = execution_time
            self.stats['last_run'] = end_time
            
            # Package final results
            final_results = {
                'status': 'success',
                'execution_time': execution_time,
                'statistics': self.stats,
                'analysis_results': analysis_results,
                'visualization_paths': visualization_paths,
                'report_path': report_path,
                'report_content': report_content,
                'processed_data': {
                    'tweets_collected': len(raw_tweets),
                    'tweets_processed': len(processed_tweets),
                    'signals_generated': len(signals_df)
                }
            }
            
            logger.info(f"Full analysis completed successfully in {execution_time:.2f} seconds")
            logger.info(f"Results saved to: {output_dir}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error during full analysis: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and statistics"""
        return {
            'status': 'ready',
            'statistics': self.stats,
            'last_run': self.stats.get('last_run'),
            'components': {
                'scraper': 'active',
                'storage': 'active',
                'processor': 'active',
                'analyzer': 'active',
                'visualizer': 'active'
            }
        }


async def main():
    """Main entry point for the application"""
    logger.info("Market Intelligence System - Starting")
    
    # Create orchestrator
    orchestrator = MarketIntelligenceOrchestrator()
    
    # Run full analysis
    results = await orchestrator.run_full_analysis()
    
    if results['status'] == 'success':
        logger.info("Analysis completed successfully!")
        print("\n" + "="*60)
        print("MARKET INTELLIGENCE ANALYSIS COMPLETED")
        print("="*60)
        print(f"Execution Time: {results['execution_time']:.2f} seconds")
        print(f"Tweets Collected: {results['processed_data']['tweets_collected']}")
        print(f"Tweets Processed: {results['processed_data']['tweets_processed']}")
        print(f"Signals Generated: {results['processed_data']['signals_generated']}")
        print(f"Report Location: {results['report_path']}")
        print("="*60)
    else:
        logger.error(f"Analysis failed: {results.get('message', 'Unknown error')}")
        print(f"\nAnalysis Failed: {results.get('message', 'Unknown error')}")


if __name__ == "__main__":
    asyncio.run(main())
