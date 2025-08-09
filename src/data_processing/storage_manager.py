"""
Storage manager for efficient data storage and retrieval using Parquet format
"""
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import hashlib
import json

from ..utils import get_logger, get_config

logger = get_logger(__name__)


class StorageManager:
    """
    Manages data storage in Parquet format with partitioning and compression
    """
    
    def __init__(self):
        self.config = get_config()
        self.paths = self.config.paths
        self.storage_config = self.config.data_processing.storage
        
        # Create storage directories
        for path in [self.paths.raw_data_dir, self.paths.processed_data_dir, self.paths.output_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)
        
        logger.info("Storage manager initialized")
    
    def _get_partition_path(self, base_path: str, date: datetime) -> str:
        """Generate partition path based on date"""
        year = date.strftime('%Y')
        month = date.strftime('%m')
        day = date.strftime('%d')
        return os.path.join(base_path, f"year={year}", f"month={month}", f"day={day}")
    
    def _prepare_dataframe_for_storage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for efficient Parquet storage"""
        # Optimize data types
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to categorical if low cardinality
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.5:
                    df[col] = df[col].astype('category')
        
        # Ensure timestamp columns are properly formatted
        timestamp_cols = ['timestamp', 'created_at', 'collected_at']
        for col in timestamp_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        return df
    
    def save_tweets(self, tweets: List[Dict[str, Any]], 
                   partition_by_date: bool = True,
                   filename_prefix: str = "tweets") -> str:
        """
        Save tweets to Parquet format with optional partitioning
        
        Args:
            tweets: List of tweet dictionaries
            partition_by_date: Whether to partition by date
            filename_prefix: Prefix for the filename
            
        Returns:
            Path to saved file(s)
        """
        if not tweets:
            logger.warning("No tweets to save")
            return ""
        
        df = pd.DataFrame(tweets)
        df = self._prepare_dataframe_for_storage(df)
        
        # Add collection metadata
        df['collected_at'] = datetime.now()
        df['data_version'] = '1.0'
        
        if partition_by_date and 'timestamp' in df.columns:
            # Save with partitioning
            return self._save_partitioned_data(df, filename_prefix, 'raw')
        else:
            # Save as single file
            return self._save_single_file(df, filename_prefix, 'raw')
    
    def _save_partitioned_data(self, df: pd.DataFrame, 
                              filename_prefix: str, 
                              data_type: str) -> str:
        """Save data with date partitioning"""
        base_path = getattr(self.paths, f"{data_type}_data_dir")
        
        # Group by date for partitioning
        df['date'] = df['timestamp'].dt.date
        
        saved_paths = []
        
        for date, group in df.groupby('date'):
            partition_path = self._get_partition_path(base_path, pd.Timestamp(date))
            Path(partition_path).mkdir(parents=True, exist_ok=True)
            
            timestamp_str = datetime.now().strftime('%H%M%S')
            filename = f"{filename_prefix}_{timestamp_str}.parquet"
            file_path = os.path.join(partition_path, filename)
            
            # Remove the temporary date column
            group_clean = group.drop('date', axis=1)
            
            # Save with compression
            group_clean.to_parquet(
                file_path,
                compression=self.storage_config.get('compression', 'snappy'),
                index=False,
                engine='pyarrow'
            )
            
            saved_paths.append(file_path)
            logger.info(f"Saved {len(group_clean)} records to {file_path}")
        
        return base_path
    
    def _save_single_file(self, df: pd.DataFrame, 
                         filename_prefix: str, 
                         data_type: str) -> str:
        """Save data as single file"""
        base_path = getattr(self.paths, f"{data_type}_data_dir")
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{filename_prefix}_{timestamp_str}.parquet"
        file_path = os.path.join(base_path, filename)
        
        df.to_parquet(
            file_path,
            compression=self.storage_config.get('compression', 'snappy'),
            index=False,
            engine='pyarrow'
        )
        
        logger.info(f"Saved {len(df)} records to {file_path}")
        return file_path
    
    def load_tweets(self, 
                   date_range: Optional[tuple] = None,
                   hashtags: Optional[List[str]] = None,
                   data_type: str = 'raw') -> pd.DataFrame:
        """
        Load tweets from storage with optional filtering
        
        Args:
            date_range: Tuple of (start_date, end_date)
            hashtags: List of hashtags to filter by
            data_type: Type of data to load ('raw', 'processed', 'output')
            
        Returns:
            DataFrame containing filtered tweets
        """
        base_path = getattr(self.paths, f"{data_type}_data_dir")
        
        if not os.path.exists(base_path):
            logger.warning(f"Data directory does not exist: {base_path}")
            return pd.DataFrame()
        
        # Find all parquet files
        parquet_files = []
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.endswith('.parquet'):
                    parquet_files.append(os.path.join(root, file))
        
        if not parquet_files:
            logger.warning(f"No parquet files found in {base_path}")
            return pd.DataFrame()
        
        # Load and combine all files
        dfs = []
        for file_path in parquet_files:
            try:
                df = pd.read_parquet(file_path)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
        
        if not dfs:
            return pd.DataFrame()
        
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Apply filters
        if date_range:
            start_date, end_date = date_range
            combined_df = combined_df[
                (combined_df['timestamp'] >= start_date) & 
                (combined_df['timestamp'] <= end_date)
            ]
        
        if hashtags:
            # Filter by hashtags (assuming hashtags are stored as lists)
            mask = combined_df['hashtags'].apply(
                lambda x: any(tag.lower() in [h.lower() for h in hashtags] for tag in x) 
                if isinstance(x, list) else False
            )
            combined_df = combined_df[mask]
        
        logger.info(f"Loaded {len(combined_df)} tweets from storage")
        return combined_df
    
    def save_processed_data(self, df: pd.DataFrame, 
                           data_name: str,
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save processed analysis data"""
        df = self._prepare_dataframe_for_storage(df)
        
        # Add metadata
        if metadata:
            for key, value in metadata.items():
                df[f"meta_{key}"] = value
        
        return self._save_single_file(df, data_name, 'processed')
    
    def save_output_data(self, data: Union[pd.DataFrame, Dict[str, Any]], 
                        output_name: str) -> str:
        """Save final output data (signals, visualizations, etc.)"""
        if isinstance(data, dict):
            # Save as JSON for metadata/config outputs
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{output_name}_{timestamp_str}.json"
            file_path = os.path.join(self.paths.output_dir, filename)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Saved output data to {file_path}")
            return file_path
        
        elif isinstance(data, pd.DataFrame):
            return self._save_single_file(data, output_name, 'output')
        
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        stats = {}
        
        for data_type in ['raw', 'processed', 'output']:
            path = getattr(self.paths, f"{data_type}_data_dir")
            
            if os.path.exists(path):
                file_count = 0
                total_size = 0
                
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.endswith('.parquet') or file.endswith('.json'):
                            file_path = os.path.join(root, file)
                            file_count += 1
                            total_size += os.path.getsize(file_path)
                
                stats[data_type] = {
                    'file_count': file_count,
                    'total_size_mb': round(total_size / (1024 * 1024), 2),
                    'path': path
                }
            else:
                stats[data_type] = {
                    'file_count': 0,
                    'total_size_mb': 0,
                    'path': path
                }
        
        return stats
    
    def cleanup_old_data(self, days_to_keep: int = 7) -> None:
        """Clean up old data files"""
        cutoff_date = datetime.now() - pd.Timedelta(days=days_to_keep)
        
        for data_type in ['raw', 'processed', 'output']:
            path = getattr(self.paths, f"{data_type}_data_dir")
            
            if not os.path.exists(path):
                continue
            
            deleted_count = 0
            
            for root, dirs, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    if file_mtime < cutoff_date:
                        try:
                            os.remove(file_path)
                            deleted_count += 1
                        except Exception as e:
                            logger.error(f"Error deleting {file_path}: {e}")
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old files from {data_type} directory")


def create_storage_manager() -> StorageManager:
    """Factory function to create storage manager"""
    return StorageManager()
