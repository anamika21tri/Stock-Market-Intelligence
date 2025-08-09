"""
Configuration management module for Market Intelligence System
"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, Field


class DataCollectionConfig(BaseModel):
    """Configuration for data collection"""
    target_hashtags: list[str] = Field(default_factory=list)
    hashtags: list[str] = Field(default_factory=list)  # Alias for target_hashtags
    target_tweets: int = 2000
    max_tweets_per_hashtag: int = 500
    time_window_hours: int = 24
    hours_back: int = 24  # Alias for time_window_hours
    scraping: Dict[str, Any] = Field(default_factory=dict)
    rate_limiting: Dict[str, Any] = Field(default_factory=dict)
    
    def __init__(self, **data):
        # Handle aliases
        if 'target_hashtags' in data and 'hashtags' not in data:
            data['hashtags'] = data['target_hashtags']
        if 'time_window_hours' in data and 'hours_back' not in data:
            data['hours_back'] = data['time_window_hours']
        super().__init__(**data)


class DataProcessingConfig(BaseModel):
    """Configuration for data processing"""
    storage: Dict[str, Any] = Field(default_factory=dict)
    deduplication: Dict[str, Any] = Field(default_factory=dict)
    cleaning: Dict[str, Any] = Field(default_factory=dict)


class AnalysisConfig(BaseModel):
    """Configuration for analysis"""
    text_processing: Dict[str, Any] = Field(default_factory=dict)
    signal_generation: Dict[str, Any] = Field(default_factory=dict)
    signal_aggregation: Dict[str, Any] = Field(default_factory=dict)
    visualization: Dict[str, Any] = Field(default_factory=dict)


class PerformanceConfig(BaseModel):
    """Configuration for performance settings"""
    memory: Dict[str, Any] = Field(default_factory=dict)
    processing: Dict[str, Any] = Field(default_factory=dict)
    caching: Dict[str, Any] = Field(default_factory=dict)


class LoggingConfig(BaseModel):
    """Configuration for logging"""
    level: str = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    file_rotation: str = "10 MB"
    retention: str = "7 days"


class PathsConfig(BaseModel):
    """Configuration for paths"""
    data_dir: str = "data"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    output_dir: str = "data/output"
    logs_dir: str = "logs"


class Config(BaseModel):
    """Main configuration class"""
    data_collection: DataCollectionConfig = Field(default_factory=DataCollectionConfig)
    data_processing: DataProcessingConfig = Field(default_factory=DataProcessingConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)


class ConfigManager:
    """Configuration manager for loading and managing application settings"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self._config: Optional[Config] = None
    
    def _find_config_file(self) -> str:
        """Find the configuration file in standard locations"""
        possible_paths = [
            "config.yaml",
            "config.yml",
            "../config.yaml",
            os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError("Configuration file not found in standard locations")
    
    def load_config(self) -> Config:
        """Load configuration from YAML file"""
        if self._config is None:
            try:
                with open(self.config_path, 'r', encoding='utf-8') as file:
                    yaml_data = yaml.safe_load(file)
                
                self._config = Config(**yaml_data)
                self._ensure_directories()
                
            except Exception as e:
                raise RuntimeError(f"Failed to load configuration: {e}")
        
        return self._config
    
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        if self._config:
            paths = self._config.paths
            directories = [
                paths.data_dir,
                paths.raw_data_dir,
                paths.processed_data_dir,
                paths.output_dir,
                paths.logs_dir
            ]
            
            for directory in directories:
                Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_config(self) -> Config:
        """Get the loaded configuration"""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def update_config(self, **kwargs) -> None:
        """Update configuration with new values"""
        if self._config is None:
            self.load_config()
        
        # Update config with new values
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
    
    def save_config(self, path: Optional[str] = None) -> None:
        """Save current configuration to file"""
        save_path = path or self.config_path
        
        if self._config:
            config_dict = self._config.dict()
            
            with open(save_path, 'w', encoding='utf-8') as file:
                yaml.dump(config_dict, file, default_flow_style=False, indent=2)


# Global configuration instance
config_manager = ConfigManager()


def get_config() -> Config:
    """Get the global configuration instance"""
    return config_manager.get_config()


def reload_config() -> Config:
    """Reload configuration from file"""
    config_manager._config = None
    return config_manager.load_config()


# For backward compatibility
CONFIG = get_config()
