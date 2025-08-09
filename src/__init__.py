"""
Market Intelligence System - Main Package
Real-time data collection and analysis system for Indian stock market discussions
"""

from . import utils
from . import data_collection
from . import data_processing
from . import analysis
from .main import MarketIntelligenceOrchestrator

__version__ = "1.0.0"
__author__ = "Market Intelligence System"

__all__ = [
    'utils',
    'data_collection', 
    'data_processing',
    'analysis',
    'MarketIntelligenceOrchestrator'
]
