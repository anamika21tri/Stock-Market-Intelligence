#!/usr/bin/env python3
"""
Market Intelligence System - Test Script
Verify all components can be imported and initialized
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_imports():
    """Test all module imports"""
    print("🧪 Testing module imports...")
    
    try:
        # Test utility imports
        from src.utils import get_config, get_logger
        print("✅ Utils module imported successfully")
        
        # Test data collection imports
        from src.data_collection import TwitterScraper, RateLimiter
        print("✅ Data collection module imported successfully")
        
        # Test data processing imports
        from src.data_processing import StorageManager, DataCleaner, Deduplicator
        print("✅ Data processing module imported successfully")
        
        # Test analysis imports
        from src.analysis import TextToSignalConverter, SignalAggregator, MemoryEfficientVisualizer
        print("✅ Analysis module imported successfully")
        
        # Test main orchestrator
        from src.main import MarketIntelligenceOrchestrator
        print("✅ Main orchestrator imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_configuration():
    """Test configuration loading"""
    print("\n🔧 Testing configuration...")
    
    try:
        from src.utils import get_config
        config = get_config()
        
        print(f"✅ Configuration loaded successfully")
        print(f"   - Data collection targets: {len(config.data_collection.hashtags)} hashtags")
        print(f"   - Max tweets per hashtag: {config.data_collection.max_tweets_per_hashtag}")
        print(f"   - Storage format: {config.data_processing.storage.get('format', 'parquet')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_component_initialization():
    """Test component initialization"""
    print("\n🏗️  Testing component initialization...")
    
    try:
        from src.data_collection import TwitterScraper, RateLimiter
        from src.data_processing import StorageManager, DataCleaner, Deduplicator
        from src.analysis import TextToSignalConverter, SignalAggregator, MemoryEfficientVisualizer
        from src.main import MarketIntelligenceOrchestrator
        
        # Initialize components
        scraper = TwitterScraper()
        rate_limiter = RateLimiter()
        storage_manager = StorageManager()
        data_cleaner = DataCleaner()
        deduplicator = Deduplicator()
        text_to_signal = TextToSignalConverter()
        signal_aggregator = SignalAggregator()
        visualizer = MemoryEfficientVisualizer()
        orchestrator = MarketIntelligenceOrchestrator()
        
        print("✅ All components initialized successfully")
        print(f"   - Twitter Scraper: Ready")
        print(f"   - Rate Limiter: Ready")
        print(f"   - Storage Manager: Ready")
        print(f"   - Data Cleaner: Ready")
        print(f"   - Deduplicator: Ready")
        print(f"   - Text-to-Signal Converter: Ready")
        print(f"   - Signal Aggregator: Ready")
        print(f"   - Visualizer: Ready")
        print(f"   - Main Orchestrator: Ready")
        
        return True
        
    except Exception as e:
        print(f"❌ Component initialization error: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Market Intelligence System - Component Test")
    print("="*50)
    
    tests = [
        test_imports,
        test_configuration,
        test_component_initialization
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n📊 Test Results:")
    print("="*30)
    
    if all(results):
        print("🎉 All tests passed! System is ready for execution.")
        print("\n🚀 To run the full analysis, execute:")
        print("   python run_analysis.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("   Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    main()
