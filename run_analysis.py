#!/usr/bin/env python3
"""
Market Intelligence System - Run Script
Execute the complete market intelligence analysis pipeline
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.main import main

if __name__ == "__main__":
    print("🚀 Market Intelligence System")
    print("📊 Real-time Indian Stock Market Analysis")
    print("="*50)
    
    try:
        # Run the main analysis
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running analysis: {e}")
    finally:
        print("\n👋 Market Intelligence System - Session End")
