#!/usr/bin/env python3
"""
Debug script to check treatment database structure
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app import treatment_db
    
    print("Testing treatment database...")
    treatments = treatment_db.get_treatment_info('Aphids')
    print(f"Treatment result type: {type(treatments)}")
    print(f"Treatment result: {treatments}")
    
    if isinstance(treatments, dict):
        print(f"Keys in treatment result: {list(treatments.keys())}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
