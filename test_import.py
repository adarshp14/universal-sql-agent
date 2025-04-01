#!/usr/bin/env python3
"""
Test script to check if DynamicSchemaParser can be imported correctly.
"""

import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from agents.schema_parser import DynamicSchemaParser
    print("Successfully imported DynamicSchemaParser")
except ImportError as e:
    print(f"Error importing DynamicSchemaParser: {e}")

try:
    from agents.schema_parser import SchemaParser
    print("Successfully imported SchemaParser")
except ImportError as e:
    print(f"Error importing SchemaParser: {e}")
