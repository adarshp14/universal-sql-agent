#!/usr/bin/env python3
"""
Test script to check if the Streamlit app imports work correctly.
"""

import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules in the same order as the Streamlit app
print("Importing modules...")

try:
    import streamlit as st
    print("✓ Successfully imported streamlit")
except ImportError as e:
    print(f"✗ Error importing streamlit: {e}")

try:
    import pandas as pd
    print("✓ Successfully imported pandas")
except ImportError as e:
    print(f"✗ Error importing pandas: {e}")

try:
    from agents.schema_parser import SchemaParser
    print("✓ Successfully imported SchemaParser")
except ImportError as e:
    print(f"✗ Error importing SchemaParser: {e}")

try:
    from agents.schema_parser import DynamicSchemaParser
    print("✓ Successfully imported DynamicSchemaParser")
except ImportError as e:
    print(f"✗ Error importing DynamicSchemaParser: {e}")

try:
    from agents.query_translator import QueryTranslator
    print("✓ Successfully imported QueryTranslator")
except ImportError as e:
    print(f"✗ Error importing QueryTranslator: {e}")

try:
    from agents.sql_executor import SQLExecutor
    print("✓ Successfully imported SQLExecutor")
except ImportError as e:
    print(f"✗ Error importing SQLExecutor: {e}")

try:
    from agents.responder import Responder
    print("✓ Successfully imported Responder")
except ImportError as e:
    print(f"✗ Error importing Responder: {e}")

try:
    from database.db_setup import setup_sample_database, get_db_engine
    print("✓ Successfully imported setup_sample_database and get_db_engine")
except ImportError as e:
    print(f"✗ Error importing database.db_setup: {e}")

try:
    from sql_agent import AgentPipeline
    print("✓ Successfully imported AgentPipeline")
except ImportError as e:
    print(f"✗ Error importing AgentPipeline: {e}")

print("\nAll import tests completed.")
