"""
Dynamic Schema Parser agent for the Universal SQL Conversational Agent.
Responsible for automatically detecting and extracting database schema information.
"""

import re
import json
from sqlalchemy import inspect, MetaData, Table, Column, ForeignKey
from sqlalchemy.sql import select
import pandas as pd
import streamlit as st

class DynamicSchemaParser:
    """
    Agent responsible for dynamically detecting database schema and understanding database structure.
    Automatically extracts table names, column names, data types, and relationships from the connected database.
    Provides intelligent schema inference for databases without explicit relationship definitions.
    """
    
    def __init__(self, db_engine):
        """
        Initialize the DynamicSchemaParser agent with a database engine.
        
        Args:
            db_engine: SQLAlchemy engine for database access
        """
        self.db_engine = db_engine
        self.dialect_name = db_engine.dialect.name if db_engine else 'unknown'
        self.schema_info = None
    
    def _detect_schema(self):
        """
        Dynamically detect the database schema using SQLAlchemy's inspection capabilities.
        Caches results in session state for performance and persistence across interactions.
        
        Returns:
            dict: Detected schema information with tables, columns, keys, and relationships
        """
        try:
            # Check if schema needs to be refreshed (if tables have been added/removed)
            should_refresh = False
            
            # Use session state if available to avoid repeated schema detection
            if 'schema_info' in st.session_state and st.session_state.schema_info:
                # Verify if the schema is still valid by checking if tables match
                inspector = inspect(self.db_engine)
                current_tables = set(inspector.get_table_names())
                cached_tables = set(table_name for table_name in st.session_state.schema_info.keys() 
                                   if table_name != 'database_type' and table_name != 'data_patterns')
                
                # If tables have changed, we need to refresh the schema
                if current_tables != cached_tables:
                    should_refresh = True
                    print("Schema refresh needed: tables have changed")
                else:
                    # Return the cached schema
                    return st.session_state.schema_info
            else:
                should_refresh = True
            
            # Detect schema if needed
            if should_refresh:
                # Create a schema information dictionary
                schema_info = {}
                
                # Add database type information
                schema_info['database_type'] = self.dialect_name
                
                # Use SQLAlchemy's inspection to get table information
                inspector = inspect(self.db_engine)
                
                # Get all table names
                table_names = inspector.get_table_names()
                
                # Process each table
                for table_name in table_names:
                    # Create table entry in schema info
                    schema_info[table_name] = {
                        'columns': {},
                        'primary_key': [],
                        'foreign_keys': [],
                        'indexes': [],
                        'data_patterns': {}
                    }
                    
                    # Get column information
                    columns = inspector.get_columns(table_name)
                    for column in columns:
                        col_name = column['name']
                        col_type = str(column['type'])
                        schema_info[table_name]['columns'][col_name] = {
                            'type': col_type,
                            'nullable': column.get('nullable', True),
                            'default': str(column.get('default', 'NULL')),
                            'autoincrement': column.get('autoincrement', False)
                        }
                    
                    # Get primary key information
                    pk = inspector.get_pk_constraint(table_name)
                    if pk and 'constrained_columns' in pk:
                        schema_info[table_name]['primary_key'] = pk['constrained_columns']
                    
                    # Get foreign key information
                    fks = inspector.get_foreign_keys(table_name)
                    for fk in fks:
                        if 'constrained_columns' in fk and 'referred_table' in fk and 'referred_columns' in fk:
                            schema_info[table_name]['foreign_keys'].append({
                                'constrained_columns': fk['constrained_columns'],
                                'referred_table': fk['referred_table'],
                                'referred_columns': fk['referred_columns']
                            })
                    
                    # Get index information
                    indexes = inspector.get_indexes(table_name)
                    for index in indexes:
                        if 'name' in index and 'column_names' in index:
                            schema_info[table_name]['indexes'].append({
                                'name': index['name'],
                                'column_names': index['column_names'],
                                'unique': index.get('unique', False)
                            })
                
                # Infer relationships if no foreign keys are defined
                if not any(len(table_info.get('foreign_keys', [])) > 0 for table_info in schema_info.values() 
                          if isinstance(table_info, dict) and 'foreign_keys' in table_info):
                    self._infer_relationships(schema_info)
                
                # Store in session state for future use
                st.session_state.schema_info = schema_info
                
                return schema_info
        except Exception as e:
            print(f"Error detecting schema: {str(e)}")
            return {}
    
    def _infer_relationships(self, schema_info):
        """
        Infer relationships between tables when no explicit foreign keys are defined.
        Uses naming conventions and primary key matching to identify potential relationships.
        
        Args:
            schema_info (dict): Schema information dictionary to update with inferred relationships
        """
        # Skip if database type is already in schema_info
        if not isinstance(schema_info, dict) or 'database_type' not in schema_info:
            return
        
        # Get all tables and their primary keys
        tables_with_pk = {}
        for table_name, table_info in schema_info.items():
            if isinstance(table_info, dict) and 'primary_key' in table_info and table_info['primary_key']:
                tables_with_pk[table_name] = table_info['primary_key']
        
        # Look for potential foreign keys based on naming conventions
        for table_name, table_info in schema_info.items():
            if not isinstance(table_info, dict) or table_name == 'database_type':
                continue
                
            # Skip if table already has foreign keys defined
            if 'foreign_keys' in table_info and table_info['foreign_keys']:
                continue
                
            # Get column names for this table
            if 'columns' not in table_info:
                continue
                
            column_names = list(table_info['columns'].keys())
            
            # Check each column for potential foreign key relationship
            for column_name in column_names:
                # Common patterns for foreign keys: table_id, tableId, id_table
                potential_tables = []
                
                # Check for table_id pattern
                if column_name.endswith('_id'):
                    base_name = column_name[:-3]  # Remove _id
                    # Check for singular/plural variations
                    for t_name in tables_with_pk:
                        if t_name.lower() == base_name.lower() or t_name.lower() == f"{base_name}s".lower():
                            potential_tables.append(t_name)
                
                # Check for id_table pattern
                elif column_name.startswith('id_'):
                    base_name = column_name[3:]  # Remove id_
                    for t_name in tables_with_pk:
                        if t_name.lower() == base_name.lower() or t_name.lower() == f"{base_name}s".lower():
                            potential_tables.append(t_name)
                
                # Check for tableId pattern (camelCase)
                elif 'id' in column_name.lower():
                    # Try to split at 'Id'
                    parts = re.split(r'(?i)id', column_name)
                    if len(parts) > 1:
                        base_name = parts[0]
                        for t_name in tables_with_pk:
                            if t_name.lower() == base_name.lower() or t_name.lower() == f"{base_name}s".lower():
                                potential_tables.append(t_name)
                
                # Add inferred foreign key if found
                for ref_table in potential_tables:
                    # Only add if the referenced table has a primary key
                    if ref_table in tables_with_pk and tables_with_pk[ref_table]:
                        # Add the inferred foreign key
                        if 'foreign_keys' not in table_info:
                            table_info['foreign_keys'] = []
                            
                        table_info['foreign_keys'].append({
                            'constrained_columns': [column_name],
                            'referred_table': ref_table,
                            'referred_columns': tables_with_pk[ref_table],
                            'inferred': True  # Mark as inferred, not explicit
                        })
    
    def _analyze_data_patterns(self, schema_info):
        """
        Analyze data patterns in the database to provide additional context.
        Samples data to identify common values, ranges, and patterns.
        
        Args:
            schema_info (dict): Schema information dictionary to update with data patterns
        """
        if not isinstance(schema_info, dict) or 'database_type' not in schema_info:
            return
            
        try:
            # Process each table
            for table_name, table_info in schema_info.items():
                if not isinstance(table_info, dict) or table_name == 'database_type' or 'columns' not in table_info:
                    continue
                    
                # Initialize data patterns dictionary if not present
                if 'data_patterns' not in table_info:
                    table_info['data_patterns'] = {}
                
                # Create a metadata object
                metadata = MetaData()
                
                # Reflect the table
                table = Table(table_name, metadata, autoload_with=self.db_engine)
                
                # Sample data (limit to 1000 rows for performance)
                query = select([table]).limit(1000)
                result = self.db_engine.execute(query)
                
                # Convert to DataFrame for easier analysis
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                
                # Analyze each column
                for column_name in table_info['columns']:
                    if column_name not in df.columns:
                        continue
                        
                    # Skip analysis for large text columns
                    col_type = table_info['columns'][column_name]['type'].lower()
                    if 'text' in col_type or 'blob' in col_type:
                        continue
                    
                    # Initialize column patterns
                    column_patterns = {
                        'sample_values': [],
                        'null_percentage': 0,
                        'unique_percentage': 0
                    }
                    
                    # Calculate null percentage
                    null_count = df[column_name].isna().sum()
                    column_patterns['null_percentage'] = round((null_count / len(df)) * 100, 2) if len(df) > 0 else 0
                    
                    # Calculate unique percentage
                    unique_count = df[column_name].nunique()
                    column_patterns['unique_percentage'] = round((unique_count / len(df)) * 100, 2) if len(df) > 0 else 0
                    
                    # Get sample values (non-null)
                    non_null_values = df[column_name].dropna()
                    if len(non_null_values) > 0:
                        # For numeric columns, get min, max, avg
                        if pd.api.types.is_numeric_dtype(non_null_values):
                            column_patterns['min'] = float(non_null_values.min())
                            column_patterns['max'] = float(non_null_values.max())
                            column_patterns['avg'] = float(non_null_values.mean())
                        
                        # For date columns, get min, max
                        elif pd.api.types.is_datetime64_dtype(non_null_values):
                            column_patterns['min_date'] = str(non_null_values.min())
                            column_patterns['max_date'] = str(non_null_values.max())
                        
                        # For all columns, get most common values
                        value_counts = non_null_values.value_counts().head(5)
                        for value, count in value_counts.items():
                            column_patterns['sample_values'].append({
                                'value': str(value),
                                'count': int(count),
                                'percentage': round((count / len(non_null_values)) * 100, 2)
                            })
                    
                    # Add column patterns to schema info
                    table_info['data_patterns'][column_name] = column_patterns
        
        except Exception as e:
            print(f"Error analyzing data patterns: {str(e)}")
    
    def _format_schema_for_llm(self, schema_info):
        """
        Format the schema information in a way that's easy for LLMs to understand.
        Creates a text representation of the schema with tables, columns, and relationships.
        
        Args:
            schema_info (dict): Schema information dictionary
            
        Returns:
            str: Formatted schema text for LLM consumption
        """
        if not schema_info:
            return "No schema information available."
        
        formatted_schema = []
        
        # Add database type
        db_type = schema_info.get('database_type', 'unknown')
        formatted_schema.append(f"DATABASE TYPE: {db_type}")
        formatted_schema.append("")
        
        # Process each table
        for table_name, table_info in schema_info.items():
            if table_name == 'database_type' or not isinstance(table_info, dict):
                continue
            
            formatted_schema.append(f"TABLE: {table_name}")
            
            # Add columns
            if 'columns' in table_info:
                formatted_schema.append("COLUMNS:")
                for col_name, col_info in table_info['columns'].items():
                    col_type = col_info.get('type', 'unknown')
                    nullable = "NULL" if col_info.get('nullable', True) else "NOT NULL"
                    default = f"DEFAULT {col_info.get('default', 'NULL')}"
                    
                    # Mark primary key columns
                    pk_marker = ""
                    if 'primary_key' in table_info and col_name in table_info['primary_key']:
                        pk_marker = " PRIMARY KEY"
                    
                    formatted_schema.append(f"  - {col_name}: {col_type} {nullable} {default}{pk_marker}")
            
            # Add foreign keys
            if 'foreign_keys' in table_info and table_info['foreign_keys']:
                formatted_schema.append("FOREIGN KEYS:")
                for fk in table_info['foreign_keys']:
                    constrained_cols = ", ".join(fk.get('constrained_columns', []))
                    referred_table = fk.get('referred_table', 'unknown')
                    referred_cols = ", ".join(fk.get('referred_columns', []))
                    inferred = " (inferred)" if fk.get('inferred', False) else ""
                    
                    formatted_schema.append(f"  - {constrained_cols} -> {referred_table}({referred_cols}){inferred}")
            
            # Add data patterns if available
            if 'data_patterns' in table_info and table_info['data_patterns']:
                formatted_schema.append("DATA PATTERNS:")
                for col_name, patterns in table_info['data_patterns'].items():
                    # Skip if no patterns detected
                    if not patterns:
                        continue
                    
                    formatted_schema.append(f"  - {col_name}:")
                    
                    # Add null and unique percentages
                    null_pct = patterns.get('null_percentage', 0)
                    unique_pct = patterns.get('unique_percentage', 0)
                    formatted_schema.append(f"    - NULL: {null_pct}%, Unique: {unique_pct}%")
                    
                    # Add numeric ranges if available
                    if 'min' in patterns and 'max' in patterns:
                        min_val = patterns['min']
                        max_val = patterns['max']
                        avg_val = patterns.get('avg', 'N/A')
                        formatted_schema.append(f"    - Range: {min_val} to {max_val}, Avg: {avg_val}")
                    
                    # Add date ranges if available
                    if 'min_date' in patterns and 'max_date' in patterns:
                        min_date = patterns['min_date']
                        max_date = patterns['max_date']
                        formatted_schema.append(f"    - Date range: {min_date} to {max_date}")
                    
                    # Add sample values if available
                    if 'sample_values' in patterns and patterns['sample_values']:
                        formatted_schema.append("    - Common values:")
                        for value_info in patterns['sample_values']:
                            value = value_info.get('value', 'unknown')
                            count = value_info.get('count', 0)
                            percentage = value_info.get('percentage', 0)
                            formatted_schema.append(f"      - {value}: {count} occurrences ({percentage}%)")
            
            formatted_schema.append("")  # Empty line between tables
        
        # Add usage hints for the LLM
        formatted_schema.append("SCHEMA USAGE GUIDELINES:")
        formatted_schema.append("1. Always use table and column names exactly as shown above")
        formatted_schema.append("2. Respect primary key and foreign key relationships when joining tables")
        formatted_schema.append("3. Consider data patterns and common values when filtering data")
        formatted_schema.append("4. Be aware of denormalized structures that might require special handling")
        formatted_schema.append("5. Use appropriate SQL syntax for the database type")
        
        return "\n".join(formatted_schema)
    
    def get_schema(self, force_refresh=False, analyze_data=True):
        """
        Public method to retrieve schema information with options for refresh and data analysis.
        
        Args:
            force_refresh (bool): If True, forces a complete schema refresh regardless of cache
            analyze_data (bool): If True, performs data pattern analysis on columns
            
        Returns:
            dict: Complete schema information with tables, columns, relationships, and patterns
        """
        # Clear cache if force refresh is requested
        if force_refresh and 'schema_info' in st.session_state:
            del st.session_state.schema_info
            print("Schema cache cleared for forced refresh")
        
        # Detect schema (will use cache if available and valid)
        schema_info = self._detect_schema()
        
        # Skip data analysis if not requested (for performance)
        if not analyze_data:
            # Remove data pattern information to save processing time
            for table_name in schema_info:
                if isinstance(schema_info[table_name], dict) and 'data_patterns' in schema_info[table_name]:
                    schema_info[table_name]['data_patterns'] = {}
        
        return schema_info
    
    def process(self, input_data):
        """
        Process input data and provide dynamically detected schema information.
        Performs intelligent schema detection and caching for efficient operation.
        
        Args:
            input_data (dict): Input data containing user query
            
        Returns:
            dict: Output data with schema information and metadata
        """
        # Get schema information
        schema_info = self._detect_schema()
        
        # Format schema for LLM consumption
        formatted_schema = self._format_schema_for_llm(schema_info)
        
        # Add schema information to the output data
        output_data = input_data.copy()
        output_data["schema"] = formatted_schema
        output_data["schema_info"] = schema_info
        output_data["db_dialect"] = self.dialect_name
        
        return output_data


# Keep the original SchemaParser for backward compatibility
class SchemaParser:
    """
    Legacy schema parser that works with static schema strings.
    Maintained for backward compatibility.
    """
    
    def __init__(self, schema):
        """
        Initialize the SchemaParser agent with a SQL schema.
        
        Args:
            schema (str): SQL schema as a string
        """
        self.schema = schema
        self.parsed_schema = self._parse_schema(schema)
    
    def _parse_schema(self, schema):
        """
        Parse the SQL schema to extract table and column information.
        
        Args:
            schema (str): SQL schema as a string
            
        Returns:
            dict: Parsed schema information
        """
        # Simple parsing logic for CREATE TABLE statements
        tables = {}
        
        # Find all CREATE TABLE statements
        create_table_pattern = r'CREATE\s+TABLE\s+(\w+)\s*\((.*?)\);'
        matches = re.findall(create_table_pattern, schema, re.DOTALL | re.IGNORECASE)
        
        for table_name, columns_text in matches:
            tables[table_name] = {
                'columns': {},
                'primary_key': [],
                'foreign_keys': []
            }
            
            # Parse columns
            column_entries = columns_text.split(',')
            for entry in column_entries:
                entry = entry.strip()
                
                # Skip if empty
                if not entry:
                    continue
                
                # Check if it's a PRIMARY KEY constraint
                pk_match = re.match(r'PRIMARY\s+KEY\s*\(([^)]+)\)', entry, re.IGNORECASE)
                if pk_match:
                    pk_columns = [col.strip() for col in pk_match.group(1).split(',')]
                    tables[table_name]['primary_key'] = pk_columns
                    continue
                
                # Check if it's a FOREIGN KEY constraint
                fk_match = re.match(r'FOREIGN\s+KEY\s*\(([^)]+)\)\s+REFERENCES\s+(\w+)\s*\(([^)]+)\)', entry, re.IGNORECASE)
                if fk_match:
                    fk_columns = [col.strip() for col in fk_match.group(1).split(',')]
                    ref_table = fk_match.group(2)
                    ref_columns = [col.strip() for col in fk_match.group(3).split(',')]
                    
                    tables[table_name]['foreign_keys'].append({
                        'constrained_columns': fk_columns,
                        'referred_table': ref_table,
                        'referred_columns': ref_columns
                    })
                    continue
                
                # Regular column definition
                col_match = re.match(r'(\w+)\s+([\w\(\)]+)(\s+NOT\s+NULL)?', entry, re.IGNORECASE)
                if col_match:
                    col_name = col_match.group(1)
                    col_type = col_match.group(2)
                    nullable = col_match.group(3) is None  # If NOT NULL is present, nullable is False
                    
                    tables[table_name]['columns'][col_name] = {
                        'type': col_type,
                        'nullable': nullable
                    }
        
        return tables
    
    def process(self, input_data):
        """
        Process input data and provide schema information.
        
        Args:
            input_data (dict): Input data containing user query
            
        Returns:
            dict: Output data with schema information
        """
        # Add schema information to the output data
        output_data = input_data.copy()
        output_data["schema"] = self.schema
        output_data["schema_info"] = self.parsed_schema
        
        return output_data
