"""
QueryTranslator agent for the Universal SQL Conversational Agent.
Responsible for translating natural language queries into SQL commands.
"""

import os
import re
import google.generativeai as genai

class QueryTranslator:
    """
    Agent responsible for translating natural language queries into SQL commands.
    Uses OpenAI API to perform the translation based on schema context.
    """
    
    def __init__(self):
        """Initialize the QueryTranslator agent."""
        # Ensure Gemini API key is set
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not found. Please set the GEMINI_API_KEY environment variable.")
        
        # Configure Gemini API
        genai.configure(api_key=self.api_key)
    
    def _translate_to_sql(self, user_query, schema_info, conversation_history=None, last_query_results=None, last_sql_query=None):
        """
        Translate a natural language query to SQL using Gemini API.
        
        Args:
            user_query (str): Natural language query from the user
            schema_info (dict): Parsed schema information
            conversation_history (list): Previous conversation exchanges
            last_query_results (list): Results from the last query
            last_sql_query (str): The last SQL query executed
            
        Returns:
            str: Generated SQL query
        """
        # Format the schema information for the prompt
        schema_description = self._format_schema_for_prompt(schema_info)
        
        # Build conversation context
        conversation_context = ""
        if conversation_history and len(conversation_history) > 0:
            conversation_context = "Previous conversation:\n"
            for i, exchange in enumerate(conversation_history[-3:]):  # Use last 3 exchanges for context
                conversation_context += f"User: {exchange['user_query']}\n"
                conversation_context += f"Answer: {exchange['response']}\n"
        
        # Add information about the last query results if available
        last_results_context = ""
        if last_query_results and last_sql_query:
            last_results_context = f"""
            The last SQL query was: {last_sql_query}
            
            It returned these results (limited to first 5 rows):
            {str(last_query_results[:5] if len(last_query_results) > 5 else last_query_results)}
            """
        
        # Determine database type from schema_info for specific SQL dialect rules
        database_type = schema_info.get("database_type", "sqlite").lower()
        
        # Create database-specific SQL compatibility rules
        sql_compatibility_rules = ""
        if "sqlite" in database_type:
            sql_compatibility_rules = """
            ⚙️ SQLITE COMPATIBILITY RULES:
            1. SQLite can only execute one statement at a time, so:
               - Do not use multiple statements separated by semicolons
               - Do not use BEGIN/ROLLBACK transaction blocks
               - Include LIMIT clauses as part of the main query, not as a separate statement
            
            2. SQLite has limited support for complex features, so:
               - Avoid window functions (OVER, PARTITION BY) 
               - Avoid Common Table Expressions (WITH clauses) if possible
               - Use standard JOIN syntax instead of specialized variants
            
            3. If uncertain about data volume, include LIMIT 100 within the main query.
            """
        elif "mysql" in database_type or "mariadb" in database_type:
            sql_compatibility_rules = """
            ⚙️ MYSQL/MARIADB COMPATIBILITY RULES:
            1. Use backticks (`) for table and column names if they contain spaces or are reserved words
            2. LIMIT must come after ORDER BY
            3. Use IFNULL() instead of COALESCE() for better compatibility
            4. For date operations, use DATE_FORMAT() and STR_TO_DATE() functions
            5. Use GROUP_CONCAT() for string aggregation
            """
        elif "postgres" in database_type or "postgresql" in database_type:
            sql_compatibility_rules = """
            ⚙️ POSTGRESQL COMPATIBILITY RULES:
            1. Use double quotes (") for table and column names if they contain spaces or are reserved words
            2. Take advantage of PostgreSQL's rich features:
               - Common Table Expressions (WITH clauses) for complex queries
               - Window functions for analytical queries
               - JSONB operations for JSON data
            3. Use || for string concatenation
            4. Use COALESCE() for NULL handling
            """
        elif "mssql" in database_type or "sqlserver" in database_type:
            sql_compatibility_rules = """
            ⚙️ SQL SERVER COMPATIBILITY RULES:
            1. Use square brackets [] for table and column names if they contain spaces or are reserved words
            2. Use TOP instead of LIMIT for row limiting
            3. Use ISNULL() instead of COALESCE() for better performance
            4. For date operations, use CONVERT() and CAST() functions
            5. Use STRING_AGG() for string aggregation (SQL Server 2017+)
            """
        else:
            # Default to generic SQL rules if database type is unknown
            sql_compatibility_rules = """
            ⚙️ SQL COMPATIBILITY RULES:
            1. Use standard SQL syntax that works across most database systems
            2. Avoid database-specific functions and features
            3. Include LIMIT 100 for large result sets
            4. Use single quotes for string literals
            """
        
        # Create the prompt for the Gemini API with comprehensive SQL generation guidelines
        prompt = f"""
        Given the following database schema:
        {schema_description}
        
        {conversation_context}
        
        {last_results_context}
        
        Translate this natural language query into a syntactically correct, schema-aware, optimized SQL query:
        "{user_query}"
        
        🔧 CRITICAL SQL GENERATION PRINCIPLES:
        1. SCHEMA AWARENESS: Use ONLY tables and columns that exist in the provided schema.
           - Never assume column names, data types, or values unless they are explicitly defined in the schema
           - If a column or value is mentioned in the query but doesn't exist in the schema, find appropriate alternatives
           - Use the exact column names and data types as defined in the schema
           - IMPORTANT: All string comparisons should be CASE INSENSITIVE (the system will handle this automatically)
        
        2. PERFORMANCE OPTIMIZATION:
           - Avoid unnecessary joins to large tables unless explicitly required
           - Use EXISTS or IN for set-based existence checks rather than JOINs when appropriate
           - For conditions like "at least X of something", use proper aggregation with COUNT() and appropriate joins
           - Prefer simpler queries over complex ones when they achieve the same result
        
        3. RELATIONSHIP HANDLING:
           - Use the foreign key relationships defined in the schema for joins
           - For queries involving multiple entity types, use appropriate subqueries or joins
           - When checking for multiple conditions across related tables, use EXISTS with correlated subqueries
        
        4. AGGREGATION AND GROUPING:
           - Use GROUP BY with appropriate HAVING clauses for aggregation
           - Ensure aggregation functions (COUNT, SUM, AVG, etc.) are used correctly
           - For queries requiring minimum counts (e.g., "items with at least X related records"), use COUNT() with proper grouping
        
        5. ALIAS CONSISTENCY:
           - Use meaningful table aliases that reflect the table name (e.g., p for products)
           - Maintain consistent alias usage throughout the query
           - Qualify column references with table aliases to avoid ambiguity
        
        6. RESULT HANDLING:
           - Even if no data might match the query, ensure the SQL is syntactically valid and optimized
           - Include appropriate ORDER BY for sorted results when it makes sense
           - Use LIMIT appropriately based on the expected result size
        
        7. CONTEXT AWARENESS:
           - If the query refers to previous results or contains pronouns, use conversation context to resolve references
           - When the query is ambiguous, prefer the most logical interpretation based on the schema
           - String values in queries should be treated as case insensitive (e.g., 'Category' should match 'category', 'CATEGORY', etc.)
        
        {sql_compatibility_rules}
        
        FORMAT REQUIREMENTS:
        - Format the SQL with proper indentation for readability
        - Capitalize SQL keywords for clarity
        - Return only the SQL query without any explanations or markdown formatting
        - IMPORTANT: Ensure the query is a single statement with no semicolons except at the very end
        
        QUERY CONSTRUCTION APPROACH:
        1. Analyze the schema to identify relevant tables and their relationships
        2. Determine the main entities and attributes needed for the query
        3. Construct appropriate JOIN conditions using primary/foreign keys
        4. Apply precise filtering conditions using WHERE clauses with exact column names
        5. Add GROUP BY, HAVING, and ORDER BY as needed for aggregation and sorting
        6. Include appropriate LIMIT clause based on the database type
        7. Verify that all column references are valid against the schema
        """
        
        try:
            # Configure Gemini model
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Call Gemini API
            response = model.generate_content(prompt)
            
            # Extract and clean the SQL query from the response
            sql_query = response.text.strip()
            
            # Remove any backticks or markdown formatting that might be in the response
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            
            # Ensure the query is properly formatted with consistent indentation
            sql_query = self._format_sql_query(sql_query)
            
            return sql_query
            
        except Exception as e:
            print(f"Error translating query: {e}")
            return f"ERROR: {str(e)}"
    
    def _format_schema_for_prompt(self, schema_info):
        """
        Format the schema information for inclusion in the prompt.
        
        Args:
            schema_info (dict): Parsed schema information from DynamicSchemaParser
            
        Returns:
            str: Formatted schema description with tables, columns, primary keys, foreign keys,
                 relationships, and sample data patterns
        """
        schema_description = ""
        
        # Add database type information if available
        if "database_type" in schema_info:
            schema_description += f"Database Type: {schema_info['database_type']}\n\n"
        
        # Process each table in the schema
        for table_name, table_info in schema_info.items():
            if table_name == "database_type":
                continue
                
            schema_description += f"Table: {table_name}\n"
            
            # Add table description if available
            if "description" in table_info:
                schema_description += f"Description: {table_info['description']}\n"
            
            # Add primary key information if available
            if "primary_keys" in table_info and table_info["primary_keys"]:
                schema_description += f"Primary Key: {', '.join(table_info['primary_keys'])}\n"
            
            # Add foreign key information if available
            if "foreign_keys" in table_info and table_info["foreign_keys"]:
                schema_description += "Foreign Keys:\n"
                for fk in table_info["foreign_keys"]:
                    if isinstance(fk, dict) and 'columns' in fk and 'referred_table' in fk and 'referred_columns' in fk:
                        for i, col in enumerate(fk['columns']):
                            ref_col = fk['referred_columns'][i] if i < len(fk['referred_columns']) else fk['referred_columns'][0]
                            schema_description += f"  - {col} references {fk['referred_table']}.{ref_col}\n"
            
            # Add inferred relationships if available
            if "relationships" in table_info and table_info["relationships"]:
                schema_description += "Inferred Relationships:\n"
                for rel in table_info["relationships"]:
                    schema_description += f"  - {rel}\n"
            
            # Add columns with enhanced type information
            schema_description += "Columns:\n"
            for col_name, col_info in table_info["columns"].items():
                # Add nullable information if available
                nullable_info = "" if col_info.get("nullable", True) else " NOT NULL"
                
                # Add data pattern information if available
                data_pattern = ""
                if "data_pattern" in col_info:
                    data_pattern = f" [Pattern: {col_info['data_pattern']}]"
                
                # Add sample values if available
                sample_values = ""
                if "sample_values" in col_info and col_info["sample_values"]:
                    sample_values = f" [Examples: {', '.join(str(v) for v in col_info['sample_values'][:3])}]"
                
                schema_description += f"  - {col_name} ({col_info['type']}{nullable_info}){data_pattern}{sample_values}\n"
            
            schema_description += "\n"
        
        # Add denormalized patterns if available
        if "denormalized_patterns" in schema_info:
            schema_description += "Denormalized Patterns:\n"
            for pattern in schema_info["denormalized_patterns"]:
                schema_description += f"  - {pattern}\n\n"
        
        # Add sample data information based on detected patterns
        schema_description += "Sample Data Information:\n"
        
        # Use dynamically detected patterns if available, otherwise use generic patterns
        if "data_patterns" in schema_info:
            for pattern_desc in schema_info["data_patterns"]:
                schema_description += f"- {pattern_desc}\n"
        else:
            # Fallback to generic patterns based on the detected tables
            schema_description += "- Use exact column and table names from the schema above\n"
            schema_description += "- String comparisons should be case-insensitive where possible\n"
            
            # Add table-specific generic patterns based on detected tables
            tables = [t for t in schema_info.keys() if t != "database_type"]
            for table in tables:
                # Add some sample patterns based on table name
                if "order" in table.lower():
                    schema_description += f"- The {table} table likely contains order information\n"
                if "product" in table.lower():
                    schema_description += f"- The {table} table likely contains product information\n"
                if "user" in table.lower() or "customer" in table.lower():
                    schema_description += f"- The {table} table likely contains user/customer information\n"
                if "categor" in table.lower():
                    schema_description += f"- The {table} table likely contains category information\n"
        
        return schema_description
    
    def _format_sql_query(self, sql_query):
        """
        Format the SQL query for better readability.
        
        Args:
            sql_query (str): The raw SQL query
            
        Returns:
            str: Formatted SQL query with consistent indentation
        """
        # List of keywords that should start on a new line
        new_line_keywords = [
            "SELECT", "FROM", "WHERE", "GROUP BY", "HAVING", "ORDER BY", 
            "LIMIT", "JOIN", "LEFT JOIN", "RIGHT JOIN", "INNER JOIN", 
            "OUTER JOIN", "FULL JOIN", "UNION", "INTERSECT", "EXCEPT"
        ]
        
        # Additional SQL keywords to capitalize
        additional_keywords = [
            "AS", "ON", "AND", "OR", "NOT", "IN", "EXISTS", "BETWEEN", "LIKE",
            "IS NULL", "IS NOT NULL", "ASC", "DESC", "WITH", "CASE", "WHEN", "THEN", "ELSE", "END",
            "COUNT", "SUM", "AVG", "MIN", "MAX", "DISTINCT"
        ]
        
        # Replace keywords with new line and proper indentation
        formatted_query = sql_query
        for keyword in new_line_keywords:
            # Only add newlines for keywords that are not at the start of the query
            # Use raw string for regex pattern to avoid escape sequence issues
            formatted_query = re.sub(fr"(?i)(?<!^)\s+{keyword}\s+", f"\n{keyword} ", formatted_query)
        
        # Ensure the first keyword is capitalized and not indented
        for keyword in new_line_keywords:
            if formatted_query.upper().startswith(keyword):
                formatted_query = keyword + formatted_query[len(keyword):]
                break
        
        # Capitalize all SQL keywords for consistency
        for keyword in new_line_keywords + additional_keywords:
            # Use word boundaries to avoid replacing parts of other words
            formatted_query = re.sub(fr"(?i)\b{keyword}\b", keyword, formatted_query)
        
        # Add proper indentation for clauses after the first line
        lines = formatted_query.split('\n')
        indented_lines = [lines[0]]
        for line in lines[1:]:
            indented_lines.append("  " + line)  # Add 2 spaces for indentation
        
        # Check for potential performance issues and add warning comments if needed
        formatted_query = '\n'.join(indented_lines)
        if self._is_potentially_expensive_query(formatted_query):
            formatted_query = self._add_performance_warnings(formatted_query)
        
        return formatted_query
        
    def _is_potentially_expensive_query(self, sql_query):
        """
        Check if a query might be expensive to execute.
        
        Args:
            sql_query (str): The SQL query to check
            
        Returns:
            bool: True if the query might be expensive, False otherwise
        """
        # Check for patterns that might indicate an expensive query
        expensive_patterns = [
            # No WHERE clause in a query with multiple joins
            r'(?i)FROM\s+\w+\s+JOIN.*JOIN.*(?!WHERE)',
            # GROUP BY with no apparent filtering
            r'(?i)GROUP BY.*(?!WHERE|HAVING)',
            # Nested subqueries with aggregations
            r'(?i)SELECT.*\(SELECT.*(?:COUNT|SUM|AVG).*FROM.*\)',
            # Cross joins (cartesian products)
            r'(?i)FROM\s+\w+\s*,\s*\w+\s*(?!WHERE)',
            # Full table scans on potentially large tables
            r'(?i)FROM\s+(?:transactions|logs|events|history)\s*(?!WHERE)'
        ]
        
        for pattern in expensive_patterns:
            if re.search(pattern, sql_query):
                return True
        
        return False
    
    def _add_performance_warnings(self, sql_query):
        """
        Add performance warning comments to a potentially expensive query.
        
        Args:
            sql_query (str): The SQL query to add warnings to
            
        Returns:
            str: The SQL query with performance warnings
        """
        # Add a comment and LIMIT clause if not already present
        if not re.search(r'(?i)\bLIMIT\b', sql_query):
            # Split into lines to add the comment and LIMIT at the appropriate position
            lines = sql_query.split('\n')
            
            # Find if there's an ORDER BY clause to add LIMIT after it
            order_by_index = -1
            for i, line in enumerate(lines):
                if re.search(r'(?i)\bORDER BY\b', line):
                    order_by_index = i
            
            # Add the performance warning comment at the top
            warning = "-- NOTE: This query might be expensive. Added LIMIT 100 for preview.\n"
            warning += "-- Remove LIMIT for full results if needed.\n"
            lines.insert(0, warning)
            
            # Add LIMIT clause after ORDER BY or at the end - ensure it's part of the main query for SQLite compatibility
            if order_by_index >= 0:
                # Add LIMIT after ORDER BY clause
                order_by_line = lines[order_by_index + 1]
                if not order_by_line.strip().endswith(';'):
                    lines[order_by_index + 1] = order_by_line + " LIMIT 100"
                else:
                    # If there's a semicolon, insert before it
                    lines[order_by_index + 1] = order_by_line.replace(';', ' LIMIT 100;')
            else:
                # Find the last non-empty line to add LIMIT to
                last_line_index = len(lines) - 1
                while last_line_index >= 0 and not lines[last_line_index].strip():
                    last_line_index -= 1
                
                if last_line_index >= 0:
                    last_line = lines[last_line_index]
                    if not last_line.strip().endswith(';'):
                        lines[last_line_index] = last_line + " LIMIT 100"
                    else:
                        # If there's a semicolon, insert before it
                        lines[last_line_index] = last_line.replace(';', ' LIMIT 100;')
                else:
                    # Fallback if no suitable line found
                    lines.append("LIMIT 100")
            
            return '\n'.join(lines)
        
        return sql_query
        
    def process(self, input_data):
        """
        Process input data and translate the user query to SQL.
        
        Args:
            input_data (dict): Input data containing user query and schema info
            
        Returns:
            dict: Output data with translated SQL query
        """
        # Extract information from the input data
        user_query = input_data.get("user_query", "")
        schema_info = input_data.get("schema_info", {})
        conversation_history = input_data.get("conversation_history", [])
        last_query_results = input_data.get("last_query_results", None)
        last_sql_query = input_data.get("last_sql_query", None)
        
        # Translate the query to SQL with conversation context
        sql_query = self._translate_to_sql(
            user_query, 
            schema_info, 
            conversation_history, 
            last_query_results, 
            last_sql_query
        )
        
        # Prepare output for the next agent
        output_data = {
            "user_query": user_query,
            "schema_info": schema_info,
            "sql_query": sql_query,
            "conversation_history": conversation_history,
            "last_query_results": last_query_results,
            "last_sql_query": last_sql_query
        }
        
        return output_data
