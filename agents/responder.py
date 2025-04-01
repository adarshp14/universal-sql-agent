"""
Responder agent for the Universal SQL Conversational Agent.
Responsible for formatting query results into natural language responses.
"""

import os
import pandas as pd
import google.generativeai as genai

class Responder:
    """
    Agent responsible for formatting query results into natural language responses.
    Takes query results and generates human-readable responses using OpenAI API.
    """
    
    def __init__(self):
        """Initialize the Responder agent."""
        # Ensure Gemini API key is set
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not found. Please set the GEMINI_API_KEY environment variable.")
        
        # Configure Gemini API
        genai.configure(api_key=self.api_key)
    
    def _format_response(self, user_query, sql_query, execution_result, schema_info=None, conversation_history=None):
        """
        Format the query results into a natural language response with schema context awareness.
        
        Args:
            user_query (str): Original natural language query
            sql_query (str): Translated SQL query
            execution_result (dict): Results of the SQL query execution
            schema_info (dict, optional): Database schema information
            conversation_history (list, optional): Previous conversation exchanges
            
        Returns:
            str: Natural language response with contextual insights
        """
        # Check if the query execution was successful
        if not execution_result["success"]:
            error_msg = execution_result['error']
            
            # Provide more helpful error responses based on common SQL errors
            if "no such table" in error_msg.lower():
                table_name = error_msg.split("no such table:")[-1].strip().split()[0].strip("'\"`;,")
                return f"I couldn't find a table named '{table_name}' in the database. Please check the table name and try again."
            elif "no such column" in error_msg.lower():
                col_name = error_msg.split("no such column:")[-1].strip().split()[0].strip("'\"`;,")
                return f"I couldn't find a column named '{col_name}' in the database. Please check the column name and try again."
            elif "syntax error" in error_msg.lower():
                return f"There was a syntax error in the SQL query. This might be due to a complex request. Could you try rephrasing your question?"
            
            return f"I encountered an error while trying to answer your question: {error_msg}"
        
        # Get the query results
        results = execution_result["results"]
        
        # If there are no results, return a more informative response
        if not results:
            # Check if this is a SELECT query (most likely case for no results)
            if sql_query.strip().lower().startswith("select"):
                return "I didn't find any data matching your criteria. This could mean that there are no records that match your specific conditions."
            elif any(keyword in sql_query.lower() for keyword in ["update", "insert", "delete"]):
                return "The operation was completed successfully, but no records were affected."
            else:
                return "I couldn't find any data matching your query."
        
        # Format the results for the prompt
        # For large result sets, summarize rather than listing everything
        result_count = len(results)
        if result_count > 10:
            results_sample = results[:5]
            results_str = "\n".join([str(result) for result in results_sample])
            results_str += f"\n... and {result_count - 5} more results (showing first 5 only)"
        else:
            results_str = "\n".join([str(result) for result in results])
        
        # Add schema context if available
        schema_context = ""
        if schema_info:
            # Extract table names from the SQL query to provide relevant schema context
            tables_in_query = self._extract_tables_from_query(sql_query)
            if tables_in_query:
                schema_context = "Database context:\n"
                for table in tables_in_query:
                    if table in schema_info:
                        # Add table description if available
                        if 'description' in schema_info[table]:
                            schema_context += f"- {table}: {schema_info[table]['description']}\n"
                        
                        # Add relationship information if available
                        if 'relationships' in schema_info[table] and schema_info[table]['relationships']:
                            for rel in schema_info[table]['relationships'][:2]:  # Limit to 2 relationships
                                schema_context += f"  - {rel['description']}\n"
        
        # Add conversation context if available
        conversation_context = ""
        if conversation_history and len(conversation_history) > 0:
            # Get the last exchange (excluding the current one)
            if len(conversation_history) > 1:
                last_exchange = conversation_history[-2]
                conversation_context = f"Previous question: {last_exchange['user_query']}\n"
                conversation_context += f"Previous answer summary: {last_exchange['response'][:100]}...\n"
        
        # Create the prompt for the Gemini API with enhanced context
        prompt = f"""
        Given the following:
        
        User's question: "{user_query}"
        SQL query used: {sql_query}
        Query results: {results_str}
        {schema_context}
        {conversation_context}
        
        Please provide a natural language response that answers the user's question based on these results.
        The response should be:
        1. Conversational and easy to understand for non-technical users
        2. Directly addressing the user's question with insights from the data
        3. Highlighting any interesting patterns or anomalies in the results
        4. Contextually aware of the database structure and relationships
        
        Do not mention the SQL query or technical database details in your response unless specifically asked.
        Focus on providing a clear, insightful answer that a business user would find valuable.
        """
        
        try:
            # Configure Gemini model - use a more capable model for complex responses
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Call Gemini API
            response = model.generate_content(prompt)
            
            # Extract the response from the API result
            natural_response = response.text.strip()
            
            # Add follow-up suggestions if appropriate
            if len(results) > 5 and not any(phrase in user_query.lower() for phrase in ["how many", "count", "total"]):
                natural_response += "\n\nWould you like to see more detailed information about specific aspects of these results?"
        except Exception as e:
            print(f"Error formatting response: {e}")
            natural_response = f"I found some results, but encountered an error while formatting the response: {str(e)}"
        
        return natural_response
        
    def _extract_tables_from_query(self, sql_query):
        """
        Extract table names from an SQL query using basic parsing.
        
        Args:
            sql_query (str): The SQL query to analyze
            
        Returns:
            list: List of table names found in the query
        """
        # Normalize the query for easier parsing
        normalized_query = " " + sql_query.lower() + " "
        
        # Look for common SQL patterns that reference tables
        tables = set()
        
        # FROM clause
        from_parts = normalized_query.split(" from ")
        for i in range(1, len(from_parts)):
            part = from_parts[i].strip()
            # Extract the table name (stopping at the next clause or join)
            next_clause = min(
                (part.find(f" {clause} ") for clause in ["where", "group", "having", "order", "limit", "join", "inner", "left", "right"] 
                 if part.find(f" {clause} ") > 0),
                default=len(part)
            )
            table_part = part[:next_clause].strip()
            # Handle aliases and quoted names
            table_name = table_part.split(" ")[0].strip("`\"' ;	\n")
            if table_name and not table_name.startswith("("):  # Skip subqueries
                tables.add(table_name)
        
        # JOIN clauses
        join_keywords = [" join ", " inner join ", " left join ", " right join ", " full join "]
        for keyword in join_keywords:
            join_parts = normalized_query.split(keyword)
            for i in range(1, len(join_parts)):
                part = join_parts[i].strip()
                # Extract the table name (stopping at ON or the next clause)
                next_part = min(
                    part.find(" on "),
                    min((part.find(f" {clause} ") for clause in ["where", "group", "having", "order", "limit", "join", "inner", "left", "right"] 
                         if part.find(f" {clause} ") > 0),
                        default=len(part))
                )
                table_part = part[:next_part].strip()
                # Handle aliases and quoted names
                table_name = table_part.split(" ")[0].strip("`\"' ;	\n")
                if table_name and not table_name.startswith("("):  # Skip subqueries
                    tables.add(table_name)
        
        return list(tables)
    
    def process(self, input_data):
        """
        Process input data and generate a natural language response with schema awareness.
        
        Args:
            input_data (dict): Input data containing query results and schema information
            
        Returns:
            dict: Output data with natural language answer and visualization metadata
        """
        # Extract information from the input data
        user_query = input_data.get("user_query", "")
        sql_query = input_data.get("sql_query", "")
        execution_result = input_data.get("execution_result", {})
        conversation_history = input_data.get("conversation_history", [])
        last_query_results = input_data.get("last_query_results", [])
        last_sql_query = input_data.get("last_sql_query", "")
        
        # Extract schema information if available
        schema_info = input_data.get("schema_info", {})
        db_dialect = input_data.get("db_dialect", "unknown")
        
        # Format the response with enhanced context awareness
        response_text = self._format_response(
            user_query=user_query, 
            sql_query=sql_query, 
            execution_result=execution_result,
            schema_info=schema_info,
            conversation_history=conversation_history
        )
        
        # Generate follow-up question suggestions if appropriate
        follow_up_suggestions = []
        if execution_result.get("success", False) and execution_result.get("results", []):
            # Only suggest follow-ups for successful queries with results
            results = execution_result.get("results", [])
            
            # Detect if this was an aggregation query
            is_aggregation = any(key.lower().startswith(('count', 'sum', 'avg', 'min', 'max')) 
                               for result in results[:1] for key in result.keys())
            
            # Suggest appropriate follow-ups based on query type
            if is_aggregation and len(results) > 1:
                follow_up_suggestions.append("Show me a breakdown by category")
                follow_up_suggestions.append("What's driving these numbers?")
            elif len(results) > 5:
                follow_up_suggestions.append("Can you show me more details?")
                follow_up_suggestions.append("What patterns do you see in this data?")
        
        # Update conversation history with this exchange
        conversation_history.append({
            "user_query": user_query,
            "response": response_text,
            "sql_query": sql_query,
            "results": execution_result.get("results", []),
            "timestamp": pd.Timestamp.now().isoformat()
        })
        
        # Determine appropriate visualization types based on the data
        viz_recommendations = self._recommend_visualizations(execution_result.get("results", []), sql_query)
        
        # Prepare the final output with all necessary data for visualization
        output_data = {
            "response": response_text,
            "sql_query": sql_query,
            "query_results": execution_result.get("results", []),
            "conversation_history": conversation_history,
            "last_query_results": execution_result.get("results", []),
            "last_sql_query": sql_query,
            "follow_up_suggestions": follow_up_suggestions,
            "visualization_recommendations": viz_recommendations,
            "db_dialect": db_dialect
        }
        
        return output_data
        
    def _recommend_visualizations(self, results, sql_query):
        """
        Recommend appropriate visualization types based on the query results.
        
        Args:
            results (list): Query results
            sql_query (str): The SQL query that produced the results
            
        Returns:
            dict: Recommended visualization types and configuration
        """
        if not results:
            return {"recommended_types": []}
            
        # Analyze the first result to understand the data structure
        sample = results[0]
        column_types = {}
        
        # Determine column types (numeric, categorical, date, etc.)
        for key, value in sample.items():
            if isinstance(value, (int, float)):
                column_types[key] = "numeric"
            elif isinstance(value, str):
                # Check if it might be a date
                if any(date_pattern in key.lower() for date_pattern in ["date", "time", "year", "month", "day"]):
                    column_types[key] = "date"
                else:
                    column_types[key] = "categorical"
            else:
                column_types[key] = "other"
        
        # Count column types
        numeric_cols = [col for col, type_ in column_types.items() if type_ == "numeric"]
        categorical_cols = [col for col, type_ in column_types.items() if type_ == "categorical"]
        date_cols = [col for col, type_ in column_types.items() if type_ == "date"]
        
        # Initialize recommendations
        recommendations = {
            "recommended_types": ["Table"],  # Table is always a safe default
            "primary_visualization": "Table",
            "column_types": column_types
        }
        
        # Recommend visualizations based on data characteristics
        if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
            recommendations["recommended_types"].append("Bar Chart")
            if len(results) <= 15:  # Bar charts work best with limited categories
                recommendations["primary_visualization"] = "Bar Chart"
        
        if len(numeric_cols) >= 2:
            recommendations["recommended_types"].append("Scatter Plot")
        
        if len(date_cols) >= 1 and len(numeric_cols) >= 1:
            recommendations["recommended_types"].append("Line Chart")
            if "time series" in sql_query.lower() or "trend" in sql_query.lower():
                recommendations["primary_visualization"] = "Line Chart"
        
        if len(numeric_cols) > 1:
            recommendations["recommended_types"].append("Correlation Heatmap")
        
        if len(numeric_cols) == 1 and len(categorical_cols) >= 1:
            recommendations["recommended_types"].append("Pie Chart")
            if len(results) <= 7:  # Pie charts work best with few categories
                recommendations["primary_visualization"] = "Pie Chart"
        
        # Add configuration hints
        if "Bar Chart" in recommendations["recommended_types"]:
            recommendations["bar_chart_config"] = {
                "suggested_x": categorical_cols[0] if categorical_cols else None,
                "suggested_y": numeric_cols[0] if numeric_cols else None
            }
        
        if "Line Chart" in recommendations["recommended_types"]:
            recommendations["line_chart_config"] = {
                "suggested_x": date_cols[0] if date_cols else (categorical_cols[0] if categorical_cols else None),
                "suggested_y": numeric_cols[0] if numeric_cols else None
            }
        
        return recommendations
