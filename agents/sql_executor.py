"""
SQLExecutor agent for the Universal SQL Conversational Agent.
Responsible for executing SQL queries on the database.
"""

from database.db_setup import execute_query

class SQLExecutor:
    """
    Agent responsible for executing SQL queries on the database.
    Takes translated SQL queries and executes them against the database.
    """
    
    def __init__(self, db_engine):
        """
        Initialize the SQLExecutor agent with a database engine.
        
        Args:
            db_engine: SQLAlchemy engine for database access
        """
        self.db_engine = db_engine
    
    def _execute_sql(self, sql_query):
        """
        Execute a SQL query on the database.
        
        Args:
            sql_query (str): SQL query to execute
            
        Returns:
            list: Query results as a list of dictionaries
        """
        try:
            # Execute the query using the database utility function
            results = execute_query(self.db_engine, sql_query)
            return {
                "success": True,
                "results": results,
                "error": None
            }
        except Exception as e:
            # Handle any errors during query execution
            return {
                "success": False,
                "results": None,
                "error": str(e)
            }
    
    def process(self, input_data):
        """
        Process input data and execute the SQL query.
        
        Args:
            input_data (dict): Input data containing SQL query
            
        Returns:
            dict: Output data with query execution results
        """
        # Extract information from the input data
        user_query = input_data.get("user_query", "")
        schema_info = input_data.get("schema_info", {})
        sql_query = input_data.get("sql_query", "")
        conversation_history = input_data.get("conversation_history", [])
        
        # Execute the SQL query
        execution_result = self._execute_sql(sql_query)
        
        # Prepare output for the next agent
        output_data = {
            "user_query": user_query,
            "schema_info": schema_info,
            "sql_query": sql_query,
            "execution_result": execution_result,
            "conversation_history": conversation_history,
            "last_sql_query": sql_query,
            "last_query_results": execution_result.get("results", [])
        }
        
        return output_data
