#!/usr/bin/env python3
"""
Universal SQL Conversational Agent
Main application file that orchestrates the multi-agent system for natural language SQL queries.
"""

import os
import argparse
from dotenv import load_dotenv
from agents.schema_parser import SchemaParser, DynamicSchemaParser
from agents.query_translator import QueryTranslator
from agents.sql_executor import SQLExecutor
from agents.responder import Responder
from database.db_setup import setup_sample_database, get_db_engine

# Load environment variables from .env file
load_dotenv()

class AgentPipeline:
    """
    A pipeline to orchestrate multiple agents in sequence with enhanced context sharing.
    Replaces the CAMEL OWL framework with a more capable implementation.
    """
    
    def __init__(self, agents):
        """
        Initialize the agent pipeline with a list of agents.
        
        Args:
            agents (list): List of agent objects to process in sequence
        """
        self.agents = agents
        self.schema_parser = None
        
        # Identify the schema parser agent for later use
        for agent in self.agents:
            if hasattr(agent, 'get_schema'):
                self.schema_parser = agent
                break
    
    def process(self, input_data):
        """
        Process input data through the pipeline of agents with enhanced context sharing.
        
        Args:
            input_data (dict): Initial input data
            
        Returns:
            dict: Final output data after processing through all agents
        """
        current_data = input_data.copy()
        
        # Get the database dialect if available
        db_dialect = "unknown"
        if self.schema_parser and hasattr(self.schema_parser, 'db_engine'):
            try:
                db_dialect = self.schema_parser.db_engine.dialect.name
            except:
                pass
        
        # Add db_dialect to the data
        current_data["db_dialect"] = db_dialect
        
        # Get schema information if not already present
        if self.schema_parser and "schema_info" not in current_data:
            try:
                # Get the schema with analysis if needed
                schema_info = self.schema_parser.get_schema(force_refresh=False, analyze_data=False)
                current_data["schema_info"] = schema_info
            except Exception as e:
                print(f"Error getting schema: {str(e)}")
        
        # Process through each agent
        for agent in self.agents:
            current_data = agent.process(current_data)
        
        return current_data

def create_agent_pipeline(schema, db_engine):
    """
    Create and configure the agent pipeline with the four specialized agents.
    
    Args:
        schema (str): The SQL schema as a string (used only for backward compatibility)
        db_engine: SQLAlchemy engine for database access
        
    Returns:
        AgentPipeline: Configured pipeline ready to process queries
    """
    # Initialize the agents
    # Use DynamicSchemaParser for automatic schema detection
    schema_parser = DynamicSchemaParser(db_engine)
    query_translator = QueryTranslator()
    sql_executor = SQLExecutor(db_engine)
    responder = Responder()
    
    # Create the agent pipeline
    pipeline = AgentPipeline(
        agents=[schema_parser, query_translator, sql_executor, responder]
    )
    
    return pipeline

def process_query(pipeline, query):
    """
    Process a natural language query through the agent pipeline.
    
    Args:
        pipeline (AgentPipeline): The configured agent pipeline
        query (str): Natural language query from the user
        
    Returns:
        str: Human-readable response to the query
    """
    # Start the pipeline with the user query
    input_data = {"user_query": query}
    result = pipeline.process(input_data)
    
    # Return the final response
    return result["response"]

def main():
    """Main function to run the SQL Conversational Agent."""
    parser = argparse.ArgumentParser(description="Universal SQL Conversational Agent")
    parser.add_argument("--query", type=str, help="Natural language query to process")
    parser.add_argument("--schema_file", type=str, help="Path to SQL schema file")
    parser.add_argument("--db_path", type=str, default="database/sample.db", 
                        help="Path to SQLite database file")
    args = parser.parse_args()
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please create a .env file with your OpenAI API key.")
        return
    
    # Setup the sample database if it doesn't exist
    db_path = args.db_path
    setup_sample_database(db_path)
    db_engine = get_db_engine(db_path)
    
    # Get the schema
    if args.schema_file:
        with open(args.schema_file, 'r') as f:
            schema = f.read()
    else:
        # Use default schema
        schema = """
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name VARCHAR(100),
            hire_date DATE,
            salary FLOAT
        );
        """
    
    # Create the agent pipeline
    pipeline = create_agent_pipeline(schema, db_engine)
    
    # Process the query
    if args.query:
        query = args.query
    else:
        query = "Who earns more than 50,000?"
        print(f"Using default query: '{query}'")
    
    print("\nProcessing query...")
    response = process_query(pipeline, query)
    
    print("\nQuery:", query)
    print("Response:", response)

if __name__ == "__main__":
    main()
