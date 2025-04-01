"""
Streamlit UI for the Universal SQL Conversational Agent.
Provides a simple web interface for interacting with the agent.
"""

import os
import sys
import io
import time
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our custom UI components

# Clear all caches on startup
st.cache_data.clear()
print("All caches cleared on startup")

# Reset session state for database connections
if 'current_db_connection' in st.session_state:
    del st.session_state.current_db_connection
    print("Reset current database connection")
    
if 'previous_db_connection' in st.session_state:
    del st.session_state.previous_db_connection
    print("Reset previous database connection")
    
if 'generated_questions' in st.session_state:
    del st.session_state.generated_questions
    print("Reset generated questions")
from ui.status_manager import StatusManager
from ui.profile_manager import ProfileManager

from agents.schema_parser import SchemaParser, DynamicSchemaParser
from agents.query_translator import QueryTranslator
from agents.sql_executor import SQLExecutor
from agents.responder import Responder
from database.db_setup import setup_sample_database, get_db_engine
from sqlalchemy import inspect, MetaData, Table
from sql_agent import AgentPipeline

# Global pipeline variable
pipeline = None

# Function to reset pipeline when database connection changes
def reset_pipeline_on_db_change():
    # Clear pipeline and force refresh on next use
    global pipeline
    print(f"[Reset Pipeline] Resetting pipeline. Current global pipeline: {pipeline}") # DEBUG LOG
    pipeline = None
    if 'cached_schema' in st.session_state:
        st.session_state.cached_schema = None
    print("[Reset Pipeline] Pipeline and cached schema reset.") # DEBUG LOG

# Function to detect tables in a database using DynamicSchemaParser
def detect_database_tables(engine, force_refresh=False, analyze_data=True):
    """Detect all tables in the connected database and return their structure using DynamicSchemaParser.
    
    Args:
        engine: SQLAlchemy engine for database access
        force_refresh: Whether to force a refresh of the schema cache
        analyze_data: Whether to perform data pattern analysis
        
    Returns:
        tuple: (schema_info, tables) - Complete schema information and list of table names
    """
    try:
        # Use our enhanced DynamicSchemaParser for schema detection
        schema_parser = DynamicSchemaParser(engine)
        schema_info = schema_parser.get_schema(force_refresh=force_refresh, analyze_data=analyze_data)
        
        # Extract table names from schema_info, excluding special keys
        tables = [table_name for table_name in schema_info.keys() 
                 if table_name not in ["database_type", "data_patterns"]]
        
        return schema_info, tables
    except Exception as e:
        st.sidebar.error(f"Error detecting tables: {str(e)}")
        return {}, []

# Function to generate example questions based on the database schema using Gemini API
def generate_example_questions(schema_info, tables):
    """Generate relevant example questions based on the database schema using Gemini API.
    
    Args:
        schema_info: Dictionary containing schema information
        tables: List of table names in the database
        
    Returns:
        list: List of example questions relevant to the database schema
    """
    # Generic business-friendly questions as fallback
    default_questions = [
        "How many records do we have in each table?",
        "What are the most common values in our main categories?",
        "Which items appear most frequently in our database?",
        "What's the average value for our main metrics?",
        "Can you show me the distribution of data across categories?",
        "What are the relationships between our main data entities?"
    ]
    
    # If no tables, return default questions
    if not tables or not schema_info:
        return default_questions
    
    try:
        # Import Gemini API
        import google.generativeai as genai
        import json
        import time
        
        # Configure Gemini API
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("No Gemini API key found. Using default questions.")
            return default_questions
            
        genai.configure(api_key=api_key)
        
        # Create a simplified schema representation for the prompt
        simplified_schema = {}
        for table in tables:
            if table in schema_info:
                simplified_schema[table] = {
                    "columns": {}
                }
                
                # Add column information
                for column, info in schema_info[table].get('columns', {}).items():
                    simplified_schema[table]["columns"][column] = {
                        "type": info.get('type', 'unknown')
                    }
                
                # Add foreign keys if available
                if 'foreign_keys' in schema_info[table]:
                    simplified_schema[table]["foreign_keys"] = schema_info[table]['foreign_keys']
                
                # Add primary key if available
                if 'primary_key' in schema_info[table]:
                    simplified_schema[table]["primary_key"] = schema_info[table]['primary_key']
        
        # Extract table names and their columns for a more targeted prompt
        table_details = []
        for table in tables:
            if table in schema_info:
                columns = list(schema_info[table].get('columns', {}).keys())
                table_details.append(f"- {table}: {', '.join(columns)}")
        
        table_summary = "\n".join(table_details)
        
        # Create a focused prompt for Gemini that's specific to the actual tables
        prompt = f"""
        You are a business analyst helping non-technical executives understand their data. 
        Based on the following database schema, generate 6 natural language questions that a business executive might want to ask.
        
        Database Tables and Columns:
        {table_summary}
        
        Detailed Schema (JSON format):
        {json.dumps(simplified_schema, indent=2)}
        
        Rules for generating questions:
        1. Questions MUST ONLY reference tables and columns that actually exist in the schema above
        2. Questions must be directly relevant to the specific tables in this database
        3. DO NOT use technical column names directly - instead use business-friendly terms
        4. DO NOT mention table names directly - instead refer to business concepts
        5. Include questions about relationships between entities if multiple related tables exist
        6. Include a mix of questions about: counts, top items, averages, totals, comparisons
        7. Make questions concise, clear and focused on business insights
        8. NEVER use SQL terms or database jargon
        9. Focus on questions that would help make business decisions
        10. Tailor questions to the specific domain suggested by the table names
        11. IMPORTANT: ONLY use examples from the actual tables in the schema, do not invent tables or columns
        
        Examples of good business-friendly questions (ONLY use if relevant to the actual schema):
        - "How many items do we currently have in our database?"
        - "Which category has the most entries?"
        - "What are our top 5 most common values?"
        - "How is data distributed across our main categories?"
        - "What patterns can we see in our data?"
        
        Return exactly 6 questions as a JSON array of strings. ONLY return the JSON array, nothing else.
        """
        
        # Generate questions using Gemini
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt, generation_config={
            "temperature": 0.2,  # Lower temperature for more focused responses
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 1024,
        })
        
        # Extract questions from the response
        response_text = response.text
        
        # Try to parse the JSON response
        try:
            # Clean up the response text to extract just the JSON array
            import re
            
            # Find anything that looks like a JSON array
            json_match = re.search(r'\[\s*".*"\s*(,\s*".*"\s*)*\]', response_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                questions = json.loads(json_str)
                
                # Ensure we have exactly 6 questions
                if len(questions) > 6:
                    questions = questions[:6]
                while len(questions) < 6:
                    questions.append(default_questions[len(questions) % len(default_questions)])
                    
                # Cache the generated questions with a timestamp
                if 'generated_questions' not in st.session_state:
                    st.session_state.generated_questions = {}
                
                question_key = ",".join(sorted(tables))
                st.session_state.generated_questions[question_key] = {
                    "questions": questions,
                    "timestamp": time.time()
                }
                
                return questions
            else:
                # Try to extract line by line
                lines = response_text.split('\n')
                questions = []
                
                for line in lines:
                    # Remove list markers, quotes and leading/trailing whitespace
                    clean_line = line.strip()
                    clean_line = re.sub(r'^\d+\.\s*', '', clean_line)  # Remove leading numbers like "1. "
                    clean_line = re.sub(r'^[\"\']|[\"\']$', '', clean_line)  # Remove quotes
                    
                    if clean_line and '```' not in clean_line and len(clean_line) > 10:
                        questions.append(clean_line)
                
                # Filter and ensure we have exactly 6 questions
                questions = [q for q in questions if len(q.split()) > 3]  # Filter out very short lines
                if len(questions) > 6:
                    questions = questions[:6]
                while len(questions) < 6:
                    questions.append(default_questions[len(questions) % len(default_questions)])
                
                # Cache the generated questions
                if 'generated_questions' not in st.session_state:
                    st.session_state.generated_questions = {}
                
                question_key = ",".join(sorted(tables))
                st.session_state.generated_questions[question_key] = {
                    "questions": questions,
                    "timestamp": time.time()
                }
                
                return questions
                
        except Exception as e:
            print(f"Error parsing Gemini response: {str(e)}")
            print(f"Response was: {response_text}")
            return default_questions
    
    except Exception as e:
        print(f"Error generating questions with Gemini: {str(e)}")
        return default_questions

# Load environment variables
load_dotenv()

# Check for Gemini API key
if not os.getenv("GEMINI_API_KEY"):
    st.error("Gemini API key not found. Please set the GEMINI_API_KEY environment variable.")
    st.stop()

# Set up the page
st.set_page_config(
    page_title="Universal SQL Conversational Agent",
    page_icon="🤖",
    layout="wide"
)

# Hide schema analysis messages using custom CSS
st.markdown("""
<style>
    [data-testid="stSuccessMessage"] {display: none !important;}
</style>
""", unsafe_allow_html=True)

# Initialize the status manager and profile manager
status_manager = StatusManager()
profile_manager = ProfileManager()

# Create a container for status messages - this is the ONLY place where status messages will appear
status_container = st.empty()

# Set the container for the status manager
status_manager.set_container(status_container)

# Initialize session state for conversation history and schema caching
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'last_query_results' not in st.session_state:
    st.session_state.last_query_results = None
    
if 'last_sql_query' not in st.session_state:
    st.session_state.last_sql_query = None
    
if 'conversation_mode' not in st.session_state:
    st.session_state.conversation_mode = True
    
# Schema caching for better performance
if 'cached_schema' not in st.session_state:
    st.session_state.cached_schema = None
    
if 'schema_last_updated' not in st.session_state:
    st.session_state.schema_last_updated = None
    
if 'current_db_connection' not in st.session_state:
    st.session_state.current_db_connection = None

# We're not using schema status tracking anymore - removing all schema status messages from the UI

# Initialize db_engine from session state or set to None
if 'db_engine' not in st.session_state:
    st.session_state.db_engine = None

# Use the database engine from session state
db_engine = st.session_state.db_engine

st.title("Universal SQL Conversational Agent")
st.markdown("""
Ask questions about your SQL database in plain English.
This agent will translate your questions into SQL queries and provide human-readable answers.
""")

# Create a container for status messages - this is the ONLY place where status messages will appear
status_container = st.empty()

# Sidebar for database and schema configuration
st.sidebar.header("Configuration")

# Conversation mode toggle
st.sidebar.header("Conversation Settings")
conversation_mode = st.sidebar.toggle(
    "Enable Conversation Mode",
    value=st.session_state.conversation_mode,
    help="When enabled, the agent will remember previous questions and answers to provide context for follow-up questions."
)
# Update session state with the new value
st.session_state.conversation_mode = conversation_mode

# Database selection
st.sidebar.header("Database Connection")
db_option = st.sidebar.radio(
    "Database",
    ["Use sample SQLite database", "Connect to custom database", "Connect via URL", "Saved Profiles"]
)

if db_option == "Use sample SQLite database":
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database", "sample.db")
    
    # Track if the database connection has changed
    db_connection_changed = False
    if 'previous_db_connection' not in st.session_state:
        st.session_state.previous_db_connection = None
    
    if st.session_state.previous_db_connection != db_path:
        db_connection_changed = True
        st.session_state.previous_db_connection = db_path
        # Force regeneration of questions when database changes
        st.session_state.force_regenerate_questions = True
        print("Database connection changed, will regenerate questions")
    
    # Clear any existing connection first
    if 'db_engine' in st.session_state:
        st.session_state.db_engine = None
    
    # Set up the sample database if it doesn't exist
    setup_sample_database(db_path)
    db_engine = get_db_engine(db_path)
    print(f"[Connect Sample DB] Setting session state db_engine: {db_engine}") # DEBUG LOG
    st.session_state.db_engine = db_engine
    st.session_state.current_db_connection = db_path
    
    # Reset the pipeline to use the new connection
    reset_pipeline_on_db_change()
    
    status_manager.connection_success("SQLite", "Sample Database")
    
    # Detect tables in the database
    schema_info, tables = detect_database_tables(db_engine)
    
    # Display database structure
    st.sidebar.subheader("Database Structure")
    if tables:
        table_info = """
**Tables:**
"""
        for table in tables:
            table_info += f"- {table}: "
            if table in schema_info:
                column_count = len(schema_info[table]['columns'])
                table_info += f"{column_count} columns"  
            table_info += "\n"
        st.sidebar.markdown(table_info)
    else:
        st.sidebar.markdown("""
        **Tables:**
        No tables detected. Please refresh the schema.
        """)
elif db_option == "Saved Profiles":
    # Load and display saved profiles
    profiles = profile_manager.get_profiles()
    
    if not profiles:
        st.sidebar.info("No saved profiles found. Connect to a database and save it as a profile first.")
    else:
        selected_profile = st.sidebar.selectbox("Select a profile", profiles)
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("Connect", key="connect_profile"):
                success, message, engine = profile_manager.connect_to_profile(selected_profile)
                if success:
                    # Clear any existing connection first
                    if 'db_engine' in st.session_state:
                        st.session_state.db_engine = None
                    
                    # Set the new database engine
                    db_engine = engine
                    st.session_state.db_engine = db_engine
                    print(f"[Connect Profile] Setting session state db_engine: {db_engine} for profile {selected_profile}") # DEBUG LOG
                    
                    # Reset the pipeline to use the new connection
                    reset_pipeline_on_db_change()
                    
                    # Track database connection changes
                    db_connection_changed = False
                    if 'previous_db_connection' not in st.session_state:
                        st.session_state.previous_db_connection = None
                    
                    if st.session_state.previous_db_connection != selected_profile:
                        db_connection_changed = True
                        st.session_state.previous_db_connection = selected_profile
                        # Force regeneration of questions when database changes
                        st.session_state.force_regenerate_questions = True
                        print("Database connection changed, will regenerate questions")
                    
                    st.session_state.current_db_connection = selected_profile
                    status_manager.connection_success("Database", f"Profile: {selected_profile}")
                    # Force schema refresh when connecting to a new profile
                    st.session_state.cached_schema = None
                    
                    # Detect and display tables in the database
                    try:
                        schema_info, tables = detect_database_tables(db_engine)
                        
                        # Display database structure
                        st.sidebar.subheader("Database Structure")
                        if tables:
                            table_info = """
**Tables:**
"""
                            for table in tables:
                                table_info += f"- {table}: "
                                if table in schema_info:
                                    column_count = len(schema_info[table]['columns'])
                                    table_info += f"{column_count} columns"  
                                table_info += "\n"
                            st.sidebar.markdown(table_info)
                        else:
                            st.sidebar.info("No tables found in this database.")
                    except Exception as e:
                        st.sidebar.error(f"Error detecting tables: {str(e)}")
                else:
                    status_manager.connection_failure(message)
        
        with col2:
            if st.button("Delete", key="delete_profile"):
                success, message = profile_manager.delete_profile(selected_profile)
                if success:
                    st.sidebar.success(message)
                    st.rerun()  # Refresh the page to update the profile list
                else:
                    st.sidebar.error(message)

elif db_option == "Connect to custom database":
    # Custom database connection
    db_type = st.sidebar.selectbox(
        "Database Type",
        ["SQLite", "PostgreSQL", "MySQL", "SQL Server", "Oracle"]
    )
    
    if db_type == "SQLite":
        uploaded_file = st.sidebar.file_uploader("Upload SQLite Database", type=["db", "sqlite", "sqlite3"])
        if uploaded_file is not None:
            # Save the uploaded file to a temporary location
            db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database", "uploaded.db")
            with open(db_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            db_engine = get_db_engine(db_path)
            st.session_state.db_engine = db_engine
            st.sidebar.success(f"Database uploaded successfully: {uploaded_file.name}")
            
            # Detect tables in the uploaded database - don't show schema status message here
            schema_info, tables = detect_database_tables(db_engine)
            
            # Display database structure
            st.sidebar.subheader("Database Structure")
            if tables:
                table_info = """
**Tables:**
"""
                for table in tables:
                    table_info += f"- {table}: "
                    if table in schema_info:
                        column_count = len(schema_info[table]['columns'])
                        table_info += f"{column_count} columns"  
                    table_info += "\n"
                st.sidebar.markdown(table_info)
            else:
                st.sidebar.info("No tables found in this database.")
        else:
            db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database", "sample.db")
            setup_sample_database(db_path)
            db_engine = get_db_engine(db_path)
            st.session_state.db_engine = db_engine
    else:
        # For other database types
        with st.sidebar.form(key=f"{db_type.lower()}_connection_form"):
            st.subheader(f"Connect to {db_type}")
            
            # Default ports for different database types
            default_ports = {
                "PostgreSQL": "5432",
                "MySQL": "3306",
                "SQL Server": "1433",
                "Oracle": "1521"
            }
            
            # Connection parameters
            host = st.text_input("Host", "localhost")
            port = st.text_input("Port", default_ports.get(db_type, ""))
            database = st.text_input("Database Name")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            # Additional parameters for specific databases
            if db_type == "Oracle":
                sid = st.text_input("SID/Service Name")
            
            # Add option to save this connection as a profile
            save_profile = st.checkbox("Save as profile")
            profile_name = ""
            if save_profile:
                profile_name = st.text_input("Profile Name", placeholder="My Database Profile")
            
            # Connect button
            connect_submitted = st.form_submit_button("Connect")
        
        if connect_submitted:
            try:
                # Build connection string based on database type
                if db_type == "PostgreSQL":
                    # URL encode the username and password to handle special characters
                    import urllib.parse
                    encoded_username = urllib.parse.quote_plus(username)
                    encoded_password = urllib.parse.quote_plus(password)
                    
                    # Handle database name - PostgreSQL is case-sensitive
                    # If no database is specified, try to connect to 'postgres' (default)
                    db_name = database.strip() if database.strip() else "postgres"
                    
                    # Build the connection string
                    connection_string = f"postgresql://{encoded_username}:{encoded_password}@{host}:{port}/{db_name}"
                elif db_type == "MySQL":
                    # URL encode the username and password to handle special characters
                    import urllib.parse
                    encoded_username = urllib.parse.quote_plus(username)
                    encoded_password = urllib.parse.quote_plus(password)
                    connection_string = f"mysql+mysqlconnector://{encoded_username}:{encoded_password}@{host}:{port}/{database}"
                elif db_type == "SQL Server":
                    # URL encode the username and password to handle special characters
                    import urllib.parse
                    encoded_username = urllib.parse.quote_plus(username)
                    encoded_password = urllib.parse.quote_plus(password)
                    connection_string = f"mssql+pymssql://{encoded_username}:{encoded_password}@{host}:{port}/{database}"
                elif db_type == "Oracle":
                    # URL encode the username and password to handle special characters
                    import urllib.parse
                    encoded_username = urllib.parse.quote_plus(username)
                    encoded_password = urllib.parse.quote_plus(password)
                    encoded_sid = urllib.parse.quote_plus(sid)
                    connection_string = f"oracle+cx_oracle://{encoded_username}:{encoded_password}@{host}:{port}/?service_name={encoded_sid}"
                
                # Clear any existing connection first
                if 'db_engine' in st.session_state:
                    st.session_state.db_engine = None
                
                # Create engine
                from sqlalchemy import create_engine
                db_engine = create_engine(connection_string)
                print(f"[Connect Custom DB] Setting session state db_engine: {db_engine}") # DEBUG LOG
                st.session_state.db_engine = db_engine
                
                # Reset the pipeline to use the new connection
                reset_pipeline_on_db_change()
                
                # Test connection
                with db_engine.connect() as conn:
                    # Show success message for database connection
                    status_manager.connection_success(db_type, f"{host}:{port}/{database}")
                    st.session_state.current_db_connection = connection_string
                    db_path = connection_string
                    
                    # Save as profile if requested
                    if save_profile and profile_name:
                        connection_info = {
                            "type": db_type.lower(),
                            "host": host,
                            "port": port,
                            "database": database,
                            "username": username,
                            "password": password
                        }
                        success, message = profile_manager.save_profile(profile_name, connection_info)
                        if success:
                            st.sidebar.success(message)
                        else:
                            st.sidebar.error(message)
                    
                    # Detect tables in the database
                    schema_info, tables = detect_database_tables(db_engine)
                    if tables:
                        st.sidebar.subheader("Detected Tables")
                        st.sidebar.write(f"Found {len(tables)} tables in the database:")
                        for table in tables:
                            st.sidebar.markdown(f"- {table}")
            except Exception as e:
                st.sidebar.error(f"Connection failed: {str(e)}")
                # Don't automatically fall back to sample database
                st.error("Please check your database connection settings and try again.")
                db_engine = None
                st.session_state.db_engine = None
        else:
            # Only use the sample database if the user explicitly selected it
            st.error("Please enter valid database credentials and click Connect.")
            db_engine = None
            st.session_state.db_engine = None
elif db_option == "Connect via URL":
    # Connection via URL
    connection_url = st.sidebar.text_input("Connection URL", placeholder="dialect+driver://username:password@host:port/database")
    
    # Add option to save this connection as a profile
    save_profile = st.sidebar.checkbox("Save as profile")
    if save_profile:
        profile_name = st.sidebar.text_input("Profile Name", placeholder="My Database Profile")
    
    if st.sidebar.button("Connect via URL"):
        if connection_url:
            try:
                # Clear any existing connection first
                if 'db_engine' in st.session_state:
                    st.session_state.db_engine = None
                
                from sqlalchemy import create_engine
                db_engine = create_engine(connection_url)
                st.session_state.db_engine = db_engine
                print(f"[Connect URL] Setting session state db_engine: {db_engine}") # DEBUG LOG
                
                # Reset the pipeline to use the new connection
                reset_pipeline_on_db_change()
                
                # Test connection
                with db_engine.connect() as conn:
                    # Extract the database type from the URL
                    db_type = connection_url.split("://")[0].split("+")[0].upper()
                    status_manager.connection_success(db_type, "URL connection")
                    st.session_state.current_db_connection = connection_url
                    db_path = connection_url
                    
                    # Save as profile if requested
                    if save_profile and profile_name:
                        connection_info = {
                            "type": "url",
                            "url": connection_url
                        }
                        success, message = profile_manager.save_profile(profile_name, connection_info)
                        if success:
                            st.sidebar.success(message)
                        else:
                            st.sidebar.error(message)
                
                # Detect tables in the database
                schema_info, tables = detect_database_tables(db_engine)
                
                # Store in session state for persistence
                st.session_state.schema_info = schema_info
                st.session_state.tables = tables
                
                # Display minimal information about connected database in sidebar only
                if tables:
                    st.sidebar.success(f"Connected to database with {len(tables)} tables")
            except Exception as e:
                st.sidebar.error(f"Connection failed: {str(e)}")
                # Don't automatically fall back to sample database
                st.error("Please check your database connection settings and try again.")
                db_engine = None
                st.session_state.db_engine = None
    else:
        # Only use the sample database if the user explicitly selected it
        st.error("Please enter a valid database URL and click Connect.")
        db_engine = None
        st.session_state.db_engine = None

# Automatic schema detection - no UI elements needed for schema configuration
# The schema will be automatically detected from the connected database

# Initialize session state variables if they don't exist
if 'schema_info' not in st.session_state:
    st.session_state.schema_info = {}
    
if 'tables' not in st.session_state:
    st.session_state.tables = []
    
if 'current_db_connection' not in st.session_state:
    st.session_state.current_db_connection = ''
    
if 'cached_schema' not in st.session_state:
    st.session_state.cached_schema = None
    
if 'schema_last_updated' not in st.session_state:
    st.session_state.schema_last_updated = None

# Update schema info based on detected database tables
if db_engine:
    try:
        # IMPORTANT: We're completely disabling schema status messages here
        # They will be shown only once in specific places (like when connecting to a database)
        
        # Detect tables in the database
        schema_info, tables = detect_database_tables(db_engine)
        
        # Store in session state for persistence
        st.session_state.schema_info = schema_info
        st.session_state.tables = tables
    except Exception as e:
        status_manager.schema_failure(str(e))
else:
    # If no database engine is available, show a warning
    st.warning("No database connection established. Please connect to a database first.")
    # Clear any existing schema info
    st.session_state.schema_info = {}
    st.session_state.tables = []

# Create the agent pipeline
class AgentPipeline:
    """
    A simple pipeline to orchestrate multiple agents in sequence.
    """
    
    def __init__(self, agents):
        """
        Initialize the agent pipeline with a list of agents.
        
        Args:
            agents (list): List of agent objects to process in sequence
        """
        self.agents = agents
    
    def process(self, input_data):
        """
        Process input data through the pipeline of agents.
        
        Args:
            input_data (dict): Initial input data
            
        Returns:
            dict: Final output data after processing through all agents
        """
        current_data = input_data
        
        for agent in self.agents:
            current_data = agent.process(current_data)
        
        return current_data

def get_pipeline(_db_engine, force_refresh=False, analyze_data=True):
    """
    Create the agent pipeline with dynamic schema detection.
    Uses cached schema information when available for better performance.
    
    Args:
        _db_engine: SQLAlchemy engine for database access
        force_refresh: Whether to force a refresh of the schema cache
        analyze_data: Whether to perform data pattern analysis
        
    Returns:
        AgentPipeline: Configured pipeline ready to process queries
    """
    # Check if we need to update the schema cache
    connection_string = str(_db_engine.url)
    current_time = pd.Timestamp.now()
    
    # Force refresh or connection changed or cache expired (older than 5 minutes)
    refresh_needed = force_refresh or \
                    st.session_state.current_db_connection != connection_string or \
                    st.session_state.cached_schema is None or \
                    (st.session_state.schema_last_updated is not None and \
                     (current_time - st.session_state.schema_last_updated).total_seconds() > 300)
    
    # Initialize the dynamic schema parser
    dynamic_schema_parser = DynamicSchemaParser(_db_engine)
    
    # Update the schema cache if needed
    if refresh_needed:
        with st.spinner("Analyzing database schema..."):
            # Detect schema and update cache with specified analysis level
            schema_info = dynamic_schema_parser.get_schema(force_refresh=True, analyze_data=analyze_data)
            st.session_state.cached_schema = schema_info
            st.session_state.schema_last_updated = current_time
            st.session_state.current_db_connection = connection_string
            
            # Display a dynamic success message with details about the schema
            table_count = len([t for t in schema_info.keys() if t not in ['database_type', 'data_patterns']])
            status_manager.schema_success(table_count)
    else:
        # Use cached schema
        dynamic_schema_parser.schema_info = st.session_state.cached_schema
    
    # Initialize other agents
    query_translator = QueryTranslator()
    sql_executor = SQLExecutor(_db_engine)
    responder = Responder()
    
    # Create the agent pipeline
    pipeline = AgentPipeline(
        agents=[dynamic_schema_parser, query_translator, sql_executor, responder]
    )
    
    return pipeline

# Add schema analysis options in the sidebar
with st.sidebar.expander("Schema Analysis Options", expanded=False):
    st.subheader("Schema Detection Settings")
    
    # Schema refresh options
    col1, col2 = st.columns(2)
    with col1:
        refresh_schema = st.button("Refresh Schema Cache")
    with col2:
        deep_analysis = st.checkbox("Deep Data Analysis", value=True, 
                                   help="Analyzes data patterns and relationships more thoroughly. May be slower for large databases.")
    
    # Schema visualization options
    st.subheader("Schema Visualization")
    show_schema_details = st.checkbox("Show Schema Details", value=False)
    
    if show_schema_details and 'current_db_connection' in st.session_state and st.session_state.current_db_connection:
        try:
            # Get the schema with current settings
            schema_info, tables = detect_database_tables(db_engine, force_refresh=refresh_schema, analyze_data=deep_analysis)
            if not schema_info:
                status_manager.schema_failure("No schema information returned")
            
            if schema_info:
                # Display database type
                db_type = schema_info.get('database_type', 'Unknown')
                st.info(f"Database Type: {db_type.upper()}")
                
                # Create tabs for different schema views
                schema_tabs = st.tabs(["Tables", "Relationships", "Data Patterns"])
                
                # Tables tab
                with schema_tabs[0]:
                    for table in sorted(tables):
                        with st.expander(f"📋 {table}"):
                            # Show columns
                            if 'columns' in schema_info[table]:
                                st.markdown("**Columns:**")
                                cols_data = []
                                for col_name, col_info in schema_info[table]['columns'].items():
                                    pk_status = "✓" if col_name in schema_info[table].get('primary_keys', []) else ""
                                    cols_data.append({
                                        "Column": col_name,
                                        "Type": col_info['type'],
                                        "Primary Key": pk_status,
                                        "Nullable": "Yes" if col_info.get('nullable', True) else "No"
                                    })
                                st.dataframe(pd.DataFrame(cols_data), hide_index=True)
                
                # Relationships tab
                with schema_tabs[1]:
                    # Display all foreign keys
                    all_relationships = []
                    for table in tables:
                        for fk in schema_info[table].get('foreign_keys', []):
                            source_cols = ", ".join(fk['columns'])
                            target_cols = ", ".join(fk['referred_columns'])
                            confidence = fk.get('confidence', 'High') if fk.get('inferred', False) else "Defined"
                            all_relationships.append({
                                "Source Table": table,
                                "Source Columns": source_cols,
                                "Target Table": fk['referred_table'],
                                "Target Columns": target_cols,
                                "Type": "Inferred" if fk.get('inferred', False) else "Defined",
                                "Confidence": confidence
                            })
                    
                    if all_relationships:
                        st.dataframe(pd.DataFrame(all_relationships), hide_index=True)
                    else:
                        st.info("No relationships detected in the database schema.")
                
                # Data Patterns tab
                with schema_tabs[2]:
                    if 'data_patterns' in schema_info and schema_info['data_patterns']:
                        for pattern in schema_info['data_patterns']:
                            if pattern['type'] == 'schema_pattern' and pattern['pattern'] == 'star_schema':
                                st.success("⭐ Star Schema Detected")
                                st.markdown(f"**Fact Tables:** {', '.join(pattern['fact_tables'])}")
                                st.markdown(f"**Dimension Tables:** {', '.join(pattern['dimension_tables'])}")
                    
                    # Show denormalized patterns if any
                    denorm_found = False
                    for table in tables:
                        if 'denormalized_patterns' in schema_info[table] and schema_info[table]['denormalized_patterns']:
                            if not denorm_found:
                                st.subheader("Denormalized Structures")
                                denorm_found = True
                            
                            st.markdown(f"**Table: {table}**")
                            for key, pattern in schema_info[table]['denormalized_patterns'].items():
                                if isinstance(pattern, dict):
                                    st.markdown(f"- {key}: {pattern.get('type', 'Unknown pattern')}")
            else:
                st.warning("No schema information available. Try refreshing the schema.")
                
        except Exception as e:
            st.error(f"Error displaying schema details: {str(e)}")
    
    if refresh_schema:
        st.session_state.cached_schema = None  # Force schema refresh
        status_manager.schema_refresh_initiated()

# Get the pipeline with dynamic schema detection
# Force refresh when database connection changes or when explicitly requested
force_refresh_needed = (
    (refresh_schema if 'refresh_schema' in locals() else False) or
    db_connection_changed if 'db_connection_changed' in locals() else False
)
print(f"Pipeline initialization - Force refresh: {force_refresh_needed}")
# Only initialize the pipeline if we have a valid database engine
if db_engine:
    # Update the current database connection identifier for change detection
    # Use a unique identifier based on the database connection details
    if hasattr(db_engine, 'url'):
        st.session_state.current_db_connection = str(db_engine.url)
    else:
        st.session_state.current_db_connection = str(id(db_engine))
    
    print(f"[Pipeline Initialization] Initializing pipeline with DB Engine: {db_engine}") # DEBUG LOG
    pipeline = get_pipeline(db_engine, force_refresh=force_refresh_needed, analyze_data=deep_analysis if 'deep_analysis' in locals() else True)
    print(f"[Pipeline Initialization] Pipeline initialized: {pipeline}") # DEBUG LOG
else:
    pipeline = None
    st.session_state.current_db_connection = None

# Track the current database connection to detect changes
if 'previous_db_connection' not in st.session_state:
    st.session_state.previous_db_connection = None

# Check if database connection has changed
db_connection_changed = False
if 'current_db_connection' in st.session_state and st.session_state.current_db_connection != st.session_state.previous_db_connection:
    print(f"Database connection changed from {st.session_state.previous_db_connection} to {st.session_state.current_db_connection}")
    db_connection_changed = True
    st.session_state.previous_db_connection = st.session_state.current_db_connection
    # We'll clear the cache after the function is defined

# Function to get example questions (without caching to ensure fresh results)
def get_example_questions(schema_info_str, tables_str):
    """Get example questions with caching to prevent regeneration on every UI interaction.
    
    Args:
        schema_info_str: String representation of schema_info for cache key
        tables_str: String representation of tables for cache key
        
    Returns:
        list: List of example questions
    """
    # Default questions if we can't generate schema-specific ones
    default_questions = [
        "How many records do we have in each table?",
        "What are the most common values in our main categories?",
        "Which items appear most frequently in our database?",
        "What's the average value for our main metrics?",
        "Can you show me the distribution of data across categories?",
        "What are the relationships between our main data entities?"
    ]
    
    try:
        # Convert string representations back to objects
        import json
        schema_info = json.loads(schema_info_str)
        tables = json.loads(tables_str)
        
        # Log the schema and tables for debugging
        print(f"Schema info for question generation: {json.dumps(schema_info, indent=2)}")
        print(f"Tables for question generation: {tables}")
        
        # Generate questions
        questions = generate_example_questions(schema_info, tables)
        
        # Log the generated questions
        print(f"Generated questions from API: {questions}")
        
        # If generation failed, use defaults
        if not questions or len(questions) < 6:
            print("Question generation failed or returned too few questions, using defaults")
            return default_questions
            
        return questions
    except Exception as e:
        print(f"Error generating cached questions: {str(e)}")
        return default_questions

# Main interface
st.header("Ask a Question")

# Generate example questions based on the current database schema
col1, col2 = st.columns([4, 1])
with col1:
    st.subheader("Example questions:")
with col2:
    refresh_questions = st.button("🔄 Refresh", help="Refresh example questions based on current database schema")
    if refresh_questions:
        # Set a flag to force regeneration of questions
        st.session_state.force_regenerate_questions = True
        print("Example questions will be regenerated due to refresh button click")
        # Force a page rerun to regenerate questions
        st.rerun()

# Always force a fresh schema detection for example questions
try:
    # Force schema detection to get the latest tables
    print("Detecting database schema for example questions...")
    # Use a separate detection to ensure we get the latest schema
    schema_info_for_questions, tables_for_questions = detect_database_tables(db_engine, force_refresh=True)
    print(f"Detected tables for questions: {tables_for_questions}")
    
    # Override the existing schema_info and tables variables
    schema_info = schema_info_for_questions
    tables = tables_for_questions
    
    # Log the schema detection for debugging
    print(f"Schema detection complete. Found {len(tables)} tables: {tables}")
    
    # If no tables were found, try again with deeper analysis
    if not tables:
        print("No tables found, trying again with deeper analysis...")
        schema_info, tables = detect_database_tables(db_engine, force_refresh=True, analyze_data=True)
        print(f"Second attempt detected {len(tables)} tables: {tables}")
        
except Exception as e:
    print(f"Error detecting schema for questions: {str(e)}")
    schema_info, tables = {}, []

# Initialize force regeneration flag if not present
if 'force_regenerate_questions' not in st.session_state:
    st.session_state.force_regenerate_questions = True

# Store generated questions in session state to avoid regeneration on each rerun
if 'generated_example_questions' not in st.session_state:
    st.session_state.generated_example_questions = None

# Initialize example questions
if schema_info and tables:
    # Check if we need to regenerate questions
    regenerate_needed = (
        st.session_state.force_regenerate_questions or 
        st.session_state.generated_example_questions is None
    )
    
    if regenerate_needed:
        try:
            # Convert objects to strings
            import json
            schema_info_str = json.dumps(schema_info)
            tables_str = json.dumps(tables)
            
            # Generate fresh questions with a spinner
            placeholder = st.empty()
            with placeholder.container():
                with st.spinner("Generating example questions based on your database schema..."):
                    print("Generating fresh example questions...")
                    example_questions = get_example_questions(schema_info_str, tables_str)
                    # Store in session state to avoid regeneration
                    st.session_state.generated_example_questions = example_questions
                    # Reset the force regeneration flag
                    st.session_state.force_regenerate_questions = False
            placeholder.empty()  # Remove the spinner after questions are generated
        except Exception as e:
            print(f"Error generating fresh questions: {str(e)}")
            # Use default questions as fallback
            example_questions = [
                "How many records do we have in each table?",
                "What are the most common values in our main categories?",
                "Which items appear most frequently in our database?",
                "What's the average value for our main metrics?",
                "Can you show me the distribution of data across categories?",
                "What are the relationships between our main data entities?"
            ]
            st.session_state.generated_example_questions = example_questions
    else:
        # Use the questions we already generated
        print("Using previously generated questions from session state")
        example_questions = st.session_state.generated_example_questions
else:
    # Fallback to default questions if schema info isn't available
    print("No schema info available, using default questions")
    example_questions = [
        "How many records do we have in each table?",
        "What are the most common values in our main categories?",
        "Which items appear most frequently in our database?",
        "What's the average value for our main metrics?",
        "Can you show me the distribution of data across categories?",
        "What are the relationships between our main data entities?"
    ]
    # Store in session state
    st.session_state.generated_example_questions = example_questions

# Create buttons for example questions
cols = st.columns(len(example_questions))

# Use session state to store the selected question
if 'selected_example_question' not in st.session_state:
    st.session_state.selected_example_question = None

# Function to set the selected question in session state without triggering immediate processing
def set_example_question(question):
    st.session_state.selected_example_question = question
    # Don't set query_submitted to True here - we just want to populate the text field
    st.session_state.query_input = question

# Create the buttons that only populate the text field without submitting
for i, col in enumerate(cols):
    col.button(example_questions[i], key=f"example_btn_{i}", on_click=set_example_question, args=(example_questions[i],))

# User input
if 'followup_question' in st.session_state and st.session_state.followup_question:
    initial_value = st.session_state.followup_question
    # Clear it after using it
    st.session_state.followup_question = ""
else:
    initial_value = ""  # Always start with empty input - example questions are handled via session state

# Function to handle query submission
def handle_query_submission():
    if st.session_state.query_input:
        st.session_state.submitted_query = st.session_state.query_input
        st.session_state.query_submitted = True
        


# Initialize session state for query submission
if 'query_submitted' not in st.session_state:
    st.session_state.query_submitted = False
if 'submitted_query' not in st.session_state:
    st.session_state.submitted_query = ""

user_query = st.text_input(
    "Or type your own question:", 
    value=initial_value,
    key="query_input",
    on_change=handle_query_submission
)

# Process the query when the user submits
if st.button("Ask") or st.session_state.query_submitted:
    # Use either the text input or the submitted query
    query_to_process = st.session_state.submitted_query if st.session_state.query_submitted else user_query
    
    # Reset the submitted flag for next time
    if st.session_state.query_submitted:
        st.session_state.query_submitted = False
        st.session_state.submitted_query = ""
    
    if query_to_process:
        # Check if we have a valid database connection
        # Get the latest database engine from session state
        db_engine = st.session_state.db_engine
        print(f"[Query Processing] Using DB Engine from session state: {db_engine}") # DEBUG LOG
        
        # Reset any existing pipeline to ensure it uses the current database connection
        reset_pipeline_on_db_change()
        
        if not db_engine:
            st.error("No database connection established. Please connect to a database before asking questions.")
            st.stop()
            
        with st.spinner("Processing your question..."):
            # Start the pipeline with the user query
            try:
                # Always create a fresh pipeline with the current database engine when processing a query
                print(f"[Query Processing] Initializing pipeline with DB Engine: {db_engine}") # DEBUG LOG
                pipeline = get_pipeline(db_engine, force_refresh=True, analyze_data=True)
                print(f"[Query Processing] Pipeline initialized: {pipeline}") # DEBUG LOG
                
                # Force a schema refresh before processing the query
                # This ensures the query uses the latest schema information
                print("Refreshing schema before processing query...")
                # Use the get_schema method instead of refresh_schema
                pipeline.agents[0].get_schema(force_refresh=True, analyze_data=True)
                
                # Process the query with or without conversation context based on mode
                if st.session_state.conversation_mode:
                    input_data = {
                        "user_query": query_to_process,
                        "conversation_history": st.session_state.conversation_history,
                        "last_query_results": st.session_state.last_query_results,
                        "last_sql_query": st.session_state.last_sql_query
                    }
                else:
                    # Single mode - no conversation context
                    input_data = {
                        "user_query": query_to_process,
                        "conversation_history": [],
                        "last_query_results": None,
                        "last_sql_query": None
                    }
                result = pipeline.process(input_data)
                
                # Get the final response
                natural_response = result["response"]
                
                # Update session state with new conversation context only if in conversation mode
                if st.session_state.conversation_mode:
                    st.session_state.conversation_history = result.get("conversation_history", [])
                    st.session_state.last_query_results = result.get("query_results", [])
                    st.session_state.last_sql_query = result.get("sql_query", "")
                else:
                    # In single mode, we don't update the conversation history
                    pass
                
                # Display dynamic status based on query results
                if "query_results" in result:
                    row_count = len(result["query_results"])
                    status_manager.query_success(row_count)
                elif "error" in result and result["error"]:
                    status_manager.query_failure(result["error"])
                
                # Create columns for the result
                col1, col2 = st.columns([2, 3])
                
                with col1:
                    st.subheader("Your Question")
                    st.write(query_to_process)
                    
                    # Show follow-up suggestions if available
                    if "follow_up_suggestions" in result and result["follow_up_suggestions"]:
                        st.subheader("Follow-up Questions")
                        for suggestion in result["follow_up_suggestions"]:
                            if st.button(f"🔍 {suggestion}", key=f"followup_{suggestion}"):
                                # Store the suggestion to be used as the next query
                                st.session_state.followup_question = suggestion
                                st.rerun()
                
                with col2:
                    st.subheader("Answer")
                    st.markdown(natural_response)
                
                # Show the SQL query and raw results (expandable)
                with st.expander("View SQL Query and Raw Results"):
                    if "sql_query" in result:
                        st.subheader("Generated SQL Query")
                        st.code(result["sql_query"], language="sql")
                        
                        # Show database dialect information
                        if "db_dialect" in result:
                            st.caption(f"Database Type: {result['db_dialect'].upper()}")
                    else:
                        st.info("SQL query information is not available.")
                    
                    if "query_results" in result:
                        st.subheader("Raw Results")
                        # Convert to DataFrame for better display
                        df = pd.DataFrame(result["query_results"])
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("Query results are not available in raw format.")
                
                # Data Visualization Section
                if "query_results" in result and len(result["query_results"]) > 0:
                    with st.expander("Data Visualization", expanded=True):
                        # Convert to DataFrame for better display
                        df = pd.DataFrame(result["query_results"])
                        
                        # Add export options
                        if not df.empty:
                            st.subheader("Export Results")
                            export_col1, export_col2, export_col3 = st.columns(3)
                            
                            with export_col1:
                                # CSV export
                                csv = df.to_csv(index=False)
                                st.download_button(
                                    label="Download CSV",
                                    data=csv,
                                    file_name="query_results.csv",
                                    mime="text/csv"
                                )
                            
                            with export_col2:
                                # Excel export
                                excel_buffer = io.BytesIO()
                                df.to_excel(excel_buffer, index=False, engine="openpyxl")
                                excel_data = excel_buffer.getvalue()
                                st.download_button(
                                    label="Download Excel",
                                    data=excel_data,
                                    file_name="query_results.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                            
                            with export_col3:
                                # JSON export
                                json_str = df.to_json(orient="records")
                                st.download_button(
                                    label="Download JSON",
                                    data=json_str,
                                    file_name="query_results.json",
                                    mime="application/json"
                                )
                        
                        # Only show visualization options if we have data
                        if not df.empty:
                            st.subheader("Visualize Results")
                            
                            # Check if we have visualization recommendations from the responder
                            has_recommendations = "visualization_recommendations" in result and result["visualization_recommendations"]
                            
                            # Determine visualization types based on recommendations or data analysis
                            if has_recommendations:
                                viz_recommendations = result["visualization_recommendations"]
                                viz_options = viz_recommendations.get("recommended_types", ["Table"])
                                primary_viz = viz_recommendations.get("primary_visualization", "Table")
                                column_types = viz_recommendations.get("column_types", {})
                                
                                # Extract suggested configurations if available
                                bar_config = viz_recommendations.get("bar_chart_config", {})
                                line_config = viz_recommendations.get("line_chart_config", {})
                            else:
                                # Fallback to manual detection if no recommendations
                                viz_options = ["Table"]
                                primary_viz = "Table"
                                
                                # Check if we have numeric columns for charts
                                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                                # Try to convert string columns that might be numeric
                                for col in df.columns:
                                    if col not in numeric_cols:
                                        try:
                                            # First check if column exists and has data
                                            if col in df.columns and not df[col].isna().all():
                                                # Try to convert to numeric, errors='coerce' will convert errors to NaN
                                                df[col] = pd.to_numeric(df[col], errors='coerce')
                                                # Only add to numeric_cols if conversion was successful and not all values are NaN
                                                if not df[col].isna().all() and col not in numeric_cols:
                                                    numeric_cols.append(col)
                                        except Exception as e:
                                            # Just continue if conversion fails
                                            pass
                                
                                categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
                                date_cols = [col for col in df.columns if any(date_pattern in col.lower() 
                                                                            for date_pattern in ["date", "time", "year", "month", "day"])]
                                
                                if len(numeric_cols) > 0:
                                    viz_options.extend(["Bar Chart", "Line Chart"])
                                    if len(numeric_cols) > 1:
                                        viz_options.append("Scatter Plot")
                                        viz_options.append("Correlation Heatmap")
                                
                                if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                                    viz_options.append("Pie Chart")
                                    viz_options.append("Box Plot")
                                    
                                # Set default configurations
                                bar_config = {
                                    "suggested_x": categorical_cols[0] if categorical_cols else None,
                                    "suggested_y": numeric_cols[0] if numeric_cols else None
                                }
                                
                                line_config = {
                                    "suggested_x": date_cols[0] if date_cols else (categorical_cols[0] if categorical_cols else None),
                                    "suggested_y": numeric_cols[0] if numeric_cols else None
                                }
                            
                            # Let user select visualization type with the recommended one as default
                            viz_type = st.selectbox("Select Visualization Type", viz_options, 
                                                   index=viz_options.index(primary_viz) if primary_viz in viz_options else 0)
                            
                            # Initialize these variables to avoid undefined errors
                            numeric_cols = []
                            categorical_cols = []
                            
                            # Default detection of column types if we don't have recommendations
                            if not has_recommendations:
                                numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                                categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
                            
                            if viz_type == "Table":
                                st.dataframe(df, use_container_width=True)
                            
                            elif viz_type == "Bar Chart":
                                # Get numeric and categorical columns
                                if has_recommendations:
                                    numeric_cols = [col for col, type_ in column_types.items() if type_ == "numeric"]
                                    categorical_cols = [col for col, type_ in column_types.items() if type_ == "categorical"]
                                    
                                    # Use recommended configuration if available
                                    suggested_x = bar_config.get("suggested_x")
                                    suggested_y = bar_config.get("suggested_y")
                                    
                                    # Set default index for selectboxes
                                    x_index = categorical_cols.index(suggested_x) if suggested_x in categorical_cols else 0
                                    y_index = numeric_cols.index(suggested_y) if suggested_y in numeric_cols else 0
                                else:
                                    # Use detected columns
                                    x_index = 0
                                    y_index = 0
                                
                                if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                                    x_axis = st.selectbox("X-axis (Category)", categorical_cols, index=x_index if x_index < len(categorical_cols) else 0)
                                    y_axis = st.selectbox("Y-axis (Value)", numeric_cols, index=y_index if y_index < len(numeric_cols) else 0)
                                    
                                    try:
                                        # Make sure we're working with valid data
                                        if df[x_axis].isna().all() or df[y_axis].isna().all():
                                            st.error("Cannot create chart: One or both selected columns contain only null values")
                                        else:
                                            # Create a copy of the dataframe with only the columns we need
                                            chart_df = df[[x_axis, y_axis]].copy().dropna()
                                            
                                            if len(chart_df) > 0:
                                                # Sort data for better visualization
                                                if len(chart_df) <= 25:  # Only sort for smaller datasets
                                                    chart_df = chart_df.sort_values(by=y_axis, ascending=False)
                                                
                                                # Create bar chart with improved styling
                                                fig = px.bar(chart_df, x=x_axis, y=y_axis, 
                                                             title=f"{y_axis} by {x_axis}",
                                                             color_discrete_sequence=px.colors.qualitative.G10,
                                                             labels={x_axis: x_axis.replace('_', ' ').title(), 
                                                                     y_axis: y_axis.replace('_', ' ').title()})
                                                
                                                # Improve layout
                                                fig.update_layout(
                                                    xaxis_title=x_axis.replace('_', ' ').title(),
                                                    yaxis_title=y_axis.replace('_', ' ').title(),
                                                    plot_bgcolor='rgba(240,240,240,0.8)',
                                                    margin=dict(l=20, r=20, t=40, b=20),
                                                    height=500
                                                )
                                                
                                                # Rotate x-axis labels if there are many categories
                                                if len(chart_df[x_axis].unique()) > 5:
                                                    fig.update_layout(xaxis_tickangle=-45)
                                                
                                                st.plotly_chart(fig, use_container_width=True)
                                            else:
                                                st.error("Cannot create chart: No valid data points after removing null values")
                                    except Exception as e:
                                        st.error(f"Error creating bar chart: {str(e)}")
                                else:
                                    st.error("Bar chart requires at least one categorical column and one numeric column")
                            
                            elif viz_type == "Line Chart":
                                # Get appropriate columns for line chart
                                if has_recommendations:
                                    numeric_cols = [col for col, type_ in column_types.items() if type_ == "numeric"]
                                    date_cols = [col for col, type_ in column_types.items() if type_ == "date"]
                                    categorical_cols = [col for col, type_ in column_types.items() if type_ == "categorical"]
                                    
                                    # Use recommended configuration if available
                                    suggested_x = line_config.get("suggested_x")
                                    suggested_y = line_config.get("suggested_y")
                                    
                                    # Set default index for selectboxes
                                    x_cols = date_cols if date_cols else categorical_cols
                                    x_index = x_cols.index(suggested_x) if suggested_x in x_cols else 0
                                    y_index = numeric_cols.index(suggested_y) if suggested_y in numeric_cols else 0
                                else:
                                    # Use detected columns
                                    # First, detect column types
                                    numeric_cols = []
                                    date_cols = []
                                    categorical_cols = []
                                    
                                    # Simple column type detection
                                    for col in df.columns:
                                        if pd.api.types.is_numeric_dtype(df[col]):
                                            numeric_cols.append(col)
                                        elif 'date' in col.lower() or 'time' in col.lower():
                                            # Try to convert to datetime
                                            try:
                                                pd.to_datetime(df[col], errors='raise')
                                                date_cols.append(col)
                                            except:
                                                categorical_cols.append(col)
                                        else:
                                            categorical_cols.append(col)
                                    
                                    x_index = 0
                                    y_index = 0
                                    x_cols = date_cols if date_cols else categorical_cols
                                
                                if len(x_cols) > 0 and len(numeric_cols) > 0:
                                    x_axis = st.selectbox("X-axis (Time/Category)", x_cols, index=x_index if x_index < len(x_cols) else 0)
                                    y_axis = st.selectbox("Y-axis (Value)", numeric_cols, index=y_index if y_index < len(numeric_cols) else 0)
                                    
                                    try:
                                        # Make sure we're working with valid data
                                        if df[x_axis].isna().all() or df[y_axis].isna().all():
                                            st.error("Cannot create chart: One or both selected columns contain only null values")
                                        else:
                                            # Create a copy of the dataframe with only the columns we need
                                            chart_df = df[[x_axis, y_axis]].copy().dropna()
                                            
                                            if len(chart_df) > 0:
                                                # Sort data by x-axis for proper line plotting
                                                try:
                                                    # Try to convert to datetime if it looks like a date
                                                    if x_axis in date_cols:
                                                        chart_df[x_axis] = pd.to_datetime(chart_df[x_axis], errors='coerce')
                                                        chart_df = chart_df.dropna(subset=[x_axis])
                                                    
                                                    # Sort by x-axis
                                                    chart_df = chart_df.sort_values(by=x_axis)
                                                except:
                                                    # If conversion fails, just use the data as is
                                                    pass
                                                
                                                # Create line chart with improved styling
                                                fig = px.line(chart_df, x=x_axis, y=y_axis, 
                                                              title=f"{y_axis} over {x_axis}",
                                                              markers=True,
                                                              labels={x_axis: x_axis.replace('_', ' ').title(), 
                                                                      y_axis: y_axis.replace('_', ' ').title()})
                                                
                                                # Improve layout
                                                fig.update_layout(
                                                    xaxis_title=x_axis.replace('_', ' ').title(),
                                                    yaxis_title=y_axis.replace('_', ' ').title(),
                                                    plot_bgcolor='rgba(240,240,240,0.8)',
                                                    margin=dict(l=20, r=20, t=40, b=20),
                                                    height=500
                                                )
                                                
                                                # Rotate x-axis labels if there are many points
                                                if len(chart_df) > 10:
                                                    fig.update_layout(xaxis_tickangle=-45)
                                                
                                                st.plotly_chart(fig, use_container_width=True)
                                            else:
                                                st.error("Cannot create chart: No valid data points after removing null values")
                                    except Exception as e:
                                        st.error(f"Error creating line chart: {str(e)}")
                                else:
                                    st.error("Line chart requires at least one time/category column and one numeric column")
                            
                            # Fallback for Bar Chart when no categorical columns are available
                            elif viz_type == "Bar Chart" and len(numeric_cols) > 0 and len(categorical_cols) == 0:
                                # If no categorical columns, use the first numeric column as category
                                st.info("No categorical columns found. Using a numeric column as category.")
                                x_axis = st.selectbox("X-axis (Numeric as Category)", numeric_cols)
                                y_axis = st.selectbox("Y-axis (Value)", [col for col in numeric_cols if col != x_axis])
                                try:
                                    fig = px.bar(df, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}")
                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error creating bar chart: {str(e)}")
                            
                            # Line Chart visualization is already implemented above
                            
                            elif viz_type == "Scatter Plot":
                                if len(numeric_cols) >= 2:
                                    x_axis = st.selectbox("X-axis", numeric_cols)
                                    y_axis = st.selectbox("Y-axis", [col for col in numeric_cols if col != x_axis])
                                    color_by = None
                                    if len(categorical_cols) > 0:
                                        use_color = st.checkbox("Color by category")
                                        if use_color:
                                            color_by = st.selectbox("Color by", categorical_cols)
                                    
                                    try:
                                        if color_by:
                                            fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by, title=f"{y_axis} vs {x_axis} by {color_by}")
                                        else:
                                            fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
                                        st.plotly_chart(fig, use_container_width=True)
                                    except Exception as e:
                                        st.error(f"Error creating scatter plot: {str(e)}")
                                        st.info("Falling back to basic chart...")
                                        # Fallback to matplotlib
                                        try:
                                            plt.figure(figsize=(10, 6))
                                            plt.scatter(df[x_axis], df[y_axis])
                                            plt.title(f"{y_axis} vs {x_axis}")
                                            plt.xlabel(x_axis)
                                            plt.ylabel(y_axis)
                                            st.pyplot(plt)
                                        except Exception as e2:
                                            st.error(f"Fallback chart also failed: {str(e2)}")
                                else:
                                    st.info("Scatter plot requires at least 2 numeric columns.")
                            
                            elif viz_type == "Pie Chart":
                                if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                                    category = st.selectbox("Category", categorical_cols)
                                    value = st.selectbox("Value", numeric_cols)
                                    try:
                                        # Group by category and sum values for better pie charts
                                        pie_data = df.groupby(category)[value].sum().reset_index()
                                        fig = px.pie(pie_data, names=category, values=value, title=f"{value} distribution by {category}")
                                        st.plotly_chart(fig, use_container_width=True)
                                    except Exception as e:
                                        st.error(f"Error creating pie chart: {str(e)}")
                                        st.info("Falling back to basic chart...")
                                        # Fallback to matplotlib
                                        try:
                                            plt.figure(figsize=(10, 6))
                                            pie_data = df.groupby(category)[value].sum()
                                            plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%')
                                            plt.title(f"{value} distribution by {category}")
                                            st.pyplot(plt)
                                        except Exception as e2:
                                            st.error(f"Fallback chart also failed: {str(e2)}")
                                else:
                                    st.info("Pie chart requires both categorical and numeric columns.")
                            
                            elif viz_type == "Correlation Heatmap":
                                if len(numeric_cols) > 1:
                                    try:
                                        corr = df[numeric_cols].corr()
                                        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
                                        fig.update_layout(title="Correlation Heatmap")
                                        st.plotly_chart(fig, use_container_width=True)
                                    except Exception as e:
                                        st.error(f"Error creating heatmap: {str(e)}")
                                        st.info("Falling back to basic heatmap...")
                                        # Fallback to seaborn
                                        try:
                                            plt.figure(figsize=(10, 8))
                                            corr = df[numeric_cols].corr()
                                            sns.heatmap(corr, annot=True, cmap="coolwarm")
                                            plt.title("Correlation Heatmap")
                                            st.pyplot(plt)
                                        except Exception as e2:
                                            st.error(f"Fallback heatmap also failed: {str(e2)}")
                                else:
                                    st.info("Correlation heatmap requires at least 2 numeric columns.")
                            
                            elif viz_type == "Box Plot":
                                if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                                    category = st.selectbox("Category", categorical_cols)
                                    value = st.selectbox("Value", numeric_cols)
                                    try:
                                        fig = px.box(df, x=category, y=value, title=f"{value} distribution by {category}")
                                        st.plotly_chart(fig, use_container_width=True)
                                    except Exception as e:
                                        st.error(f"Error creating box plot: {str(e)}")
                                        st.info("Falling back to basic box plot...")
                                        # Fallback to seaborn
                                        try:
                                            plt.figure(figsize=(10, 6))
                                            sns.boxplot(x=df[category], y=df[value])
                                            plt.title(f"{value} distribution by {category}")
                                            plt.xticks(rotation=45)
                                            plt.tight_layout()
                                            st.pyplot(plt)
                                        except Exception as e2:
                                            st.error(f"Fallback box plot also failed: {str(e2)}")
                                else:
                                    st.info("Box plot requires both categorical and numeric columns.")
                        else:
                            status_manager.visualization_empty()
            
            except Exception as e:
                status_manager.query_failure(str(e))
    else:
        st.warning("Please enter a question")

# Conversation History - only show if in conversation mode
if st.session_state.conversation_mode and st.session_state.conversation_history:
    with st.expander("View Conversation History", expanded=False):
        st.subheader("Previous Exchanges")
        for i, exchange in enumerate(st.session_state.conversation_history):
            st.markdown(f"**You:** {exchange['user_query']}")
            st.markdown(f"**Agent:** {exchange['response']}")
            st.markdown("---")
            
            # Add a button to use this question as a starting point for a new question
            if st.button(f"Ask follow-up to #{i+1}", key=f"followup_{i}"):
                st.session_state.followup_question = exchange['user_query']
                st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("Universal SQL Conversational Agent - Banking Database Edition")
st.markdown("Powered by Google's Gemini API")



# Function to clear conversation history
def clear_conversation_history():
    st.session_state.conversation_history = []
    st.session_state.last_query_results = None
    st.session_state.last_sql_query = None

# Add a button to clear the conversation history - only show if in conversation mode
if st.session_state.conversation_mode:
    if st.button("Clear Conversation History", key="clear_history_button", on_click=clear_conversation_history):
        pass
