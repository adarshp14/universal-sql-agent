"""
Database Profile Manager for the Universal SQL Conversational Agent.
Handles saving, loading, and managing database connection profiles.
"""

import os
import json
import streamlit as st
from sqlalchemy import create_engine
from database.db_setup import get_db_engine

class ProfileManager:
    """
    Manages database connection profiles for the Universal SQL Conversational Agent.
    Allows users to save, load, and switch between different database connections.
    """
    
    def __init__(self, profiles_dir=None):
        """Initialize the profile manager with a directory for storing profiles"""
        if profiles_dir is None:
            # Default to a 'profiles' directory in the database folder
            self.profiles_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "database", "profiles"
            )
        else:
            self.profiles_dir = profiles_dir
            
        # Create the profiles directory if it doesn't exist
        os.makedirs(self.profiles_dir, exist_ok=True)
    
    def get_profiles(self):
        """Get a list of all saved profiles"""
        profiles = []
        try:
            for filename in os.listdir(self.profiles_dir):
                if filename.endswith('.json'):
                    profile_name = filename[:-5]  # Remove .json extension
                    profiles.append(profile_name)
            return sorted(profiles)
        except Exception as e:
            st.error(f"Error loading profiles: {str(e)}")
            return []
    
    def save_profile(self, profile_name, connection_info):
        """
        Save a database connection profile
        
        Args:
            profile_name (str): Name of the profile
            connection_info (dict): Connection information including type, host, port, etc.
        """
        try:
            # Sanitize profile name for filename
            safe_name = "".join(c for c in profile_name if c.isalnum() or c in "._- ")
            
            # Create the profile file path
            profile_path = os.path.join(self.profiles_dir, f"{safe_name}.json")
            
            # Save the connection info to the profile file
            with open(profile_path, 'w') as f:
                json.dump(connection_info, f, indent=2)
                
            return True, f"Profile '{profile_name}' saved successfully"
        except Exception as e:
            return False, f"Error saving profile: {str(e)}"
    
    def load_profile(self, profile_name):
        """
        Load a database connection profile
        
        Args:
            profile_name (str): Name of the profile to load
            
        Returns:
            tuple: (success, message, connection_info)
        """
        try:
            # Create the profile file path
            profile_path = os.path.join(self.profiles_dir, f"{profile_name}.json")
            
            # Load the connection info from the profile file
            with open(profile_path, 'r') as f:
                connection_info = json.load(f)
                
            return True, f"Profile '{profile_name}' loaded successfully", connection_info
        except Exception as e:
            return False, f"Error loading profile: {str(e)}", None
    
    def delete_profile(self, profile_name):
        """
        Delete a database connection profile
        
        Args:
            profile_name (str): Name of the profile to delete
            
        Returns:
            tuple: (success, message)
        """
        try:
            # Create the profile file path
            profile_path = os.path.join(self.profiles_dir, f"{profile_name}.json")
            
            # Delete the profile file
            if os.path.exists(profile_path):
                os.remove(profile_path)
                return True, f"Profile '{profile_name}' deleted successfully"
            else:
                return False, f"Profile '{profile_name}' not found"
        except Exception as e:
            return False, f"Error deleting profile: {str(e)}"
    
    def connect_to_profile(self, profile_name):
        """
        Connect to a database using a saved profile
        
        Args:
            profile_name (str): Name of the profile to connect to
            
        Returns:
            tuple: (success, message, engine)
        """
        success, message, connection_info = self.load_profile(profile_name)
        
        if not success:
            return False, message, None
        
        try:
            # Connect to the database based on the connection type
            db_type = connection_info.get('type', '').lower()
            
            if db_type == 'sqlite':
                # For SQLite, we use the file path
                db_path = connection_info.get('path', '')
                engine = get_db_engine(db_path)
                
            elif db_type in ['mysql', 'postgresql', 'postgres', 'mssql', 'oracle']:
                # For other database types, we build a connection string
                host = connection_info.get('host', '')
                port = connection_info.get('port', '')
                database = connection_info.get('database', '')
                username = connection_info.get('username', '')
                password = connection_info.get('password', '')
                
                # Map the database type to the SQLAlchemy dialect
                dialect_map = {
                    'mysql': 'mysql+pymysql',
                    'postgresql': 'postgresql+psycopg2',
                    'postgres': 'postgresql+psycopg2',
                    'mssql': 'mssql+pyodbc',
                    'oracle': 'oracle+cx_oracle'
                }
                
                dialect = dialect_map.get(db_type, db_type)
                
                # URL encode the username and password to handle special characters
                import urllib.parse
                encoded_username = urllib.parse.quote_plus(username) if username else ''
                encoded_password = urllib.parse.quote_plus(password) if password else ''
                
                # Handle localhost specially - use 127.0.0.1 instead to avoid DNS resolution issues
                if host.lower() == 'localhost':
                    host = '127.0.0.1'
                    
                # Build the connection URL
                if username and password:
                    connection_url = f"{dialect}://{encoded_username}:{encoded_password}@{host}:{port}/{database}"
                else:
                    connection_url = f"{dialect}://{host}:{port}/{database}"
                
                # Create the engine
                engine = create_engine(connection_url)
                
            elif db_type == 'url':
                # For direct URL connections
                connection_url = connection_info.get('url', '')
                engine = create_engine(connection_url)
                
            else:
                return False, f"Unsupported database type: {db_type}", None
            
            # Test the connection with better error handling
            try:
                with engine.connect() as conn:
                    pass  # Just test that we can connect
                return True, f"Connected to {db_type} database using profile '{profile_name}'", engine
            except Exception as conn_error:
                # Provide a more user-friendly error message
                error_msg = str(conn_error)
                if "could not translate host name" in error_msg.lower():
                    return False, f"Could not connect to host. Please check if the host name is correct or try using IP address (127.0.0.1) instead of 'localhost'.", None
                elif "connection refused" in error_msg.lower():
                    return False, f"Connection refused. Please check if the database server is running and the port {port} is correct.", None
                elif "password authentication failed" in error_msg.lower():
                    return False, "Authentication failed. Please check your username and password.", None
                elif "database" in error_msg.lower() and "does not exist" in error_msg.lower():
                    return False, f"Database '{database}' does not exist. Please check the database name.", None
                else:
                    return False, f"Error connecting to database: {error_msg}", None
            
        except Exception as e:
            return False, f"Error connecting to database: {str(e)}", None
