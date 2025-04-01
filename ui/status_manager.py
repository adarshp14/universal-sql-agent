"""
Status Manager for the Universal SQL Conversational Agent.
Handles dynamic status banners and notifications based on database operations.
"""

import streamlit as st

class StatusManager:
    """
    Manages status messages and notifications in the UI.
    Provides context-aware banners based on database operations and query execution.
    """
    
    def __init__(self):
        """Initialize the status manager"""
        # This will be set to the container where status messages should be displayed
        self.container = None
        
    def set_container(self, container):
        """Set the container where status messages should be displayed"""
        self.container = container
    
    def schema_success(self, table_count):
        """Display a success message for schema analysis"""
        # Disabled to prevent duplicate messages
        pass
    
    def schema_failure(self, error_message):
        """Display an error message for schema analysis failure"""
        # Disabled to prevent duplicate messages
        pass
    
    def connection_success(self, db_type, connection_info):
        """Display a success message for database connection"""
        if self.container:
            with self.container:
                st.success(f"✅ Successfully connected to {db_type} database at {connection_info}")
        else:
            st.success(f"✅ Successfully connected to {db_type} database at {connection_info}")
    
    def connection_failure(self, error_message):
        """Display an error message for database connection failure"""
        if self.container:
            with self.container:
                st.error(f"❌ Database connection failed: {error_message}")
        else:
            st.error(f"❌ Database connection failed: {error_message}")
    
    def query_success(self, row_count):
        """Display a success message for query execution"""
        if self.container:
            with self.container:
                if row_count == 0:
                    st.info("✅ Query executed successfully. No matching results found.")
                else:
                    st.success(f"✅ Query executed successfully. Found {row_count} matching results.")
        else:
            if row_count == 0:
                st.info("✅ Query executed successfully. No matching results found.")
            else:
                st.success(f"✅ Query executed successfully. Found {row_count} matching results.")
    
    def query_failure(self, error_message):
        """Display an error message for query execution failure"""
        # Extract the most relevant part of the error message
        if "no such table" in error_message.lower():
            table_name = error_message.split("'")[1] if "'" in error_message else "unknown"
            message = f"❌ Query failed: Table '{table_name}' not found in the current schema."
        elif "no such column" in error_message.lower():
            column_name = error_message.split("'")[1] if "'" in error_message else "unknown"
            message = f"❌ Query failed: Column '{column_name}' not found in the table."
        elif "syntax error" in error_message.lower():
            message = f"❌ Query failed: SQL syntax error. {error_message}"
        else:
            message = f"❌ Query failed: {error_message}"
            
        if self.container:
            with self.container:
                st.error(message)
        else:
            st.error(message)
    
    def schema_refresh_initiated(self):
        """Display a message for schema refresh initiation"""
        # Disabled to prevent duplicate messages
        pass
    
    def export_success(self, format_type):
        """Display a success message for data export"""
        if self.container:
            with self.container:
                st.success(f"✅ Data successfully exported to {format_type} format.")
        else:
            st.success(f"✅ Data successfully exported to {format_type} format.")
    
    def visualization_empty(self):
        """Display an info message for empty visualization data"""
        if self.container:
            with self.container:
                st.info("ℹ️ No data available for visualization.")
        else:
            st.info("ℹ️ No data available for visualization.")
    
    def visualization_error(self, error_message):
        """Display an error message for visualization failure"""
        if self.container:
            with self.container:
                st.error(f"❌ Visualization failed: {error_message}")
        else:
            st.error(f"❌ Visualization failed: {error_message}")
        
    def schema_empty(self):
        """Display an info message when no tables are found in the schema"""
        # Disabled to prevent duplicate messages
        pass
