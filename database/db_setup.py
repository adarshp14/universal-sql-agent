"""
Database setup and utility functions for the Universal SQL Conversational Agent.
Generic e-commerce database with multiple tables.
"""

import os
import sqlite3
import re
from datetime import date, datetime, timedelta
import random
from sqlalchemy import create_engine, text

def setup_sample_database(db_path):
    """
    Set up a generic e-commerce database with multiple related tables.
    
    Args:
        db_path (str): Path to the SQLite database file
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Connect to the database (will create it if it doesn't exist)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Drop existing tables if they exist
    cursor.execute("DROP TABLE IF EXISTS order_items")
    cursor.execute("DROP TABLE IF EXISTS orders")
    cursor.execute("DROP TABLE IF EXISTS products")
    cursor.execute("DROP TABLE IF EXISTS categories")
    cursor.execute("DROP TABLE IF EXISTS customers")
    
    # Create tables
    # 1. Categories table
    cursor.execute('''
    CREATE TABLE categories (
        category_id INTEGER PRIMARY KEY,
        category_name VARCHAR(100) NOT NULL,
        description TEXT,
        parent_category_id INTEGER,
        created_date DATE NOT NULL
    )
    ''')
    
    # 2. Products table
    cursor.execute('''
    CREATE TABLE products (
        product_id INTEGER PRIMARY KEY,
        category_id INTEGER NOT NULL,
        product_name VARCHAR(100) NOT NULL,
        description TEXT,
        price FLOAT NOT NULL,
        stock_quantity INTEGER NOT NULL,
        created_date DATE NOT NULL,
        FOREIGN KEY (category_id) REFERENCES categories(category_id)
    )
    ''')
    
    # Note: SQLite doesn't support ALTER TABLE ADD FOREIGN KEY
    # We'll enforce the relationship in our application logic instead
    
    # 3. Customers table
    cursor.execute('''
    CREATE TABLE customers (
        customer_id INTEGER PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        email VARCHAR(100) UNIQUE NOT NULL,
        phone VARCHAR(20),
        address VARCHAR(200),
        registration_date DATE NOT NULL,
        last_login_date DATE
    )
    ''')
    
    # 4. Orders table
    cursor.execute('''
    CREATE TABLE orders (
        order_id INTEGER PRIMARY KEY,
        customer_id INTEGER NOT NULL,
        order_date DATETIME NOT NULL,
        total_amount FLOAT NOT NULL,
        status VARCHAR(50) NOT NULL,
        shipping_address TEXT,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    )
    ''')
    
    # 5. Order Items table
    cursor.execute('''
    CREATE TABLE order_items (
        order_item_id INTEGER PRIMARY KEY,
        order_id INTEGER NOT NULL,
        product_id INTEGER NOT NULL,
        quantity INTEGER NOT NULL,
        price_per_unit FLOAT NOT NULL,
        FOREIGN KEY (order_id) REFERENCES orders(order_id),
        FOREIGN KEY (product_id) REFERENCES products(product_id)
    )
    ''')
    
    # Insert sample data
    # 1. Categories data
    category_data = [
        (1, 'Electronics', 'Electronic devices and accessories', None, '2020-01-15'),
        (2, 'Clothing', 'Apparel and fashion items', None, '2020-02-20'),
        (3, 'Home & Kitchen', 'Home appliances and kitchenware', None, '2020-03-10'),
        (4, 'Sports & Outdoors', 'Athletic and outdoor equipment', None, '2020-04-05'),
        (5, 'Books & Media', 'Books, music, and entertainment', None, '2020-05-22')
    ]
    
    cursor.executemany(
        "INSERT INTO categories (category_id, category_name, description, parent_category_id, created_date) VALUES (?, ?, ?, ?, ?)",
        category_data
    )
    
    # 2. Products data
    product_data = [
        (1, 1, 'Smartphone', 'High-end smartphone with latest features', 699.99, 50, '2023-01-15'),
        (2, 1, 'Laptop', 'Powerful laptop for work and gaming', 1299.99, 30, '2023-02-20'),
        (3, 1, 'Headphones', 'Wireless noise-cancelling headphones', 199.99, 100, '2023-03-10'),
        (4, 2, 'T-Shirt', 'Cotton t-shirt with logo', 24.99, 200, '2023-01-20'),
        (5, 2, 'Jeans', 'Classic blue jeans', 49.99, 150, '2023-02-15'),
        (6, 2, 'Jacket', 'Winter jacket with hood', 89.99, 75, '2023-03-22'),
        (7, 1, 'Tablet', '10-inch tablet with HD display', 349.99, 40, '2023-04-05'),
        (8, 1, 'Smart Watch', 'Fitness tracker and smartwatch', 149.99, 60, '2023-04-12'),
        (9, 2, 'Sneakers', 'Athletic shoes for running', 79.99, 100, '2023-04-18'),
        (10, 2, 'Hat', 'Stylish hat for all seasons', 19.99, 120, '2023-04-25')
    ]
    
    cursor.executemany(
        "INSERT INTO products (product_id, category_id, product_name, description, price, stock_quantity, created_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
        product_data
    )
    
    # 3. Customer data
    customer_data = [
        (1, 'Alice Williams', 'alice.w@email.com', '555-123-4567', '123 Main St, New York', '2022-02-15', '2023-04-12'),
        (2, 'Bob Miller', 'bob.m@email.com', '555-234-5678', '456 Oak Ave, Chicago', '2022-07-20', '2023-03-23'),
        (3, 'Carol Davis', 'carol.d@email.com', '555-345-6789', '789 Pine Rd, Los Angeles', '2022-03-10', '2023-04-05'),
        (4, 'Daniel Garcia', 'daniel.g@email.com', '555-456-7890', '101 Elm Blvd, Miami', '2022-09-05', '2023-03-30'),
        (5, 'Emma Wilson', 'emma.w@email.com', '555-567-8901', '202 Cedar Ln, Seattle', '2022-11-12', '2023-04-18'),
        (6, 'Frank Thomas', 'frank.t@email.com', '555-678-9012', '303 Birch Dr, New York', '2022-05-22', '2023-04-09'),
        (7, 'Grace Martinez', 'grace.m@email.com', '555-789-0123', '404 Maple Ct, Chicago', '2022-08-15', '2023-04-27'),
        (8, 'Henry Johnson', 'henry.j@email.com', '555-890-1234', '505 Walnut Pl, Los Angeles', '2022-01-30', '2023-04-14'),
        (9, 'Isabella Brown', 'isabella.b@email.com', '555-901-2345', '606 Spruce Way, Miami', '2022-04-18', '2023-04-21'),
        (10, 'Jack Smith', 'jack.s@email.com', '555-012-3456', '707 Fir Ave, Seattle', '2022-12-05', '2023-04-10')
    ]
    
    cursor.executemany(
        "INSERT INTO customers (customer_id, name, email, phone, address, registration_date, last_login_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
        customer_data
    )
    
    # 4. Orders data
    order_data = [
        (1, 1, '2023-03-15 10:30:00', 899.98, 'Delivered', '123 Main St, New York'),
        (2, 2, '2023-03-18 14:45:00', 149.97, 'Shipped', '456 Oak Ave, Chicago'),
        (3, 3, '2023-03-20 09:15:00', 1299.99, 'Processing', '789 Pine Rd, Los Angeles'),
        (4, 4, '2023-03-22 16:20:00', 74.97, 'Delivered', '101 Elm Blvd, Miami'),
        (5, 5, '2023-03-25 11:05:00', 249.98, 'Shipped', '202 Cedar Ln, Seattle'),
        (6, 6, '2023-03-28 13:40:00', 1049.97, 'Delivered', '303 Birch Dr, New York'),
        (7, 7, '2023-04-01 15:30:00', 199.99, 'Processing', '404 Maple Ct, Chicago'),
        (8, 8, '2023-04-05 10:15:00', 349.99, 'Shipped', '505 Walnut Pl, Los Angeles'),
        (9, 9, '2023-04-08 14:50:00', 129.98, 'Delivered', '606 Spruce Way, Miami'),
        (10, 10, '2023-04-10 09:25:00', 699.99, 'Processing', '707 Fir Ave, Seattle'),
        (11, 1, '2023-04-12 16:35:00', 499.98, 'Shipped', '123 Main St, New York'),
        (12, 3, '2023-04-15 11:20:00', 269.97, 'Delivered', '789 Pine Rd, Los Angeles'),
        (13, 5, '2023-04-18 13:45:00', 149.99, 'Processing', '202 Cedar Ln, Seattle'),
        (14, 7, '2023-04-20 15:10:00', 899.99, 'Shipped', '404 Maple Ct, Chicago'),
        (15, 9, '2023-04-22 10:30:00', 59.98, 'Delivered', '606 Spruce Way, Miami')
    ]
    
    cursor.executemany(
        "INSERT INTO orders (order_id, customer_id, order_date, total_amount, status, shipping_address) VALUES (?, ?, ?, ?, ?, ?)",
        order_data
    )
    
    # 5. Order Items data
    order_item_data = [
        (1, 1, 1, 1, 699.99),  # Order 1, Smartphone
        (2, 1, 3, 1, 199.99),  # Order 1, Headphones
        (3, 2, 4, 2, 24.99),   # Order 2, T-Shirt (2)
        (4, 2, 9, 1, 79.99),   # Order 2, Sneakers
        (5, 2, 10, 1, 19.99),  # Order 2, Hat
        (6, 3, 2, 1, 1299.99), # Order 3, Laptop
        (7, 4, 4, 3, 24.99),   # Order 4, T-Shirt (3)
        (8, 5, 8, 1, 149.99),  # Order 5, Smart Watch
        (9, 5, 9, 1, 79.99),   # Order 5, Sneakers
        (10, 5, 10, 1, 19.99)  # Order 5, Hat
    ]
    
    # Add more order items
    order_item_data.extend([
        (11, 6, 2, 1, 1299.99), # Order 6, Laptop
        (12, 6, 3, 1, 199.99),  # Order 6, Headphones
        (13, 7, 3, 1, 199.99),  # Order 7, Headphones
        (14, 8, 7, 1, 349.99),  # Order 8, Tablet
        (15, 9, 5, 1, 49.99),   # Order 9, Jeans
        (16, 9, 9, 1, 79.99),   # Order 9, Sneakers
        (17, 10, 1, 1, 699.99), # Order 10, Smartphone
        (18, 11, 8, 1, 149.99), # Order 11, Smart Watch
        (19, 11, 7, 1, 349.99), # Order 11, Tablet
        (20, 12, 6, 3, 89.99),  # Order 12, Jacket (3)
        (21, 13, 8, 1, 149.99), # Order 13, Smart Watch
        (22, 14, 1, 1, 699.99), # Order 14, Smartphone
        (23, 14, 7, 1, 199.99), # Order 14, Tablet
        (24, 15, 10, 3, 19.99)  # Order 15, Hat (3)
    ])
    
    cursor.executemany(
        "INSERT INTO order_items (order_item_id, order_id, product_id, quantity, price_per_unit) VALUES (?, ?, ?, ?, ?)",
        order_item_data
    )
    
    # Commit changes and close the connection
    conn.commit()
    conn.close()
    
    print(f"Sample e-commerce database setup complete at {db_path}")

def get_db_engine(db_path):
    """
    Create and return a SQLAlchemy engine for the given database path.
    
    Args:
        db_path (str): Path to the SQLite database file
        
    Returns:
        Engine: SQLAlchemy engine object
    """
    # Create a SQLAlchemy engine
    engine = create_engine(f"sqlite:///{db_path}")
    
    return engine

def make_query_case_insensitive(query, db_type):
    """
    Modify a SQL query to make string comparisons case insensitive based on the database type.
    
    Args:
        query (str): The SQL query to modify
        db_type (str): The database type ('sqlite', 'mysql', 'postgres', 'mssql')
        
    Returns:
        str: Modified SQL query with case insensitive string comparisons
    """
    # For SQLite, which is our primary database type
    if db_type == 'sqlite':
        # More comprehensive approach for SQLite
        # Handle various string comparison patterns with better regex
        
        # Pattern 1: Handle column = 'value' with optional table qualifier
        # This handles both column_name = 'value' and table.column_name = 'value'
        pattern1 = r'(\b[\w\.]+\b)\s+(=|LIKE|<>|!=)\s+([\'"].*?[\'"])'
        replacement1 = r'UPPER(\1) \2 UPPER(\3)'
        query = re.sub(pattern1, replacement1, query, flags=re.IGNORECASE)
        
        # Pattern 2: Handle 'value' = column with optional table qualifier
        pattern2 = r'([\'"].*?[\'"])\s+(=|LIKE|<>|!=)\s+(\b[\w\.]+\b)'
        replacement2 = r'UPPER(\1) \2 UPPER(\3)'
        query = re.sub(pattern2, replacement2, query, flags=re.IGNORECASE)
        
        # Pattern 3: Handle IN ('value1', 'value2', ...) with optional table qualifier
        pattern3 = r'(\b[\w\.]+\b)\s+IN\s+\(([\'"].*?[\'"](?:\s*,\s*[\'"].*?[\'"])*)\)'
        
        # Custom replacement function to handle IN clause values
        def replace_in_clause(match):
            col = match.group(1)
            values = match.group(2)
            # Split the values and wrap each in UPPER()
            value_list = re.findall(r'[\'"].*?[\'"]', values)
            upper_values = [f"UPPER({v})" for v in value_list]
            return f"UPPER({col}) IN ({', '.join(upper_values)})"
        
        # Apply the IN clause replacement
        query = re.sub(pattern3, replace_in_clause, query, flags=re.IGNORECASE)
        
    elif db_type == 'mysql':
        # For MySQL, use COLLATE for case insensitivity
        # Pattern 1: column = 'value'
        pattern1 = r'(\b[\w\.]+\b)\s+(=|LIKE|<>|!=)\s+([\'"].*?[\'"])'
        replacement1 = r'\1 \2 \3 COLLATE utf8mb4_general_ci'
        query = re.sub(pattern1, replacement1, query, flags=re.IGNORECASE)
        
        # Pattern 2: IN clause
        pattern2 = r'(\b[\w\.]+\b)\s+IN\s+\(([\'"].*?[\'"](?:\s*,\s*[\'"].*?[\'"])*)\)'
        replacement2 = r'\1 IN (\2) COLLATE utf8mb4_general_ci'
        query = re.sub(pattern2, replacement2, query, flags=re.IGNORECASE)
        
    elif db_type == 'postgres':
        # For PostgreSQL, use ILIKE and LOWER()
        # Replace LIKE with ILIKE for case insensitivity
        query = re.sub(r'\bLIKE\b', 'ILIKE', query, flags=re.IGNORECASE)
        
        # Pattern 1: column = 'value'
        pattern1 = r'(\b[\w\.]+\b)\s+(=|<>|!=)\s+([\'"].*?[\'"])'
        replacement1 = r'LOWER(\1) \2 LOWER(\3)'
        query = re.sub(pattern1, replacement1, query, flags=re.IGNORECASE)
        
        # Pattern 2: IN clause
        pattern2 = r'(\b[\w\.]+\b)\s+IN\s+\(([\'"].*?[\'"](?:\s*,\s*[\'"].*?[\'"])*)\)'
        
        def replace_pg_in_clause(match):
            col = match.group(1)
            values = match.group(2)
            value_list = re.findall(r'[\'"].*?[\'"]', values)
            lower_values = [f"LOWER({v})" for v in value_list]
            return f"LOWER({col}) IN ({', '.join(lower_values)})"
        
        query = re.sub(pattern2, replace_pg_in_clause, query, flags=re.IGNORECASE)
        
    elif db_type == 'mssql':
        # For SQL Server, use COLLATE for case insensitivity
        pattern1 = r'(\b[\w\.]+\b)\s+(=|LIKE|<>|!=)\s+([\'"].*?[\'"])'
        replacement1 = r'\1 COLLATE SQL_Latin1_General_CP1_CI_AS \2 \3'
        query = re.sub(pattern1, replacement1, query, flags=re.IGNORECASE)
        
        # Pattern 2: IN clause
        pattern2 = r'(\b[\w\.]+\b)\s+IN\s+\(([\'"].*?[\'"](?:\s*,\s*[\'"].*?[\'"])*)\)'
        replacement2 = r'\1 COLLATE SQL_Latin1_General_CP1_CI_AS IN (\2)'
        query = re.sub(pattern2, replacement2, query, flags=re.IGNORECASE)
    
    return query

def execute_query(engine, query):
    """
    Execute a SQL query on the database.
    
    Args:
        engine: SQLAlchemy engine
        query (str): SQL query to execute
        
    Returns:
        list: List of dictionaries containing the query results
    """
    try:
        # Determine database type
        is_sqlite = 'sqlite' in engine.name.lower()
        is_mysql = 'mysql' in engine.name.lower() or 'mariadb' in engine.name.lower()
        is_postgres = 'postgres' in engine.name.lower() or 'postgresql' in engine.name.lower()
        is_mssql = 'mssql' in engine.name.lower() or 'sqlserver' in engine.name.lower()
        
        # Special handling for direct SQL queries with account_type and transaction_type
        # This is a more direct approach for the specific case mentioned by the user
        # Handle both unqualified and qualified column names with table aliases
        
        # For account_type comparisons
        replacements = [
            (r"(\w+\.)?account_type\s*=\s*'Savings'", r"UPPER(\1account_type) = UPPER('Savings')"),
            (r"(\w+\.)?account_type\s*=\s*'Checking'", r"UPPER(\1account_type) = UPPER('Checking')"),
            (r"(\w+\.)?account_type\s*=\s*'Loan'", r"UPPER(\1account_type) = UPPER('Loan')"),
            (r"(\w+\.)?transaction_type\s*=\s*'Payment'", r"UPPER(\1transaction_type) = UPPER('Payment')"),
            # Add more patterns for other common string comparisons as needed
            (r"(\w+\.)?account_type\s*=\s*'Credit'", r"UPPER(\1account_type) = UPPER('Credit')"),
            (r"(\w+\.)?transaction_type\s*=\s*'Deposit'", r"UPPER(\1transaction_type) = UPPER('Deposit')"),
            (r"(\w+\.)?transaction_type\s*=\s*'Withdrawal'", r"UPPER(\1transaction_type) = UPPER('Withdrawal')"),
            (r"(\w+\.)?transaction_type\s*=\s*'Transfer'", r"UPPER(\1transaction_type) = UPPER('Transfer')")
        ]
        
        # Apply all the replacements
        for pattern, replacement in replacements:
            query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
        
        # For SQLite, ensure LIMIT is part of the main query, not a separate statement
        if is_sqlite:
            # Remove any standalone LIMIT statements and ensure they're integrated into the main query
            query = query.replace(';\n  LIMIT', ' LIMIT')
            # Remove any BEGIN; or ROLLBACK; statements that might cause issues with SQLite
            query = query.replace('BEGIN;', '').replace('ROLLBACK;', '')
            # Clean up any extra semicolons that might cause the "multiple statements" error
            query = query.replace(';\n', '\n')
            
            # Make query case insensitive for SQLite
            query = make_query_case_insensitive(query, 'sqlite')
            
        elif is_mysql:
            # Make query case insensitive for MySQL/MariaDB
            query = make_query_case_insensitive(query, 'mysql')
            
        elif is_postgres:
            # Make query case insensitive for PostgreSQL
            query = make_query_case_insensitive(query, 'postgres')
            
        elif is_mssql:
            # Make query case insensitive for SQL Server
            query = make_query_case_insensitive(query, 'mssql')
        else:
            # For unknown database types, default to SQLite-style case insensitivity
            query = make_query_case_insensitive(query, 'sqlite')
        
        with engine.connect() as connection:
            # Execute the query
            result = connection.execute(text(query))
            
            # Check if the query returns rows
            if result.returns_rows:
                # Get column names
                columns = result.keys()
                
                # Fetch all results and convert to list of dictionaries
                rows = result.fetchall()
                return [dict(zip(columns, row)) for row in rows]
            else:
                # For queries that don't return rows (INSERT, UPDATE, DELETE)
                connection.commit()
                return [{"message": "Query executed successfully. No rows returned."}]
    except Exception as e:
        # Return error information
        return [{"error": str(e)}]
