# Universal SQL Conversational Agent

A powerful AI-powered tool that allows users to interact with SQL databases using natural language. This project demonstrates integration of NLP with database technologies using a custom multi-agent architecture. The application features dynamic status banners and database profile management for an enhanced user experience.

The project includes a sample e-commerce database schema with tables for categories, products, customers, orders, and order items, making it easy to get started with example queries about product sales, customer behavior, and inventory management.

## Features

- Parse SQL schema to understand database structure
- Translate natural language queries into SQL
- Execute SQL queries and return results in human-readable language
- Works with any SQL database schema
- Custom agent pipeline architecture for flexibility and maintainability
- Dynamic, context-aware status banners for operation feedback
- Database profile management to save and switch between connections
- Docker support for easy deployment and distribution

## Requirements

- Python 3.11 or higher
- Gemini API key (Google AI Studio)
- Dependencies listed in requirements.txt

## Setup Instructions

### Running Locally

1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - macOS/Linux: `source venv/bin/activate`
   - Windows: `venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `.env` file in the project root with your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```
6. Run the Streamlit application: `streamlit run ui/streamlit_app.py`

### Running with Docker

1. Clone this repository
2. Create a `.env` file in the project root with your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```
3. Build and run using Docker Compose:
   ```bash
   docker-compose up -d
   ```
4. Access the application at http://localhost:8501

## Project Structure

- `agents/`: Contains the agent implementations
  - `schema_parser.py`: Parses database schema
  - `query_translator.py`: Translates natural language to SQL
  - `sql_executor.py`: Executes SQL queries
  - `responder.py`: Formats results as natural language
- `database/`: Database utilities and sample database setup
  - `db_setup.py`: Sets up a sample e-commerce database
  - `profiles/`: Saved database connection profiles
- `ui/`: Streamlit UI components
  - `streamlit_app.py`: Main Streamlit application
  - `status_manager.py`: Dynamic status banner management
  - `profile_manager.py`: Database profile management
- `Dockerfile`: Docker configuration for containerization
- `docker-compose.yml`: Docker Compose configuration for easy deployment

## Example Usage

Input: "Who earns more than 50,000?"
Output: "Alice Smith and Bob Johnson earn more than $50,000."

## How It Works

1. The user connects to a database (sample, custom, URL, or saved profile)
2. The SchemaParser agent automatically extracts database structure information
3. The user submits a natural language query
4. The QueryTranslator agent converts the natural language query to SQL
5. The SQLExecutor agent runs the SQL query against the database
6. Dynamic status banners provide feedback on each operation

## Database Profile Management

The application allows you to save and manage database connection profiles:

1. Connect to any database (SQLite, PostgreSQL, MySQL, etc.)
2. Check the "Save as profile" option and provide a profile name
3. Access your saved profiles from the "Saved Profiles" option
4. Connect to or delete profiles as needed

Profiles are stored securely in the `database/profiles` directory.
5. The Responder agent formats the results into a natural language response
6. The response is presented to the user

## Sample E-commerce Database Schema

The project includes a sample e-commerce database with the following tables:

1. **Categories**
   - `category_id`: Primary key
   - `category_name`: Name of the category
   - `description`: Category description
   - `parent_category_id`: Self-referential foreign key
   - `created_date`: Date the category was created

2. **Products**
   - `product_id`: Primary key
   - `category_id`: Foreign key to Categories
   - `product_name`: Name of the product
   - `description`: Product description
   - `price`: Product price
   - `stock_quantity`: Available stock
   - `created_date`: Date the product was added

3. **Customers**
   - `customer_id`: Primary key
   - `name`: Customer's name
   - `email`: Customer's email
   - `phone`: Customer's phone number
   - `address`: Customer's address
   - `registration_date`: Date the customer registered
   - `last_login_date`: Date of the customer's last login

4. **Orders**
   - `order_id`: Primary key
   - `customer_id`: Foreign key to Customers
   - `order_date`: Date and time of the order
   - `total_amount`: Total order amount
   - `status`: Order status (e.g., Processing, Shipped, Delivered)
   - `shipping_address`: Shipping address

5. **Order Items**
   - `order_item_id`: Primary key
   - `order_id`: Foreign key to Orders
   - `product_id`: Foreign key to Products
   - `quantity`: Quantity ordered
   - `price_per_unit`: Price per unit at time of order

## Preparing for GitHub

Before pushing to GitHub, make sure to:

1. **Protect Sensitive Information**
   - Never commit your `.env` file containing API keys
   - Add `.env` to your `.gitignore` file
   - Include a sample `.env.example` file with placeholders

2. **Clean Up Unnecessary Files**
   - Remove any `__pycache__` directories and `.pyc` files
   - Add appropriate patterns to `.gitignore`

3. **Documentation**
   - Ensure the README is up-to-date
   - Add comments to complex code sections

## Next Steps

- Add support for multiple tables and complex relationships
- Add support for more database types (PostgreSQL, MySQL, etc.)
- Implement query history and result caching
- Add more advanced natural language understanding capabilities
