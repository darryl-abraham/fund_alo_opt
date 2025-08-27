import sqlite3
import pandas as pd
import logging
import hashlib
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get the current directory and project root
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
db_path = project_root / 'data' / 'langston.db'

def get_db_path():
    """
    Get the path to the SQLite database file
    
    Returns:
        str: Path to the database file
    """
    return str(db_path)

def get_db_connection():
    """
    Create and return a connection to the SQLite database
    
    Returns:
        sqlite3.Connection: A connection to the SQLite database
    """
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row  # This enables column access by name
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}")
        raise

def get_bank_names():
    """
    Get a list of all unique bank names from the test_data table
    
    Returns:
        list: A list of unique bank names
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT holder FROM test_data ORDER BY holder")
        banks = [row[0] for row in cursor.fetchall()]
        conn.close()
        return banks
    except Exception as e:
        logger.error(f"Error getting bank names: {str(e)}")
        raise

def get_available_banks():
    """
    Get a list of banks that have rates available in the cd_rates table
    
    Returns:
        list: A list of bank names that have available rates
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Join test_data and cd_rates tables to find banks with available rates
        query = """
        SELECT DISTINCT t.holder 
        FROM test_data t
        JOIN cd_rates c ON t.holder = c.bank_name
        ORDER BY t.holder
        """
        cursor.execute(query)
        banks = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        if not banks:
            # Fallback to all banks if no matches
            logger.warning("No banks with rates found, falling back to all banks")
            return get_bank_names()
            
        return banks
    except Exception as e:
        logger.error(f"Error getting available banks: {str(e)}")
        # Fallback to all banks on error
        return get_bank_names()

def get_bank_rates_df():
    """
    Get CD rates information from the cd_rates table as a DataFrame
    
    Returns:
        pd.DataFrame: DataFrame containing CD rates information
    """
    try:
        conn = get_db_connection()
        query = "SELECT * FROM cd_rates"
        bank_rates_df = pd.read_sql_query(query, conn)
        
        # Process data similar to how it was processed from Excel
        # Add a numeric version of CD Term for filtering
        bank_rates_df['CD Term Num'] = bank_rates_df['cd_term'].str.extract(r'(\d+)').astype(float)
        
        # Rename columns to match the previous Excel version
        bank_rates_df = bank_rates_df.rename(columns={
            'bank_name': 'Bank Name',
            'bank_code': 'Bank Code',
            'cd_term': 'CD Term',
            'cd_rate': 'CD Rate',
            'cdars_term': 'CDARS Term',
            'cdars_rate': 'CDARS Rate'
        })
        
        conn.close()
        return bank_rates_df
    except Exception as e:
        logger.error(f"Error getting bank rates: {str(e)}")
        raise

def get_bank_ranking_df():
    """
    Get bank ranking information from the test_data table as a DataFrame
    
    Returns:
        pd.DataFrame: DataFrame containing bank ranking information
    """
    try:
        conn = get_db_connection()
        query = """
        SELECT DISTINCT holder as 'Bank Name', 
                        investment_type as 'Investment Type'
        FROM test_data 
        ORDER BY holder
        """
        bank_ranking_df = pd.read_sql_query(query, conn)
        conn.close()
        return bank_ranking_df
    except Exception as e:
        logger.error(f"Error getting bank ranking: {str(e)}")
        raise

def get_constraints_df():
    """
    Create a constraints DataFrame based on the data
    
    Since the constraints would typically come from the Filter sheet in Excel,
    we're creating a default DataFrame with similar constraints
    
    Returns:
        pd.DataFrame: DataFrame containing optimization constraints
    """
    # Create a default constraints DataFrame similar to the Filter sheet
    constraints = {
        'Bank': {'Max Value': 250000, 'Min Value': 0}
    }
    constraints_df = pd.DataFrame.from_dict(constraints, orient='index')
    return constraints_df

def get_all_data_for_optimization():
    """
    Get all the necessary data for optimization
    
    Returns:
        tuple: (bank_ranking_df, bank_rates_df, constraints_df)
    """
    bank_ranking_df = get_bank_ranking_df()
    bank_rates_df = get_bank_rates_df()
    constraints_df = get_constraints_df()
    
    return bank_ranking_df, bank_rates_df, constraints_df

def get_branches():
    """
    Get a list of all unique branch names from branch_relationships
    
    Returns:
        list: A list of unique branch names
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT branch_name FROM branch_relationships ORDER BY branch_name")
        branches = [row[0] for row in cursor.fetchall()]
        conn.close()
        return branches
    except Exception as e:
        logger.error(f"Error getting branches: {str(e)}")
        return []

def get_associations():
    """
    Get associations and their mapped branches from test_data
    
    Returns:
        list: List of dictionaries with association_name and branch_name
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT 
                association_name, 
                branch_name 
            FROM test_data 
            WHERE association_name IS NOT NULL AND branch_name IS NOT NULL
            ORDER BY association_name
        """)
        associations = [{'association_name': row[0], 'branch_name': row[1]} for row in cursor.fetchall()]
        conn.close()
        return associations
    except Exception as e:
        logger.error(f"Error getting associations: {str(e)}")
        return []

def get_constraints():
    """
    Get all optimization constraints from the database
    
    Returns:
        pd.DataFrame: DataFrame containing constraints data
    """
    try:
        conn = get_db_connection()
        query = """
        SELECT id, category, name, value, weight, is_enabled
        FROM constraints
        ORDER BY category, name
        """
        constraints_df = pd.read_sql_query(query, conn)
        conn.close()
        return constraints_df
    except Exception as e:
        logger.error(f"Error getting constraints: {str(e)}")
        # Return empty DataFrame on error
        return pd.DataFrame(columns=['id', 'category', 'name', 'value', 'weight', 'is_enabled'])

def get_constraints_by_category(category):
    """
    Get optimization constraints for a specific category
    
    Args:
        category: Category of constraints to get
        
    Returns:
        pd.DataFrame: DataFrame containing constraints for the specified category
    """
    try:
        conn = get_db_connection()
        query = """
        SELECT id, category, name, value, weight, is_enabled
        FROM constraints
        WHERE category = ?
        ORDER BY name
        """
        constraints_df = pd.read_sql_query(query, conn, params=(category,))
        conn.close()
        return constraints_df
    except Exception as e:
        logger.error(f"Error getting constraints for category '{category}': {str(e)}")
        # Return empty DataFrame on error
        return pd.DataFrame(columns=['id', 'category', 'name', 'value', 'weight', 'is_enabled'])

def update_constraint(constraint_id, value, weight, is_enabled):
    """
    Update an optimization constraint
    
    Args:
        constraint_id: ID of the constraint to update
        value: New value for the constraint
        weight: New weight for the constraint
        is_enabled: New enabled status for the constraint
        
    Returns:
        bool: True if update was successful, False otherwise
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = """
        UPDATE constraints
        SET value = ?, weight = ?, is_enabled = ?
        WHERE id = ?
        """
        
        cursor.execute(query, (value, weight, is_enabled, constraint_id))
        conn.commit()
        
        success = cursor.rowcount > 0
        conn.close()
        
        return success
    except Exception as e:
        logger.error(f"Error updating constraint {constraint_id}: {str(e)}")
        return False

def reset_constraints_to_default(category=None):
    """
    Reset constraints to their default values
    
    Args:
        category: Optional category to reset. If None, all constraints are reset.
        
    Returns:
        int: Number of constraints reset
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Standardized default values
        # - All product, time, and bank constraints: set to 1.0 (equal values)
        # - Weighting constraints: Both set to 0.5 to maintain sum of 1.0
        # - All category weights set to 1.0
        
        # Reset standard values (weight, enabled status) for the specified category
        if category:
            if category != 'all':
                query = """
                UPDATE constraints
                SET weight = 1.0, is_enabled = 1
                WHERE category = ?
                """
                cursor.execute(query, (category,))
        else:
            query = """
            UPDATE constraints
            SET weight = 1.0, is_enabled = 1
            """
            cursor.execute(query)
        
        # Apply specific default values for each category
        if category == 'product' or category == 'all' or category is None:
            # Set all product constraints to equal value of 1.0
            cursor.execute("""
            UPDATE constraints
            SET value = 1.0
            WHERE category = 'product'
            """)
            
            logger.info(f"Reset product constraints to equal values (1.0)")
            
        if category == 'time' or category == 'all' or category is None:
            # Set all time constraints to equal value of 1.0
            cursor.execute("""
            UPDATE constraints
            SET value = 1.0
            WHERE category = 'time'
            """)
                
            logger.info(f"Reset time constraints to equal values (1.0)")
        
        if category == 'weighting' or category == 'all' or category is None:
            # Weighting constraints - balanced at 0.5 each (must sum to 1.0)
            weighting_defaults = {
                'Interest Rates': 0.5,
                'ECR Return': 0.5
            }
            
            for weight_name, value in weighting_defaults.items():
                cursor.execute("""
                UPDATE constraints
                SET value = ?
                WHERE category = 'weighting' AND name = ?
                """, (value, weight_name))
            
            logger.info(f"Reset weighting constraints to balanced values (0.5 each)")
            
        if category == 'bank' or category == 'all' or category is None:
            # Set all bank constraints to equal value of 1.0
            cursor.execute("""
            UPDATE constraints
            SET value = 1.0
            WHERE category = 'bank'
            """)
            
            logger.info(f"Reset bank constraints to equal values (1.0)")
        
        conn.commit()
        
        count = cursor.rowcount
        conn.close()
        
        return count
    except Exception as e:
        logger.error(f"Error resetting constraints: {str(e)}")
        return 0

def verify_admin_credentials(username, password):
    """
    Verify admin user credentials
    
    Args:
        username: Admin username
        password: Admin password
        
    Returns:
        bool: True if credentials are valid, False otherwise
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = """
        SELECT password_hash, salt
        FROM admin_users
        WHERE username = ? AND is_active = 1
        """
        
        cursor.execute(query, (username,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return False
            
        stored_hash, salt = result
        password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        
        return password_hash == stored_hash
    except Exception as e:
        logger.error(f"Error verifying admin credentials: {str(e)}")
        return False

def change_admin_password(username, new_password):
    """
    Change an admin user's password
    
    Args:
        username: Admin username
        new_password: New password
        
    Returns:
        bool: True if password was changed successfully, False otherwise
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Generate new salt and hash
        salt = os.urandom(32).hex()
        password_hash = hashlib.sha256((new_password + salt).encode()).hexdigest()
        
        query = """
        UPDATE admin_users
        SET password_hash = ?, salt = ?
        WHERE username = ?
        """
        
        cursor.execute(query, (password_hash, salt, username))
        conn.commit()
        
        success = cursor.rowcount > 0
        conn.close()
        
        return success
    except Exception as e:
        logger.error(f"Error changing admin password: {str(e)}")
        return False

def get_constraints_for_optimizer():
    """
    Get constraints formatted for use by the optimizer
    
    Returns:
        dict: Dictionary containing formatted constraints
    """
    try:
        constraints_df = get_constraints()
        if constraints_df.empty:
            return {}
            
        # Group constraints by category
        result = {}
        
        # Process product constraints
        product_df = constraints_df[constraints_df['category'] == 'product']
        if not product_df.empty:
            result['product'] = {
                row['name']: {
                    'value': row['value'],
                    'weight': row['weight'],
                    'enabled': bool(row['is_enabled'])
                } for _, row in product_df.iterrows()
            }
            
        # Process time constraints
        time_df = constraints_df[constraints_df['category'] == 'time']
        if not time_df.empty:
            result['time'] = {
                row['name']: {
                    'value': row['value'],
                    'weight': row['weight'],
                    'enabled': bool(row['is_enabled'])
                } for _, row in time_df.iterrows()
            }
            
        # Process weighting constraints
        weighting_df = constraints_df[constraints_df['category'] == 'weighting']
        if not weighting_df.empty:
            result['weighting'] = {
                row['name']: {
                    'value': row['value'],
                    'weight': row['weight'],
                    'enabled': bool(row['is_enabled'])
                } for _, row in weighting_df.iterrows()
            }
            
        # Process bank constraints
        bank_df = constraints_df[constraints_df['category'] == 'bank']
        if not bank_df.empty:
            result['bank'] = {
                row['name']: {
                    'value': row['value'],
                    'weight': row['weight'],
                    'enabled': bool(row['is_enabled'])
                } for _, row in bank_df.iterrows()
            }
            
        return result
    except Exception as e:
        logger.error(f"Error formatting constraints for optimizer: {str(e)}")
        return {} 