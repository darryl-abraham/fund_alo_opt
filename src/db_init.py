import sqlite3
from pathlib import Path

# Define the database path
DATABASE = str(Path(__file__).parent.parent / "data" / "langston.db")

def ensure_data_directory_exists():
    """Create the data directory if it doesn't exist."""
    data_dir = Path(DATABASE).parent
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Ensured data directory exists: {data_dir}")

def init_db():
    """
    Initialize the database with required tables if they don't exist.
    """
    ensure_data_directory_exists()
    
    print(f"Initializing database at: {DATABASE}")
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Create constraints table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS constraints (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category TEXT NOT NULL,
        name TEXT NOT NULL,
        value REAL DEFAULT 1.0,
        weight REAL DEFAULT 1.0,
        is_enabled INTEGER DEFAULT 1,
        UNIQUE(category, name)
    )
    ''')
    
    # Check if we need to populate with default constraints
    cursor.execute("SELECT COUNT(*) FROM constraints")
    count = cursor.fetchone()[0]
    
    if count == 0:
        # Product constraints
        products = [
            ('product', 'CD', 8.0, 1.0, 1),
            ('product', 'Checking', 5.0, 1.0, 1),
            ('product', 'Savings', 3.0, 1.0, 1)
        ]
        cursor.executemany(
            "INSERT INTO constraints (category, name, value, weight, is_enabled) VALUES (?, ?, ?, ?, ?)",
            products
        )
        
        # Time constraints
        times = [
            ('time', 'Short Term (1-3 months)', 3.0, 1.0, 1),
            ('time', 'Mid Term (4-6 months)', 5.0, 1.0, 1),
            ('time', 'Long Term (7-12 months)', 8.0, 1.0, 1)
        ]
        cursor.executemany(
            "INSERT INTO constraints (category, name, value, weight, is_enabled) VALUES (?, ?, ?, ?, ?)",
            times
        )
        
        # Weighting factors
        weightings = [
            ('weighting', 'Interest Rates', 0.7, 1.0, 1),
            ('weighting', 'ECR Return', 0.3, 1.0, 1)
        ]
        cursor.executemany(
            "INSERT INTO constraints (category, name, value, weight, is_enabled) VALUES (?, ?, ?, ?, ?)",
            weightings
        )
        
        # Bank constraints - just a few examples
        banks = [
            ('bank', 'Bank United', 7.0, 1.0, 1),
            ('bank', 'City National', 6.0, 1.0, 1),
            ('bank', 'First Citizens Bank', 8.0, 1.0, 1),
            ('bank', 'Pacific Premier Bank', 5.0, 1.0, 1)
        ]
        cursor.executemany(
            "INSERT INTO constraints (category, name, value, weight, is_enabled) VALUES (?, ?, ?, ?, ?)",
            banks
        )
        
        print("Initialized constraints table with default values.")
    
    conn.commit()
    conn.close()
    print("Database initialization complete.")

def ensure_consistent_category_weights():
    """
    Ensure that all constraints in the same category have the same weight value.
    This should be run after updating to the new category-weight model.
    """
    try:
        print(f"Using database at: {DATABASE}")
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        # Get all distinct categories
        cursor.execute("SELECT DISTINCT category FROM constraints")
        categories = [row[0] for row in cursor.fetchall()]
        
        for category in categories:
            # Get the average weight for this category
            cursor.execute(
                "SELECT AVG(weight) FROM constraints WHERE category = ?",
                (category,)
            )
            avg_weight = cursor.fetchone()[0]
            
            # Update all constraints in this category to have the same weight
            cursor.execute(
                "UPDATE constraints SET weight = ? WHERE category = ?",
                (avg_weight, category)
            )
            
            print(f"Standardized weights for category '{category}' to {avg_weight}")
        
        conn.commit()
        conn.close()
        print("Category weights have been standardized successfully.")
    except Exception as e:
        print(f"Error ensuring consistent category weights: {str(e)}")

def update_to_category_weights():
    """
    Update the database to support the category-wide weight model.
    This is a one-time migration.
    """
    ensure_consistent_category_weights()
    print("Database updated to support category-wide weights.")

# Call this when you need to update to the new model
if __name__ == "__main__":
    # Initialize the database
    init_db()
    
    # Update to category weights model
    update_to_category_weights()
    
    print("Database is now ready to use with the new category-weight model.") 