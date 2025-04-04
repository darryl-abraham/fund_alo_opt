import sqlite3
import hashlib
import os
from pathlib import Path

def create_tables():
    # Get database path
    db_path = Path(__file__).parent.parent / "data" / "langston.db"
    
    # Connect to the database
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Create admin_users table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS admin_users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        salt TEXT NOT NULL,
        is_active BOOLEAN NOT NULL DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create optimization_constraints table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS optimization_constraints (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category TEXT NOT NULL,
        name TEXT NOT NULL,
        value REAL NOT NULL,
        weight REAL NOT NULL DEFAULT 1.0,
        is_enabled BOOLEAN NOT NULL DEFAULT 1,
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(category, name)
    )
    ''')
    
    # Add default admin user if none exists
    cursor.execute("SELECT COUNT(*) FROM admin_users")
    if cursor.fetchone()[0] == 0:
        # Create a default admin user
        username = "admin"
        password = "admin123"
        salt = os.urandom(32).hex()
        password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        
        cursor.execute(
            "INSERT INTO admin_users (username, password_hash, salt) VALUES (?, ?, ?)",
            (username, password_hash, salt)
        )
        print(f"Created default admin user: {username} / {password}")
    
    # Add default constraints if none exist
    cursor.execute("SELECT COUNT(*) FROM optimization_constraints")
    if cursor.fetchone()[0] == 0:
        # Default constraints for product types
        product_constraints = [
            ("product", "Money Market", 1.0, 1.0, "Weight for Money Market accounts"),
            ("product", "Checking", 1.0, 1.0, "Weight for Checking accounts"),
            ("product", "CD", 1.0, 1.0, "Weight for CD accounts")
        ]
        
        # Default constraints for time periods
        time_constraints = [
            ("time", "Short Term (1-3 months)", 1.0, 1.0, "Weight for Short Term investments"),
            ("time", "Mid Term (4-6 months)", 1.0, 1.0, "Weight for Mid Term investments"),
            ("time", "Long Term (7-12 months)", 1.0, 1.0, "Weight for Long Term investments")
        ]
        
        # Default constraints for weighting factors
        weighting_constraints = [
            ("weighting", "ECR Return", 0.5, 1.0, "Weight for ECR return (0.0-1.0)"),
            ("weighting", "Interest Rates", 0.5, 1.0, "Weight for Interest rates (0.0-1.0)")
        ]
        
        # Default constraints for banks
        bank_constraints = [
            ("bank", "Alliance Assoc. Bank", 1.0, 1.0, "Weight for Alliance Assoc. Bank"),
            ("bank", "Banco Popular", 1.0, 1.0, "Weight for Banco Popular"),
            ("bank", "Bank United", 1.0, 1.0, "Weight for Bank United"),
            ("bank", "City National", 1.0, 1.0, "Weight for City National"),
            ("bank", "Enterprise Bank Trust", 1.0, 1.0, "Weight for Enterprise Bank Trust"),
            ("bank", "First Citizens Bank", 1.0, 1.0, "Weight for First Citizens Bank"),
            ("bank", "Harmony Bank", 1.0, 1.0, "Weight for Harmony Bank"),
            ("bank", "Pacific Premier Bank", 1.0, 1.0, "Weight for Pacific Premier Bank"),
            ("bank", "Pacific Western", 1.0, 1.0, "Weight for Pacific Western"),
            ("bank", "Southstate", 1.0, 1.0, "Weight for Southstate"),
            ("bank", "Sunwest Bank", 1.0, 1.0, "Weight for Sunwest Bank"),
            ("bank", "Capital One", 1.0, 1.0, "Weight for Capital One")
        ]
        
        # Combine all constraints
        all_constraints = product_constraints + time_constraints + weighting_constraints + bank_constraints
        
        # Insert default constraints
        cursor.executemany(
            "INSERT INTO optimization_constraints (category, name, value, weight, description) VALUES (?, ?, ?, ?, ?)",
            all_constraints
        )
        
        print(f"Added {len(all_constraints)} default constraints")
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    print("Database tables created successfully")

if __name__ == "__main__":
    create_tables() 