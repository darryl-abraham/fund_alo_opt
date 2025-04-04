import sqlite3
from pathlib import Path

def main():
    db_path = Path("data/langston.db")
    
    if not db_path.exists():
        print(f"Database file not found at: {db_path.absolute()}")
        return
    
    print(f"Database file found at: {db_path.absolute()}")
    print(f"File size: {db_path.stat().st_size} bytes")
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [table[0] for table in cursor.fetchall()]
        print(f"\nTables in database ({len(tables)}): {tables}")
        
        # Check for specific tables needed by the optimizer
        required_tables = ['branch_relationships', 'test_data', 'cd_rates', 'ecr_rates']
        for table in required_tables:
            if table in tables:
                print(f"✓ Required table '{table}' exists")
            else:
                print(f"✗ Required table '{table}' is MISSING")
        
        # Check specific tables and columns for our application
        check_branch_relationships_table(conn)
        check_test_data_table(conn)
        
        conn.close()
    except Exception as e:
        print(f"Error accessing database: {str(e)}")

def check_branch_relationships_table(conn):
    print("\n--- Checking branch_relationships table ---")
    try:
        cursor = conn.cursor()
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='branch_relationships'")
        if not cursor.fetchone():
            print("Table 'branch_relationships' does not exist!")
            return
            
        # Get columns
        cursor.execute("PRAGMA table_info(branch_relationships)")
        columns = [column[1] for column in cursor.fetchall()]
        print(f"Columns: {columns}")
        
        # Check for branch_name column
        if 'branch_name' not in columns:
            print("✗ Required column 'branch_name' is MISSING!")
            return
            
        # Count rows
        cursor.execute("SELECT COUNT(*) FROM branch_relationships")
        count = cursor.fetchone()[0]
        print(f"Number of rows: {count}")
        
        # Show sample data
        if count > 0:
            cursor.execute("SELECT * FROM branch_relationships LIMIT 2")
            sample = cursor.fetchall()
            print(f"Sample data (2 rows):")
            for row in sample:
                print(f"  {row}")
        else:
            print("✗ Table is EMPTY!")
    except Exception as e:
        print(f"Error checking branch_relationships table: {str(e)}")

def check_test_data_table(conn):
    print("\n--- Checking test_data table ---")
    try:
        cursor = conn.cursor()
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='test_data'")
        if not cursor.fetchone():
            print("Table 'test_data' does not exist!")
            return
            
        # Get columns
        cursor.execute("PRAGMA table_info(test_data)")
        columns = [column[1] for column in cursor.fetchall()]
        print(f"Columns: {columns}")
        
        # Check for association_name column
        if 'association_name' not in columns:
            print("✗ Required column 'association_name' is MISSING!")
            return
            
        # Count rows
        cursor.execute("SELECT COUNT(*) FROM test_data")
        count = cursor.fetchone()[0]
        print(f"Number of rows: {count}")
        
        # Show sample data
        if count > 0:
            cursor.execute("SELECT * FROM test_data LIMIT 2")
            sample = cursor.fetchall()
            print(f"Sample data (2 rows):")
            for row in sample:
                print(f"  {row}")
        else:
            print("✗ Table is EMPTY!")
            
        # Count distinct associations
        cursor.execute("SELECT COUNT(DISTINCT association_name) FROM test_data")
        assoc_count = cursor.fetchone()[0]
        print(f"Number of distinct associations: {assoc_count}")
        
        if assoc_count > 0:
            cursor.execute("SELECT DISTINCT association_name FROM test_data LIMIT 5")
            assocs = cursor.fetchall()
            print(f"Sample associations: {[a[0] for a in assocs]}")
        
    except Exception as e:
        print(f"Error checking test_data table: {str(e)}")

if __name__ == "__main__":
    main() 