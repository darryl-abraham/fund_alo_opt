import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import db_utils

def main():
    print("Testing db_utils functions...")
    
    # Test database connection
    try:
        conn = db_utils.get_db_connection()
        print("✓ Database connection successful")
        conn.close()
    except Exception as e:
        print(f"✗ Database connection failed: {str(e)}")
        return
    
    # Test get_branches
    try:
        branches = db_utils.get_branches()
        print(f"\nget_branches() returned {len(branches)} branches")
        print(f"First 5 branches: {branches[:5] if branches else 'None'}")
    except Exception as e:
        print(f"✗ get_branches() failed: {str(e)}")
    
    # Test get_associations
    try:
        associations = db_utils.get_associations()
        print(f"\nget_associations() returned {len(associations)} associations")
        print(f"First 5 associations: {associations[:5] if associations else 'None'}")
    except Exception as e:
        print(f"✗ get_associations() failed: {str(e)}")
    
    # Test get_available_banks
    try:
        banks = db_utils.get_available_banks()
        print(f"\nget_available_banks() returned {len(banks)} banks")
        print(f"First 5 banks: {banks[:5] if banks else 'None'}")
    except Exception as e:
        print(f"✗ get_available_banks() failed: {str(e)}")

if __name__ == "__main__":
    main() 