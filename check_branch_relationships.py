#!/usr/bin/env python3
"""
Check the branch_relationships table to understand the structure
"""

import sqlite3

def check_branch_relationships():
    """Check the branch_relationships table"""
    
    conn = sqlite3.connect('data/langston.db')
    cursor = conn.cursor()
    
    print("Checking branch_relationships table...")
    print("=" * 50)
    
    # Check table structure
    cursor.execute("PRAGMA table_info(branch_relationships)")
    columns = cursor.fetchall()
    print("Table columns:")
    for col in columns:
        print(f"  {col[1]} ({col[2]})")
    
    print("\n" + "=" * 50)
    
    # Check sample data
    cursor.execute("SELECT * FROM branch_relationships LIMIT 5")
    rows = cursor.fetchall()
    
    print("Sample data:")
    for row in rows:
        print(f"  {row}")
    
    print("\n" + "=" * 50)
    
    # Check specific branch
    cursor.execute("SELECT * FROM branch_relationships WHERE branch_name = 'Alliance Association Management, Inc.'")
    branch_row = cursor.fetchone()
    
    if branch_row:
        print("Alliance Association Management, Inc. relationships:")
        print(f"  {branch_row}")
    else:
        print("No relationships found for Alliance Association Management, Inc.")
    
    conn.close()

if __name__ == "__main__":
    check_branch_relationships()
