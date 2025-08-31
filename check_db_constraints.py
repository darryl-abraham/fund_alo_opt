#!/usr/bin/env python3
"""
Check the actual database values for constraints
"""

import sqlite3

def check_constraints_db():
    """Check the actual database values for constraints"""
    
    conn = sqlite3.connect('data/langston.db')
    cursor = conn.cursor()
    
    print("Checking constraints table in database...")
    print("=" * 50)
    
    # Check the constraints table structure
    cursor.execute("PRAGMA table_info(constraints)")
    columns = cursor.fetchall()
    print("Constraints table columns:")
    for col in columns:
        print(f"  {col[1]} ({col[2]})")
    
    print("\n" + "=" * 50)
    
    # Check all constraints
    cursor.execute("SELECT * FROM constraints ORDER BY category, name")
    rows = cursor.fetchall()
    
    print("All constraints in database:")
    for row in rows:
        print(f"  ID: {row[0]}, Category: {row[1]}, Name: {row[2]}, Value: {row[3]}, Weight: {row[4]}, Enabled: {row[5]}")
    
    print("\n" + "=" * 50)
    
    # Check specific time constraints
    print("Time constraints specifically:")
    cursor.execute("SELECT * FROM constraints WHERE category = 'time' ORDER BY name")
    time_rows = cursor.fetchall()
    
    for row in time_rows:
        print(f"  {row[2]}: enabled={row[5]}, value={row[3]}, weight={row[4]}")
    
    conn.close()

if __name__ == "__main__":
    check_constraints_db()
