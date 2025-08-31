#!/usr/bin/env python3
"""
Test script to disable a time constraint and verify it works
"""

import sqlite3

def test_disable_constraint():
    """Test disabling a time constraint"""
    
    conn = sqlite3.connect('data/langston.db')
    cursor = conn.cursor()
    
    print("Testing constraint disable functionality...")
    print("=" * 50)
    
    # First, let's disable the Long Term constraint
    print("Disabling Long Term (7-12 months) constraint...")
    cursor.execute("""
        UPDATE constraints 
        SET is_enabled = 0 
        WHERE name = 'Long Term (7-12 months)' AND category = 'time'
    """)
    
    # Check if it was updated
    cursor.execute("""
        SELECT name, is_enabled, value, weight 
        FROM constraints 
        WHERE name = 'Long Term (7-12 months)' AND category = 'time'
    """)
    result = cursor.fetchone()
    
    if result:
        print(f"Updated constraint: {result[0]}, enabled={result[1]}, value={result[2]}, weight={result[3]}")
    else:
        print("Failed to update constraint")
    
    # Now let's check all time constraints
    print("\nAll time constraints after update:")
    cursor.execute("SELECT name, is_enabled, value, weight FROM constraints WHERE category = 'time' ORDER BY name")
    time_rows = cursor.fetchall()
    
    for row in time_rows:
        status = "ENABLED" if row[1] else "DISABLED"
        print(f"  {row[0]}: {status}, value={row[2]}, weight={row[3]}")
    
    # Commit the change
    conn.commit()
    conn.close()
    
    print("\nConstraint disabled successfully!")
    print("Now test the web interface to see if the slider is properly disabled.")

if __name__ == "__main__":
    test_disable_constraint()
