#!/usr/bin/env python3
"""
Check liquidity constraints specifically
"""

import sqlite3

def check_liquidity():
    """Check liquidity constraints specifically"""
    
    conn = sqlite3.connect('data/langston.db')
    cursor = conn.cursor()
    
    print("Checking liquidity constraints...")
    print("=" * 50)
    
    # Check liquidity constraints
    cursor.execute("SELECT * FROM constraints WHERE category = 'liquidity'")
    liquidity_rows = cursor.fetchall()
    
    if liquidity_rows:
        print(f"Found {len(liquidity_rows)} liquidity constraints:")
        for row in liquidity_rows:
            print(f"  ID: {row[0]}, Category: {row[1]}, Name: {row[2]}, Value: {row[3]}, Weight: {row[4]}, Enabled: {row[5]}")
    else:
        print("No liquidity constraints found")
    
    # Check all constraints to see what we have
    print("\nAll constraints by category:")
    cursor.execute("SELECT category, COUNT(*) as count FROM constraints GROUP BY category ORDER BY category")
    category_counts = cursor.fetchall()
    
    for category, count in category_counts:
        print(f"  {category.upper()}: {count} constraints")
    
    conn.close()

if __name__ == "__main__":
    check_liquidity()
