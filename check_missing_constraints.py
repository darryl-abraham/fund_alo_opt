#!/usr/bin/env python3
"""
Check what's missing in the constraints table
"""

import sqlite3

def check_missing_constraints():
    """Check what's missing in the constraints table"""
    
    conn = sqlite3.connect('data/langston.db')
    cursor = conn.cursor()
    
    print("Checking constraints table for missing categories...")
    print("=" * 50)
    
    # Check all constraints
    cursor.execute("SELECT * FROM constraints ORDER BY category, name")
    rows = cursor.fetchall()
    
    print("All constraints in database:")
    for row in rows:
        print(f"  ID: {row[0]}, Category: {row[1]}, Name: {row[2]}, Value: {row[3]}, Weight: {row[4]}, Enabled: {row[5]}")
    
    print("\n" + "=" * 50)
    
    # Check for missing categories
    expected_categories = ['product', 'time', 'weighting', 'bank', 'liquidity']
    found_categories = set()
    
    for row in rows:
        found_categories.add(row[1])
    
    missing_categories = set(expected_categories) - found_categories
    if missing_categories:
        print(f"Missing categories: {missing_categories}")
    else:
        print("All expected categories found")
    
    # Check specific issues
    print("\n" + "=" * 50)
    print("Specific issues found:")
    
    # Check product category weight
    cursor.execute("SELECT * FROM constraints WHERE category = 'product'")
    product_rows = cursor.fetchall()
    if product_rows:
        print(f"Product constraints found: {len(product_rows)}")
        for row in product_rows:
            print(f"  {row[2]}: weight={row[4]}, enabled={row[5]}")
    else:
        print("No product constraints found")
    
    # Check liquidity category
    cursor.execute("SELECT * FROM constraints WHERE category = 'liquidity'")
    liquidity_rows = cursor.fetchall()
    if liquidity_rows:
        print(f"Liquidity constraints found: {len(liquidity_rows)}")
        for row in liquidity_rows:
            print(f"  {row[2]}: weight={row[4]}, enabled={row[5]}")
    else:
        print("No liquidity constraints found")
    
    conn.close()

if __name__ == "__main__":
    check_missing_constraints()
