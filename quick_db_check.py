#!/usr/bin/env python3
import sqlite3

conn = sqlite3.connect('data/langston.db')
cursor = conn.cursor()

# Check liquidity constraints
cursor.execute("SELECT * FROM constraints WHERE category = 'liquidity'")
liquidity = cursor.fetchall()
print(f"Liquidity constraints: {len(liquidity)}")
for row in liquidity:
    print(f"  {row}")

# Check all constraints
cursor.execute("SELECT category, COUNT(*) FROM constraints GROUP BY category")
categories = cursor.fetchall()
print(f"\nAll categories:")
for cat, count in categories:
    print(f"  {cat}: {count}")

conn.close()
