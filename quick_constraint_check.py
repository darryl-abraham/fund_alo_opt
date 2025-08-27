#!/usr/bin/env python3
import sqlite3

conn = sqlite3.connect('data/langston.db')
cursor = conn.cursor()

print("Current constraint state:")
print("=" * 50)

# Check all constraints
cursor.execute("SELECT category, name, value, weight, is_enabled FROM constraints ORDER BY category, name")
rows = cursor.fetchall()

current_category = ""
for row in rows:
    category, name, value, weight, enabled = row
    if category != current_category:
        current_category = category
        print(f"\n{category.upper()}:")
    
    status = "ENABLED" if enabled else "DISABLED"
    print(f"  {name}: {status} (value: {value:.3f}, weight: {weight:.3f})")

conn.close()
