#!/usr/bin/env python3
"""
Check the actual bank names in the cd_rates table
"""

import sqlite3

def check_bank_names():
    """Check the actual bank names in the cd_rates table"""
    
    conn = sqlite3.connect('data/langston.db')
    cursor = conn.cursor()
    
    print("Checking bank names in cd_rates table...")
    print("=" * 50)
    
    # Get all unique bank names
    cursor.execute("SELECT DISTINCT bank_name FROM cd_rates ORDER BY bank_name")
    bank_names = cursor.fetchall()
    
    print("All bank names in cd_rates table:")
    for bank in bank_names:
        print(f"  '{bank[0]}'")
    
    print("\n" + "=" * 50)
    
    # Check what the optimizer would convert the column names to
    print("What the optimizer converts column names to:")
    column_names = [
        'alliance_assoc_bank',
        'banco_popular', 
        'bank_united',
        'city_national',
        'enterprise_bank_trust',
        'first_citizens_bank',
        'harmony_bank',
        'pacific_premier_bank',
        'pacific_western',
        'southstate',
        'sunwest_bank',
        'capital_one'
    ]
    
    for col in column_names:
        converted = col.replace('_', ' ').title()
        print(f"  {col} -> '{converted}'")
    
    conn.close()

if __name__ == "__main__":
    check_bank_names()
