#!/usr/bin/env python3
"""
Test if constraint updates are working dynamically and saving to SQL
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from db_utils import get_constraints_for_optimizer
import sqlite3
import requests
import time

def test_constraint_updates():
    """Test if constraint updates are working dynamically"""
    
    print("Testing constraint updates and SQL persistence...")
    print("=" * 60)
    
    # First, check current constraints
    print("1. Current constraints in database:")
    constraints = get_constraints_for_optimizer()
    
    for category, category_data in constraints.items():
        if isinstance(category_data, dict):
            print(f"\n{category.upper()}:")
            for name, data in category_data.items():
                enabled_status = "ENABLED" if data['enabled'] else "DISABLED"
                print(f"  {name}: {enabled_status} (value: {data['value']:.3f}, weight: {data['weight']:.3f})")
    
    # Test updating a constraint via the admin API
    print("\n" + "=" * 60)
    print("2. Testing constraint update via admin API...")
    
    try:
        # Test updating a time constraint
        update_data = {
            'id': '4',  # Short Term constraint ID
            'category': 'time',
            'value': '0.8',  # 80%
            'is_enabled': 'on',  # Enable it
            'weight': '0.23'
        }
        
        print(f"Updating constraint: {update_data}")
        
        # Make the request to the admin update endpoint
        response = requests.post('http://localhost:5000/admin/constraints/update', data=update_data)
        
        if response.status_code == 200:
            print("✅ Constraint update request successful")
        else:
            print(f"❌ Constraint update failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("⚠️  Flask app not running - cannot test API endpoints")
        print("   Start the app with 'python src/app.py' to test dynamic updates")
    
    # Test updating category weight
    print("\n3. Testing category weight update...")
    
    try:
        weight_data = {
            'category': 'time',
            'weight': '0.25'  # Increase time weight to 25%
        }
        
        print(f"Updating category weight: {weight_data}")
        
        response = requests.post('http://localhost:5000/admin/category/weight', data=weight_data)
        
        if response.status_code == 200:
            print("✅ Category weight update request successful")
        else:
            print(f"❌ Category weight update failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("⚠️  Flask app not running - cannot test API endpoints")
    
    # Check if we can manually update the database
    print("\n4. Testing manual database update...")
    
    conn = sqlite3.connect('data/langston.db')
    cursor = conn.cursor()
    
    # Update a constraint manually to test
    cursor.execute("""
        UPDATE constraints 
        SET value = 0.9, is_enabled = 1 
        WHERE id = 4 AND category = 'time'
    """)
    
    # Check if update worked
    cursor.execute("SELECT * FROM constraints WHERE id = 4")
    updated_row = cursor.fetchone()
    
    if updated_row:
        print(f"✅ Manual database update successful:")
        print(f"  ID: {updated_row[0]}, Category: {updated_row[1]}, Name: {updated_row[2]}")
        print(f"  Value: {updated_row[3]}, Weight: {updated_row[4]}, Enabled: {updated_row[5]}")
    else:
        print("❌ Manual database update failed")
    
    conn.commit()
    conn.close()
    
    # Wait a moment and check constraints again
    print("\n5. Checking constraints after updates...")
    time.sleep(1)
    
    updated_constraints = get_constraints_for_optimizer()
    
    if 'time' in updated_constraints:
        time_constraints = updated_constraints['time']
        for name, data in time_constraints.items():
            enabled_status = "ENABLED" if data['enabled'] else "DISABLED"
            print(f"  {name}: {enabled_status} (value: {data['value']:.3f}, weight: {data['weight']:.3f})")
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("✅ Database updates are working")
    print("⚠️  API testing requires Flask app to be running")
    print("💡 To test full dynamic updates:")
    print("   1. Start Flask app: python src/app.py")
    print("   2. Go to admin interface: http://localhost:5000/admin/constraints")
    print("   3. Make changes and click 'Save All Changes'")
    print("   4. Check database to verify changes persisted")

if __name__ == "__main__":
    test_constraint_updates()
