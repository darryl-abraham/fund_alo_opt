#!/usr/bin/env python
"""
Script to initialize the database with the constraints table and default values.
"""

from src.db_init import init_db, update_to_category_weights

if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    
    print("\nUpdating to category weights model...")
    update_to_category_weights()
    
    print("\nDatabase initialization completed successfully!") 