#!/usr/bin/env python
"""
Migration script to update the database to use category-wide weights.
This standardizes weights across all constraints in the same category.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import the migration function
from db_init import update_to_category_weights

def main():
    print("Running migration to update to category-wide weight model...")
    print("This will standardize weights across all constraints in the same category.")
    
    # Confirm with the user
    confirmation = input("Do you want to continue? [y/N]: ")
    if confirmation.lower() != 'y':
        print("Migration aborted.")
        return
    
    # Run the migration
    update_to_category_weights()
    
    print("Migration completed successfully.")
    print("Please restart the application for changes to take effect.")

if __name__ == "__main__":
    main() 