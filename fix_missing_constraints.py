#!/usr/bin/env python3
"""
Fix missing constraints and weights - exclude liquidity from category weights
"""

import sqlite3

def fix_missing_constraints():
    """Fix missing constraints and weights - exclude liquidity from category weights"""
    
    conn = sqlite3.connect('data/langston.db')
    cursor = conn.cursor()
    
    print("Fixing missing constraints and weights...")
    print("=" * 50)
    
    # Fix product category weights - they should have some weight
    print("1. Fixing product category weights...")
    cursor.execute("""
        UPDATE constraints 
        SET weight = 0.20 
        WHERE category = 'product'
    """)
    
    # Check if update worked
    cursor.execute("SELECT * FROM constraints WHERE category = 'product'")
    product_rows = cursor.fetchall()
    print("Product constraints after update:")
    for row in product_rows:
        print(f"  {row[2]}: weight={row[4]}, enabled={row[5]}")
    
    # Add missing liquidity constraints (but with weight 0 - excluded from category total)
    print("\n2. Adding missing liquidity constraints...")
    
    # Check if liquidity constraints exist
    cursor.execute("SELECT COUNT(*) FROM constraints WHERE category = 'liquidity'")
    liquidity_count = cursor.fetchone()[0]
    
    if liquidity_count == 0:
        # Add liquidity constraint with weight 0 (excluded from category total)
        cursor.execute("""
            INSERT INTO constraints (category, name, value, weight, is_enabled, description, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
        """, ('liquidity', 'Liquidity Reserve', 0.30, 0.00, 1, 'Liquidity reserve percentage'))
        
        print("  Added liquidity constraint: Liquidity Reserve (30%, weight: 0.00)")
        print("  Note: Liquidity weight is 0.00 - excluded from category weight total")
    else:
        print("  Liquidity constraints already exist")
    
    # Now redistribute the weights to make the main categories sum to 100%
    # Main categories: Product, Time, Weighting, Bank (liquidity excluded)
    print("\n3. Redistributing main category weights to sum to 100%...")
    
    # Current weights after fixing product: Product (20%), Time (23%), Weighting (17%), Bank (60%)
    # Total: 120% - we need to reduce to 100%
    
    # Reduce bank weight from 60% to 40% to make total 100%
    cursor.execute("""
        UPDATE constraints 
        SET weight = 0.40 
        WHERE category = 'bank'
    """)
    
    print("  Reduced bank weight from 60% to 40%")
    print("  New distribution: Product (20%) + Time (23%) + Weighting (17%) + Bank (40%) = 100%")
    print("  Liquidity excluded from category weight total (weight: 0.00)")
    
    # Check final weights
    print("\n4. Final constraint weights:")
    cursor.execute("SELECT category, weight FROM constraints GROUP BY category ORDER BY category")
    final_weights = cursor.fetchall()
    
    # Calculate total excluding liquidity
    main_category_weights = {}
    liquidity_weight = 0.0
    
    for category, weight in final_weights:
        if category == 'liquidity':
            liquidity_weight = weight
            print(f"  {category.upper()}: {weight*100:.1f}% (EXCLUDED from category total)")
        else:
            main_category_weights[category] = weight
            print(f"  {category.upper()}: {weight*100:.1f}%")
    
    main_total_weight = sum(main_category_weights.values())
    print(f"\nMain category weight total (excluding liquidity): {main_total_weight*100:.1f}%")
    
    if abs(main_total_weight - 1.0) < 0.01:
        print("  ✅ Main category weights sum to 100% correctly!")
    else:
        print(f"  ⚠️  Main category weights don't sum to 100%: {main_total_weight*100:.1f}%")
    
    print(f"Liquidity weight: {liquidity_weight*100:.1f}% (separate constraint)")
    
    # Commit changes
    conn.commit()
    conn.close()
    
    print("\nConstraints fixed successfully!")

if __name__ == "__main__":
    fix_missing_constraints()
