#!/usr/bin/env python3
"""
Test script to verify that the optimizer properly handles disabled constraints
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from optimizer import optimize_fund_allocation
from db_utils import get_constraints_for_optimizer, get_branches, get_associations
import pandas as pd
import sqlite3

def test_optimizer_with_disabled_constraints():
    """Test that the optimizer properly handles disabled constraints"""
    
    print("Testing optimizer with disabled constraints...")
    print("=" * 50)
    
    # Get current constraints
    constraints = get_constraints_for_optimizer()
    
    print("Current constraints:")
    for category, category_data in constraints.items():
        if isinstance(category_data, dict):
            for name, data in category_data.items():
                enabled_status = "ENABLED" if data['enabled'] else "DISABLED"
                print(f"  {category}.{name}: {enabled_status} (value: {data['value']:.3f})")
    
    # Get sample data for testing
    branches = get_branches()
    associations = get_associations()
    
    if not branches or not associations:
        print("No branches or associations found for testing")
        return
    
    # Use first branch and association for testing
    branch_name = branches[0]  # branches is a list of strings
    association_name = associations[0]['association_name']  # associations is a list of dicts
    
    print(f"\nTesting with branch: {branch_name}")
    print(f"Testing with association: {association_name}")
    
    # Get real data from database
    conn = sqlite3.connect('data/langston.db')
    
    # Get bank ranking data
    bank_ranking_query = """
    SELECT DISTINCT 
        bank_name as "Bank Name",
        COUNT(*) as "Priority"
    FROM cd_rates
    WHERE cd_rate IS NOT NULL
    AND bank_name IS NOT NULL
    AND bank_name != ''
    GROUP BY bank_name
    ORDER BY "Priority" DESC
    LIMIT 5
    """
    bank_ranking_df = pd.read_sql_query(bank_ranking_query, conn)
    
    # Get bank rates data
    bank_rates_query = """
    SELECT 
        bank_name as "Bank Name",
        cd_term as "CD Term",
        cd_rate as "CD Rate"
    FROM cd_rates
    WHERE cd_rate IS NOT NULL
    AND bank_name IS NOT NULL
    AND bank_name != ''
    LIMIT 10
    """
    bank_rates_df = pd.read_sql_query(bank_rates_query, conn)
    
    # Clean and standardize CD terms
    bank_rates_df["CD Term"] = bank_rates_df["CD Term"].str.lower().str.strip()
    bank_rates_df["CD Term Num"] = bank_rates_df["CD Term"].str.extract(r'(\d+)').astype(float)
    
    # Remove any invalid terms
    bank_rates_df = bank_rates_df[bank_rates_df["CD Term Num"].notna()]
    
    conn.close()
    
    print(f"\nUsing {len(bank_ranking_df)} banks and {len(bank_rates_df)} rate combinations")
    print("Sample bank rates:")
    print(bank_rates_df.head().to_string(index=False))
    
    print("\n" + "=" * 50)
    print("Testing optimization with disabled constraints...")
    
    try:
        # Run optimization
        result_df, total_funds = optimize_fund_allocation(
            bank_ranking_df=bank_ranking_df,
            bank_rates_df=bank_rates_df,
            constraints=constraints,
            branch_name=branch_name,
            association_name=association_name,
            custom_allocation=100000,  # Use $100k for testing
            time_limit_seconds=10
        )
        
        if not result_df.empty:
            print("\nOptimization successful!")
            print("Results:")
            print(result_df.to_string(index=False))
            
            # Check if Long Term (7-12 months) allocations are affected
            long_term_allocations = result_df[result_df['CD Term'].str.contains('9|10|11|12')]
            if not long_term_allocations.empty:
                print(f"\nLong Term allocations found: {len(long_term_allocations)}")
                print("This suggests the disabled constraint might still be affecting optimization")
            else:
                print("\nNo Long Term allocations found - disabled constraint working correctly!")
        else:
            print("\nOptimization failed or no solution found")
            
    except Exception as e:
        print(f"\nError during optimization: {str(e)}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_optimizer_with_disabled_constraints()
