#!/usr/bin/env python3
"""
Test optimizer with all constraint categories to ensure they work properly
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from optimizer import run_optimization
import json

def test_optimizer_all_categories():
    """Test optimizer with all constraint categories"""
    
    print("Testing optimizer with all constraint categories...")
    print("=" * 60)
    
    # Test parameters
    test_params = {
        'branch_name': 'Alliance Association Management, Inc.',
        'association_name': 'Test Association',
        'allocation_amount': 1000000  # $1M test allocation
    }
    
    print(f"Test parameters: {json.dumps(test_params, indent=2)}")
    print("\nRunning optimization...")
    
    try:
        # Run optimization
        result = run_optimization(test_params)
        
        if result['success']:
            print("✅ Optimization completed successfully!")
            print(f"Message: {result['message']}")
            print(f"Results: {len(result['results'])} allocations")
            print(f"Bank count: {result['bank_count']}")
            print(f"Term count: {result['term_count']}")
            
            # Show summary
            summary = result['summary']
            print(f"\nSummary:")
            print(f"  Total allocated: ${summary['total_allocated']:,.2f}")
            print(f"  Total return: ${summary['total_return']:,.2f}")
            print(f"  Weighted avg rate: {summary['weighted_avg_rate']:.3f}%")
            print(f"  Total funds: ${summary['total_funds']:,.2f}")
            print(f"  ECR gain: ${summary['ecr_gain']:,.2f}")
            
            # Show first few results
            print(f"\nFirst 3 allocations:")
            for i, allocation in enumerate(result['results'][:3]):
                print(f"  {i+1}. {allocation['Bank Name']} - {allocation['CD Term']}: ${allocation['Allocated Amount']:,.0f} @ {allocation['CD Rate']:.3f}%")
            
        else:
            print("❌ Optimization failed!")
            print(f"Message: {result['message']}")
            
    except Exception as e:
        print(f"❌ Error during optimization: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_optimizer_all_categories()
