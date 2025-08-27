import sys
import os
sys.path.append('src')

import db_utils
import optimizer

def test_new_constraint_system():
    print("=== TESTING NEW 100% CONSTRAINT SYSTEM ===\n")
    
    try:
        # Test 1: Get constraints and verify structure
        print("1. Getting constraints from database...")
        constraints = db_utils.get_constraints_for_optimizer()
        
        if not constraints:
            print("   ❌ No constraints found")
            return
            
        print(f"   ✅ Found constraints for {len(constraints)} categories")
        
        # Test 2: Check category weights
        print("\n2. Category weights (should sum to 100%):")
        total_weight = 0
        for category, category_constraints in constraints.items():
            if category_constraints:
                # Get weight from first constraint in category
                first_constraint = next(iter(category_constraints.values()))
                category_weight = first_constraint['weight']
                total_weight += category_weight
                print(f"   {category}: {category_weight:.3f} ({category_weight*100:.1f}%)")
        
        print(f"   Total: {total_weight:.3f} ({total_weight*100:.1f}%)")
        
        if abs(total_weight - 1.0) < 0.01:
            print("   ✅ Category weights sum to 100%")
        else:
            print(f"   ⚠️  Category weights sum to {total_weight*100:.1f}% (should be 100%)")
        
        # Test 3: Check individual constraint values
        print("\n3. Individual constraint values (should be 0-1 scale):")
        for category, category_constraints in constraints.items():
            print(f"   {category.upper()}:")
            for name, data in category_constraints.items():
                if data['enabled']:
                    print(f"     {name}: {data['value']:.3f} ({data['value']*100:.1f}%)")
        
        # Test 4: Test optimization with constraints
        print("\n4. Testing optimization with new constraint system...")
        
        # Create test parameters
        test_params = {
            'branch_name': 'Test Branch',
            'association_name': 'Test Association'
        }
        
        # Run optimization (this will use the new constraint system)
        print("   Running optimization...")
        results = optimizer.run_optimization(params=test_params)
        
        if results['success']:
            print("   ✅ Optimization completed successfully")
            print(f"   Allocated ${results['summary']['total_allocated']:,.2f}")
            print(f"   Used {len(results['results'])} allocation combinations")
        else:
            print(f"   ❌ Optimization failed: {results['message']}")
            
        print("\n✅ New constraint system test completed!")
        
    except Exception as e:
        print(f"❌ Error testing new constraint system: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_new_constraint_system()
