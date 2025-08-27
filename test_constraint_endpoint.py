import requests
import json

def test_constraint_endpoint():
    print("=== TESTING CONSTRAINT UPDATE ENDPOINT ===\n")
    
    base_url = "http://localhost:5000"
    
    try:
        # Test updating a category weight
        print("1. Testing category weight update...")
        weight_data = {
            'category': 'product',
            'weight': '0.8'
        }
        
        response = requests.post(
            f"{base_url}/admin/category/weight",
            data=weight_data
        )
        
        if response.status_code == 200:
            print("   ✅ Category weight update successful")
            print(f"   Response: {response.text}")
        else:
            print(f"   ❌ Category weight update failed: {response.status_code}")
            print(f"   Response: {response.text}")
        
        # Test updating a constraint
        print("\n2. Testing constraint update...")
        constraint_data = {
            'id': '1',
            'category': 'product',
            'value': '75',  # 75%
            'weight': '0.8'
        }
        
        response = requests.post(
            f"{base_url}/admin/constraints/update",
            data=constraint_data
        )
        
        if response.status_code == 302:  # Redirect after successful update
            print("   ✅ Constraint update successful (redirected)")
        else:
            print(f"   ❌ Constraint update failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
        print("\n✅ Endpoint test completed!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Flask app. Make sure it's running on port 5000.")
    except Exception as e:
        print(f"❌ Error testing endpoints: {str(e)}")

if __name__ == "__main__":
    test_constraint_endpoint()
