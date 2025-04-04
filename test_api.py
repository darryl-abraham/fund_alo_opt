import requests
import json

def main():
    base_url = "http://localhost:5000"
    branches = []
    associations = []
    
    # Test branches API
    try:
        response = requests.get(f"{base_url}/api/branches")
        data = response.json()
        
        print("\n--- /api/branches endpoint ---")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200 and data.get('success'):
            branches = data.get('branches', [])
            print(f"Found {len(branches)} branches")
            print(f"First 5 branches: {branches[:5] if branches else 'None'}")
        else:
            print(f"Error: {data.get('message', 'Unknown error')}")
    except Exception as e:
        print(f"Error accessing /api/branches: {str(e)}")
    
    # Test associations API
    try:
        response = requests.get(f"{base_url}/api/associations")
        data = response.json()
        
        print("\n--- /api/associations endpoint ---")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200 and data.get('success'):
            associations = data.get('associations', [])
            print(f"Found {len(associations)} associations")
            print(f"First 5 associations: {associations[:5] if associations else 'None'}")
        else:
            print(f"Error: {data.get('message', 'Unknown error')}")
    except Exception as e:
        print(f"Error accessing /api/associations: {str(e)}")
    
    # Test optimize API with a sample branch and association
    if len(branches) > 0 and len(associations) > 0:
        try:
            branch_name = branches[0]
            association_name = associations[0]
            
            print(f"\n--- Testing /api/optimize with branch='{branch_name}' and association='{association_name}' ---")
            
            response = requests.post(
                f"{base_url}/api/optimize",
                data={
                    'branch_name': branch_name,
                    'association_name': association_name
                }
            )
            
            data = response.json()
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200 and data.get('success'):
                print("Optimization successful!")
                print(f"Summary: {json.dumps(data.get('summary', {}), indent=2)}")
                print(f"Results count: {len(data.get('results', []))} allocations")
            else:
                print(f"Error: {data.get('message', 'Unknown error')}")
        except Exception as e:
            print(f"Error testing /api/optimize: {str(e)}")

if __name__ == "__main__":
    main() 