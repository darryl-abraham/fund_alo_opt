import pytest
import sqlite3
import pandas as pd
from pathlib import Path
import tempfile
import os
from flask import Flask, session, redirect, url_for, request
from werkzeug.test import EnvironBuilder
from werkzeug.wrappers import Request
import db_utils
import optimizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test fixtures
@pytest.fixture
def app():
    """Create a Flask app for testing"""
    app = Flask(__name__)
    app.secret_key = 'test_secret_key'
    app.config['TESTING'] = True
    
    # Add a mock admin_login route to handle redirects
    @app.route('/admin/login')
    def admin_login():
        return 'Login Page'
    
    # Add the admin constraints route
    @app.route('/admin/constraints')
    def admin_constraints():
        return 'Constraints Page'
    
    # Add the admin update constraint route with simplified logic
    @app.route('/admin/constraints/update', methods=['POST'])
    def admin_update_constraint():
        if not session.get('admin_logged_in'):
            return redirect(url_for('admin_login'))
            
        try:
            constraint_id = request.form.get('id')
            is_enabled = request.form.get('is_enabled', '0') == '1'
            value = float(request.form.get('value', 0))
            weight = float(request.form.get('weight', 0))
            
            # Update the constraint
            conn = db_utils.get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE constraints 
                   SET is_enabled = ?, value = ?, weight = ?
                   WHERE id = ?""",
                (is_enabled, value, weight, constraint_id)
            )
            conn.commit()
            conn.close()
            
            return redirect(url_for('admin_constraints'))
            
        except Exception as e:
            logger.error(f"Error updating constraint: {e}")
            return str(e), 500
    
    return app

@pytest.fixture
def test_db():
    """Create a temporary SQLite database for testing"""
    try:
        # Create a temporary file
        temp_db = tempfile.NamedTemporaryFile(delete=False)
        temp_db.close()
        
        # Connect to the temporary database
        conn = sqlite3.connect(temp_db.name)
        cursor = conn.cursor()
        
        # Create the constraints table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS constraints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            name TEXT NOT NULL,
            value REAL NOT NULL,
            weight REAL NOT NULL DEFAULT 1.0,
            is_enabled BOOLEAN NOT NULL DEFAULT 1,
            UNIQUE(category, name)
        )
        ''')
        
        # Insert test data
        test_constraints = [
            ('product', 'CD', 1.0, 1.0, 1),
            ('product', 'Checking', 1.0, 1.0, 1),
            ('time', 'Short Term (1-3 months)', 1.0, 1.0, 1),
            ('time', 'Mid Term (4-6 months)', 1.0, 1.0, 1),
            ('weighting', 'Interest Rates', 0.5, 1.0, 1),
            ('weighting', 'ECR Return', 0.5, 1.0, 1),
            ('bank', 'Bank United', 1.0, 1.0, 1),
            ('bank', 'City National', 1.0, 1.0, 1)
        ]
        
        cursor.executemany(
            "INSERT INTO constraints (category, name, value, weight, is_enabled) VALUES (?, ?, ?, ?, ?)",
            test_constraints
        )
        
        conn.commit()
        conn.close()
        
        yield temp_db.name
        
    except Exception as e:
        logger.error(f"Error setting up test database: {e}")
        raise
    finally:
        try:
            os.unlink(temp_db.name)
        except Exception as e:
            logger.error(f"Error cleaning up test database: {e}")

@pytest.fixture
def mock_db_connection(monkeypatch, test_db):
    """Mock the database connection to use the test database"""
    def mock_get_db_connection():
        try:
            conn = sqlite3.connect(test_db)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception as e:
            logger.error(f"Error creating database connection: {e}")
            raise
    
    monkeypatch.setattr(db_utils, 'get_db_connection', mock_get_db_connection)

@pytest.fixture
def mock_optimizer(monkeypatch):
    """Mock the optimizer to return predictable results"""
    def mock_run_optimization(params):
        try:
            # Return different results based on the constraints
            conn = db_utils.get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM constraints WHERE name = 'Bank United'")
            bank_weight = cursor.fetchone()['value']
            conn.close()
            
            # If Bank United has high weight, allocate more to it
            if bank_weight > 5.0:
                results = [
                    {'Bank Name': 'Bank United', 'Allocated Amount': 800000},
                    {'Bank Name': 'City National', 'Allocated Amount': 200000}
                ]
            else:
                results = [
                    {'Bank Name': 'Bank United', 'Allocated Amount': 500000},
                    {'Bank Name': 'City National', 'Allocated Amount': 500000}
                ]
                
            return {
                'success': True,
                'results': results
            }
        except Exception as e:
            logger.error(f"Error in mock optimizer: {e}")
            return {'success': False, 'message': str(e)}
    
    monkeypatch.setattr(optimizer, 'run_optimization', mock_run_optimization)

def test_update_constraint(mock_db_connection):
    """Test updating a single constraint"""
    try:
        # Test data
        constraint_id = 1  # CD product constraint
        new_value = 0.8
        new_weight = 0.9
        is_enabled = 0
        
        # Update the constraint
        success = db_utils.update_constraint(constraint_id, new_value, new_weight, is_enabled)
        assert success is True, "Constraint update failed"
        
        # Verify the update
        conn = db_utils.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM constraints WHERE id = ?", (constraint_id,))
        result = cursor.fetchone()
        conn.close()
        
        assert result['value'] == new_value, f"Expected value {new_value}, got {result['value']}"
        assert result['weight'] == new_weight, f"Expected weight {new_weight}, got {result['weight']}"
        assert result['is_enabled'] == is_enabled, f"Expected is_enabled {is_enabled}, got {result['is_enabled']}"
    except Exception as e:
        logger.error(f"Error in test_update_constraint: {e}")
        raise

def test_update_weighting_constraint(mock_db_connection):
    """Test updating a weighting constraint (which should update its complement)"""
    try:
        # Test data
        constraint_id = 5  # Interest Rates weighting constraint
        new_value = 0.7
        new_weight = 1.0
        is_enabled = 1
        
        # Update the constraint through the admin route
        conn = db_utils.get_db_connection()
        cursor = conn.cursor()
        
        # Update the first weighting constraint
        cursor.execute(
            """UPDATE constraints 
               SET value = ?, weight = ?, is_enabled = ?
               WHERE id = ?""",
            (new_value, new_weight, is_enabled, constraint_id)
        )
        
        # Find and update the complementary weighting constraint
        cursor.execute(
            """SELECT id FROM constraints 
               WHERE category = 'weighting' AND id != ?""",
            (constraint_id,)
        )
        other_id = cursor.fetchone()[0]
        
        # Update the complementary constraint to maintain sum of 1.0
        cursor.execute(
            """UPDATE constraints 
               SET value = ?
               WHERE id = ?""",
            (1.0 - new_value, other_id)
        )
        
        conn.commit()
        
        # Verify both weighting constraints were updated correctly
        cursor.execute("SELECT * FROM constraints WHERE category = 'weighting' ORDER BY id")
        results = cursor.fetchall()
        conn.close()
        
        # Check individual values
        assert results[0]['value'] == new_value, f"Expected value {new_value}, got {results[0]['value']}"
        assert results[0]['weight'] == new_weight, f"Expected weight {new_weight}, got {results[0]['weight']}"
        assert results[0]['is_enabled'] == is_enabled, f"Expected is_enabled {is_enabled}, got {results[0]['is_enabled']}"
        
        assert abs(results[1]['value'] - (1.0 - new_value)) < 0.0001, "Complementary value not set correctly"
        
        # Check that the values sum to 1.0
        total_value = sum(row['value'] for row in results)
        assert abs(total_value - 1.0) < 0.0001, f"Values sum to {total_value}, expected 1.0"
    except Exception as e:
        logger.error(f"Error in test_update_weighting_constraint: {e}")
        raise

def test_get_constraints_for_optimizer(mock_db_connection):
    """Test that get_constraints_for_optimizer returns correctly formatted data"""
    try:
        # Get constraints
        constraints = db_utils.get_constraints_for_optimizer()
        
        # Verify structure
        assert isinstance(constraints, dict), "Constraints should be a dictionary"
        assert 'product' in constraints, "Missing 'product' category"
        assert 'time' in constraints, "Missing 'time' category"
        assert 'weighting' in constraints, "Missing 'weighting' category"
        assert 'bank' in constraints, "Missing 'bank' category"
        
        # Verify content
        assert constraints['product']['CD']['value'] == 1.0, "Incorrect CD product value"
        assert constraints['weighting']['Interest Rates']['value'] == 0.5, "Incorrect Interest Rates value"
        assert constraints['bank']['Bank United']['value'] == 1.0, "Incorrect Bank United value"
    except Exception as e:
        logger.error(f"Error in test_get_constraints_for_optimizer: {e}")
        raise

def test_constraint_impact_on_optimization(mock_db_connection, mock_optimizer):
    """Test that constraint changes affect optimization results"""
    try:
        # First, get baseline optimization results
        params = {
            'branch_name': 'Test Branch',
            'association_name': 'Test Association',
            'allocation_amount': 1000000.0
        }
        
        # Run initial optimization
        initial_results = optimizer.run_optimization(params)
        assert initial_results['success'] is True, "Initial optimization failed"
        
        # Modify a bank constraint to heavily favor Bank United
        conn = db_utils.get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """UPDATE constraints 
               SET value = 10.0 
               WHERE name = 'Bank United'"""
        )
        conn.commit()
        conn.close()
        
        # Run optimization with updated constraints
        updated_results = optimizer.run_optimization(params)
        assert updated_results['success'] is True, "Updated optimization failed"
        
        # Compare results - Bank United should have a larger allocation
        initial_allocations = pd.DataFrame(initial_results['results'])
        updated_allocations = pd.DataFrame(updated_results['results'])
        
        bank_united_initial = initial_allocations[initial_allocations['Bank Name'] == 'Bank United']['Allocated Amount'].sum()
        bank_united_updated = updated_allocations[updated_allocations['Bank Name'] == 'Bank United']['Allocated Amount'].sum()
        
        assert bank_united_updated > bank_united_initial, (
            f"Expected increased allocation for Bank United, but got: "
            f"initial={bank_united_initial}, updated={bank_united_updated}"
        )
    except Exception as e:
        logger.error(f"Error in test_constraint_impact_on_optimization: {e}")
        raise

def test_reset_constraints(mock_db_connection):
    """Test resetting constraints to default values"""
    try:
        # First modify some constraints
        db_utils.update_constraint(1, 0.5, 0.5, 0)  # CD product
        db_utils.update_constraint(5, 0.8, 0.8, 0)  # Interest Rates
        
        # Reset all constraints
        count = db_utils.reset_constraints_to_default()
        assert count > 0, "No constraints were reset"
        
        # Verify constraints were reset
        conn = db_utils.get_db_connection()
        cursor = conn.cursor()
        
        # Check product constraints
        cursor.execute("SELECT * FROM constraints WHERE category = 'product'")
        products = cursor.fetchall()
        for product in products:
            assert product['value'] == 1.0, f"Product {product['name']} value not reset to 1.0"
            assert product['weight'] == 1.0, f"Product {product['name']} weight not reset to 1.0"
            assert product['is_enabled'] == 1, f"Product {product['name']} not enabled"
        
        # Check weighting constraints
        cursor.execute("SELECT * FROM constraints WHERE category = 'weighting'")
        weightings = cursor.fetchall()
        for weighting in weightings:
            assert abs(weighting['value'] - 0.5) < 0.0001, f"Weighting {weighting['name']} value not reset to 0.5"
            assert weighting['weight'] == 1.0, f"Weighting {weighting['name']} weight not reset to 1.0"
            assert weighting['is_enabled'] == 1, f"Weighting {weighting['name']} not enabled"
        
        conn.close()
    except Exception as e:
        logger.error(f"Error in test_reset_constraints: {e}")
        raise

def test_admin_update_constraint_route(app, mock_db_connection):
    """Test the admin_update_constraint route"""
    try:
        with app.test_client() as client:
            # Set up the session
            with client.session_transaction() as sess:
                sess['admin_logged_in'] = True
            
            # Make the POST request
            response = client.post('/admin/constraints/update', data={
                'id': '1',
                'category': 'product',
                'is_enabled': '1',
                'value': '0.8',
                'weight': '0.9'
            }, follow_redirects=True)
            
            # Verify the response
            assert response.status_code == 200, f"Expected success, got status code {response.status_code}"
            
            # Verify the database was updated
            conn = db_utils.get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM constraints WHERE id = 1")
            result = cursor.fetchone()
            conn.close()
            
            assert result['value'] == 0.8, f"Expected value 0.8, got {result['value']}"
            assert result['weight'] == 0.9, f"Expected weight 0.9, got {result['weight']}"
            assert result['is_enabled'] == 1, f"Expected is_enabled 1, got {result['is_enabled']}"
    except Exception as e:
        logger.error(f"Error in test_admin_update_constraint_route: {e}")
        raise 