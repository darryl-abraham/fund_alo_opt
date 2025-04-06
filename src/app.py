from flask import Flask, request, jsonify, render_template, redirect, url_for, send_file, session, flash
import os
import io
import uuid
import tempfile
from pathlib import Path
import logging
import datetime
import db_utils
import optimizer
from functools import wraps
import pandas as pd
import sqlite3
from flask import g

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get the current directory and project root
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
template_dir = project_root / 'templates'

# Configure Flask app
app = Flask(__name__, template_folder=str(template_dir))
app.secret_key = os.environ.get('SECRET_KEY', str(uuid.uuid4()))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload (kept for future use)

# Create a variable to track if database is initialized
db_initialized = False

# Add middleware to initialize database on first request
@app.before_request
def before_request_func():
    global db_initialized
    if not db_initialized:
        init_db()
        logger.info("Database initialized through middleware.")
        db_initialized = True

# Add context processor to provide 'now' variable to all templates
@app.context_processor
def inject_now():
    return {'now': datetime.datetime.now()}

# Admin authentication decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_logged_in' not in session or not session['admin_logged_in']:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('admin_login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    """Render the main page of the application"""
    # Get the branch and association lists from the database
    try:
        branches = db_utils.get_branches()
        associations = db_utils.get_associations()
        return render_template('index.html', branches=branches, associations=associations)
    except Exception as e:
        logger.exception("Error loading data for index page")
        flash(f"Error loading data: {str(e)}", 'error')
        return render_template('index.html', branches=[], associations=[])

@app.route('/api/optimize', methods=['POST'])
def api_optimize():
    """API endpoint to run optimization with parameters"""
    try:
        # Parse parameters from the request
        params = {
            'branch_name': request.form.get('branch_name'),
            'association_name': request.form.get('association_name')
        }
        
        # Validate required parameters
        if not params['branch_name'] or not params['association_name']:
            return jsonify({
                'success': False,
                'message': "Both branch_name and association_name are required for optimization."
            }), 400
        
        # Run optimization
        results = optimizer.run_optimization(params=params)
        
        # Return JSON response
        return jsonify(results)
        
    except Exception as e:
        logger.exception("Error in optimization API")
        return jsonify({
            'success': False,
            'message': f"Error processing request: {str(e)}"
        }), 500

@app.route('/api/banks', methods=['GET'])
def get_banks():
    """API endpoint to get available banks from the database"""
    try:
        banks = db_utils.get_available_banks()
        
        return jsonify({
            'success': True,
            'banks': banks
        })
        
    except Exception as e:
        logger.exception("Error getting bank list")
        return jsonify({
            'success': False,
            'message': f"Error accessing database: {str(e)}"
        }), 500

@app.route('/api/branches', methods=['GET'])
def get_branches():
    """API endpoint to get available branches from the database"""
    try:
        branches = db_utils.get_branches()
        
        return jsonify({
            'success': True,
            'branches': branches
        })
        
    except Exception as e:
        logger.exception("Error getting branch list")
        return jsonify({
            'success': False,
            'message': f"Error accessing database: {str(e)}"
        }), 500

@app.route('/api/associations', methods=['GET'])
def get_associations():
    """API endpoint to get available associations from the database"""
    try:
        associations = db_utils.get_associations()
        
        return jsonify({
            'success': True,
            'associations': associations
        })
        
    except Exception as e:
        logger.exception("Error getting association list")
        return jsonify({
            'success': False,
            'message': f"Error accessing database: {str(e)}"
        }), 500

@app.route('/optimize', methods=['GET', 'POST'])
def web_optimize():
    """Web interface for optimization"""
    if request.method == 'POST':
        try:
            # Parse form parameters
            params = {
                'branch_name': request.form.get('branch_name'),
                'association_name': request.form.get('association_name')
            }
            
            # Validate required parameters
            if not params['branch_name'] or not params['association_name']:
                flash("Both branch name and association name are required for optimization.", 'error')
                branches = db_utils.get_branches()
                associations = db_utils.get_associations()
                return render_template('index.html', branches=branches, associations=associations, params=params)
            
            # Store parameters in session
            session['params'] = params
            
            # Run optimization
            results = optimizer.run_optimization(params=params)
            
            if not results['success']:
                flash(f"Optimization failed: {results['message']}", 'error')
                branches = db_utils.get_branches()
                associations = db_utils.get_associations()
                return render_template('index.html', branches=branches, associations=associations, params=params)
            
            # Store results in session
            session['results'] = results
            
            return render_template('results.html', results=results, params=params)
            
        except Exception as e:
            logger.exception("Error in web optimization")
            flash(f"Error: {str(e)}", 'error')
            return redirect(request.url)
            
    # For GET requests, load the branches and associations and display the form
    branches = db_utils.get_branches()
    associations = db_utils.get_associations()
    return render_template('index.html', branches=branches, associations=associations)

@app.route('/download_results')
def download_results():
    """Download optimization results as Excel file"""
    if 'results' not in session:
        flash('No results to download', 'error')
        return redirect(url_for('index'))
    
    results = session['results']
    excel_data = optimizer.export_results_to_excel(results)
    
    return send_file(
        io.BytesIO(excel_data),
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='Optimized_Fund_Allocation.xlsx'
    )

@app.route('/clear')
def clear_session():
    """Clear session data and return to index"""
    session.clear()
    flash('Session cleared', 'info')
    return redirect(url_for('index'))

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    logger.exception("Internal server error")
    flash('An internal error occurred. Please try again later.', 'error')
    return render_template('error.html', error=error), 500

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """Admin login page"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if db_utils.verify_admin_credentials(username, password):
            session['admin_logged_in'] = True
            session['admin_username'] = username
            
            # Redirect to next page if specified, otherwise go to admin dashboard
            next_page = request.args.get('next')
            if next_page and next_page.startswith('/'):
                return redirect(next_page)
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('admin/login.html')

@app.route('/admin/logout')
def admin_logout():
    """Admin logout"""
    session.pop('admin_logged_in', None)
    session.pop('admin_username', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('admin_login'))

@app.route('/admin')
@admin_required
def admin_dashboard():
    """Admin dashboard"""
    return render_template('admin/dashboard.html')

@app.route('/admin/constraints')
@admin_required
def admin_constraints():
    """Admin constraints management page."""
    # Get constraints data
    conn = get_db_connection()
    
    # Get product constraints
    products = pd.read_sql_query(
        "SELECT * FROM constraints WHERE category = 'product' ORDER BY name",
        conn
    )
    
    # Get time constraints
    times = pd.read_sql_query(
        "SELECT * FROM constraints WHERE category = 'time' ORDER BY name",
        conn
    )
    
    # Get weighting constraints
    weightings = pd.read_sql_query(
        "SELECT * FROM constraints WHERE category = 'weighting' ORDER BY name",
        conn
    )
    
    # Get bank constraints
    banks = pd.read_sql_query(
        "SELECT * FROM constraints WHERE category = 'bank' ORDER BY name",
        conn
    )

    # Get liquidity constraints
    liquidity = pd.read_sql_query(
        "SELECT * FROM constraints WHERE category = 'liquidity' ORDER BY name",
        conn
    )
    
    # Get category weights
    product_category_weight = products.iloc[0]['weight'] if not products.empty else 1.0
    time_category_weight = times.iloc[0]['weight'] if not times.empty else 1.0
    weighting_category_weight = weightings.iloc[0]['weight'] if not weightings.empty else 1.0
    bank_category_weight = banks.iloc[0]['weight'] if not banks.empty else 1.0
    liquidity_category_weight = liquidity.iloc[0]['weight'] if not liquidity.empty else 1.0
    
    conn.close()
    
    return render_template(
        'admin/constraints.html',
        products=products,
        times=times,
        weightings=weightings,
        banks=banks,
        liquidity=liquidity,
        product_category_weight=product_category_weight,
        time_category_weight=time_category_weight,
        weighting_category_weight=weighting_category_weight,
        bank_category_weight=bank_category_weight,
        liquidity_category_weight=liquidity_category_weight
    )

@app.route('/admin/constraints/update', methods=['POST'])
@admin_required
def admin_update_constraint():
    """Update a single optimization constraint."""
    try:
        constraint_id = request.form.get('id')
        category = request.form.get('category')
        is_enabled = request.form.get('is_enabled', '0') == '1'
        value = float(request.form.get('value', 0))
        weight = float(request.form.get('weight', 0))
        other_value = request.form.get('other_value')
        
        # Get the constraint from the database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Update the constraint
        cursor.execute(
            """UPDATE constraints 
               SET is_enabled = ?, value = ?, weight = ?
               WHERE id = ?""",
            (is_enabled, value, weight, constraint_id)
        )
        
        # Handle special case for linked values (Interest Rates & ECR Return)
        if other_value and category == 'weighting':
            # Find the other weighting constraint
            cursor.execute(
                """SELECT id FROM constraints 
                   WHERE category = 'weighting' AND id != ?""",
                (constraint_id,)
            )
            other_id = cursor.fetchone()[0]
            
            # Update the linked constraint with the complementary value
            cursor.execute(
                """UPDATE constraints 
                   SET value = ?
                   WHERE id = ?""",
                (float(other_value), other_id)
            )
        
        conn.commit()
        conn.close()
        
        flash('Constraint updated successfully.', 'success')
        return redirect(url_for('admin_constraints'))
    except Exception as e:
        flash(f'Error updating constraint: {str(e)}', 'danger')
        return redirect(url_for('admin_constraints'))

# Add new route for updating category weights
@app.route('/admin/category/weight', methods=['POST'])
@admin_required
def admin_update_category_weight():
    """Update weights for all constraints in a category."""
    try:
        category = request.form.get('category')
        weight = float(request.form.get('weight', 1.0))
        
        # Validate inputs
        if not category or weight < 0:
            return 'Invalid input', 400
            
        # Get the database connection
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Update all constraints in the category
        cursor.execute(
            """UPDATE constraints 
               SET weight = ?
               WHERE category = ?""",
            (weight, category)
        )
        
        conn.commit()
        conn.close()
        
        return 'Category weight updated successfully', 200
    except Exception as e:
        return f'Error updating category weight: {str(e)}', 500

@app.route('/admin/constraints/reset/<category>', methods=['POST'])
@admin_required
def admin_reset_constraints(category):
    """Reset constraints for a category"""
    try:
        # Validate category
        valid_categories = ['product', 'time', 'weighting', 'bank', 'all']
        if category not in valid_categories:
            flash('Invalid category', 'danger')
            return redirect(url_for('admin_constraints'))
            
        # Reset constraints
        if category == 'all':
            count = db_utils.reset_constraints_to_default()
        else:
            count = db_utils.reset_constraints_to_default(category)
            
        flash(f'Reset {count} constraints to default values', 'success')
            
    except Exception as e:
        flash(f'Error: {str(e)}', 'danger')
        
    return redirect(url_for('admin_constraints'))

@app.route('/admin/constraints/reset-ajax', methods=['POST'])
@admin_required
def admin_reset_constraints_ajax():
    """Reset constraints for a category (AJAX version)"""
    try:
        # Get category from form data
        category = request.form.get('category')
        
        # Validate category
        valid_categories = ['product', 'time', 'weighting', 'bank', 'all']
        if category not in valid_categories:
            return jsonify({
                'success': False,
                'message': 'Invalid category'
            })
            
        # Reset constraints
        if category == 'all':
            count = db_utils.reset_constraints_to_default()
        else:
            count = db_utils.reset_constraints_to_default(category)
        
        # Get updated constraint data to return to the client
        conn = get_db_connection()
        
        # Get constraints for the category
        if category == 'all':
            constraints = pd.read_sql_query(
                "SELECT * FROM constraints ORDER BY category, name",
                conn
            )
        else:
            constraints = pd.read_sql_query(
                "SELECT * FROM constraints WHERE category = ? ORDER BY name",
                conn,
                params=(category,)
            )
        
        # Get category weights
        category_weights = {}
        categories = ['product', 'time', 'weighting', 'bank']
        
        for cat in categories:
            if category == 'all' or category == cat:
                weight_query = f"SELECT weight FROM constraints WHERE category = '{cat}' LIMIT 1"
                weight_df = pd.read_sql_query(weight_query, conn)
                if not weight_df.empty:
                    category_weights[cat] = weight_df['weight'].iloc[0]
        
        conn.close()
        
        # Convert constraints DataFrame to dict for JSON response
        constraints_data = constraints.to_dict('records')
        
        return jsonify({
            'success': True,
            'message': f'Reset {count} constraints to default values',
            'data': {
                'category_weights': category_weights,
                'constraints': constraints_data
            }
        })
            
    except Exception as e:
        logger.exception(f"Error in AJAX reset: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

@app.route('/admin/password', methods=['GET', 'POST'])
@admin_required
def admin_change_password():
    """Change admin password"""
    if request.method == 'POST':
        try:
            current_password = request.form.get('current_password')
            new_password = request.form.get('new_password')
            confirm_password = request.form.get('confirm_password')
            
            # Verify current password
            if not db_utils.verify_admin_credentials(session['admin_username'], current_password):
                flash('Current password is incorrect', 'danger')
                return redirect(url_for('admin_change_password'))
                
            # Validate new password
            if not new_password or len(new_password) < 8:
                flash('New password must be at least 8 characters', 'danger')
                return redirect(url_for('admin_change_password'))
                
            # Confirm passwords match
            if new_password != confirm_password:
                flash('New passwords do not match', 'danger')
                return redirect(url_for('admin_change_password'))
                
            # Update password
            if db_utils.change_admin_password(session['admin_username'], new_password):
                flash('Password changed successfully', 'success')
                return redirect(url_for('admin_dashboard'))
            else:
                flash('Failed to change password', 'danger')
                
        except Exception as e:
            flash(f'Error: {str(e)}', 'danger')
            
    return render_template('admin/change_password.html')

def init_db():
    """
    Initialize the database with required tables if they don't exist.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create constraints table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS constraints (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category TEXT NOT NULL,
        name TEXT NOT NULL,
        value REAL DEFAULT 1.0,
        weight REAL DEFAULT 1.0,
        is_enabled INTEGER DEFAULT 1,
        UNIQUE(category, name)
    )
    ''')
    
    # Check if we need to populate with default constraints
    cursor.execute("SELECT COUNT(*) FROM constraints")
    count = cursor.fetchone()[0]
    
    if count == 0:
        # Product constraints - all equal value of 1.0
        products = [
            ('product', 'CD', 1.0, 1.0, 1),
            ('product', 'Checking', 1.0, 1.0, 1),
            ('product', 'Savings', 1.0, 1.0, 1)
        ]
        cursor.executemany(
            "INSERT INTO constraints (category, name, value, weight, is_enabled) VALUES (?, ?, ?, ?, ?)",
            products
        )
        
        # Time constraints - all equal value of 1.0
        times = [
            ('time', 'Short Term (1-3 months)', 1.0, 1.0, 1),
            ('time', 'Mid Term (4-6 months)', 1.0, 1.0, 1),
            ('time', 'Long Term (7-12 months)', 1.0, 1.0, 1)
        ]
        cursor.executemany(
            "INSERT INTO constraints (category, name, value, weight, is_enabled) VALUES (?, ?, ?, ?, ?)",
            times
        )
        
        # Weighting factors - balanced at 0.5 each (sum = 1.0)
        weightings = [
            ('weighting', 'Interest Rates', 0.5, 1.0, 1),
            ('weighting', 'ECR Return', 0.5, 1.0, 1)
        ]
        cursor.executemany(
            "INSERT INTO constraints (category, name, value, weight, is_enabled) VALUES (?, ?, ?, ?, ?)",
            weightings
        )
        
        # Bank constraints - all equal value of 1.0
        banks = [
            ('bank', 'Bank United', 1.0, 1.0, 1),
            ('bank', 'City National', 1.0, 1.0, 1),
            ('bank', 'First Citizens Bank', 1.0, 1.0, 1),
            ('bank', 'Pacific Premier Bank', 1.0, 1.0, 1)
        ]
        cursor.executemany(
            "INSERT INTO constraints (category, name, value, weight, is_enabled) VALUES (?, ?, ?, ?, ?)",
            banks
        )

        # Liquidity constraint - default 30%
        liquidity = [
            ('liquidity', 'Reserve Percentage', 0.3, 1.0, 1)  # 30% default
        ]
        cursor.executemany(
            "INSERT INTO constraints (category, name, value, weight, is_enabled) VALUES (?, ?, ?, ?, ?)",
            liquidity
        )
        
        logger.info("Initialized constraints table with standardized equal values.")
    
    conn.commit()
    conn.close()

# Import necessary modules for database connection
def get_db_connection():
    """Get a connection to the database."""
    conn = sqlite3.connect('data/langston.db')
    conn.row_factory = sqlite3.Row
    return conn

# Create a function to initialize the database on first request
@app.route('/init-db')
def init_db_route():
    """Route to initialize the database explicitly."""
    try:
        init_db()
        return "Database initialized successfully!"
    except Exception as e:
        return f"Error initializing database: {str(e)}", 500

if __name__ == '__main__':
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Initialize database before starting
    with app.app_context():
        init_db()
        logger.info("Database initialized.")
    
    # Run app
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true') 