from flask import Flask, request, jsonify, render_template, redirect, url_for, send_file, session, flash
import os
import io
import uuid
import tempfile
from pathlib import Path
import logging
from datetime import datetime
import db_utils
import optimizer
from functools import wraps
import pandas as pd
import sqlite3
from flask import g
import math
import numpy as np
from werkzeug.utils import secure_filename
from flask_session import Session

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
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Configure Flask-Session
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.join(current_dir, 'flask_session')
app.config['SESSION_FILE_THRESHOLD'] = 100  # Number of files to store before cleanup
Session(app)

# Configure upload folder
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')

# Create required directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

# Add abs filter to Jinja2 environment
app.jinja_env.filters['abs'] = abs

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
    return {'now': datetime.now()}

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
            
            # Get branch relationships
            branch_relationships = optimizer.get_branch_relationships(params['branch_name'])
            related_banks = []
            for bank_col in optimizer.BranchRelationshipsColumns.__dict__.values():
                if isinstance(bank_col, str) and bank_col != optimizer.BranchRelationshipsColumns.BRANCH_NAME:
                    try:
                        if branch_relationships[bank_col].iloc[0] == 1:
                            bank_name = bank_col.replace('_', ' ').title()
                            related_banks.append(bank_name)
                    except KeyError:
                        continue
            
            # Store results in session
            session['results'] = results
            
            return render_template('results.html', results=results, params=params, related_banks=related_banks)
            
        except Exception as e:
            logger.exception("Error in web optimization")
            flash(f"Error: {str(e)}", 'error')
            return redirect(request.url)
            
    # For GET requests, load the branches and associations and display the form
    branches = db_utils.get_branches()
    associations = db_utils.get_associations()
    return render_template('index.html', branches=branches, associations=associations)

@app.route('/reoptimize', methods=['POST'])
def reoptimize():
    """Reoptimize with a new allocation amount."""
    try:
        # Get the new allocation amount from the form
        new_allocation = float(request.form.get('allocation_amount', 0))
        logger.info(f"Received reoptimization request with allocation amount: {new_allocation}")
        
        # Get the original parameters from the session
        if 'params' not in session:
            logger.error("No optimization parameters found in session")
            return jsonify({
                'success': False,
                'message': 'No optimization parameters found in session. Please start a new optimization.'
            }), 400
            
        params = session['params'].copy()  # Make a copy to avoid modifying the original
        logger.info(f"Retrieved parameters from session: {params}")
        
        # Validate the new allocation amount
        if new_allocation <= 0:
            logger.error(f"Invalid allocation amount: {new_allocation}")
            return jsonify({
                'success': False,
                'message': 'Allocation amount must be greater than 0.'
            }), 400
            
        # Add the new allocation amount to the parameters
        params['allocation_amount'] = new_allocation
        logger.info(f"Updated parameters with new allocation amount: {params}")
        
        # Run optimization with the new parameters
        logger.info("Starting optimization with new parameters")
        results = optimizer.run_optimization(params=params)
        logger.info(f"Optimization completed with results: {results}")
        
        if not results['success']:
            logger.error(f"Optimization failed: {results['message']}")
            return jsonify({
                'success': False,
                'message': f"Reoptimization failed: {results['message']}"
            }), 400
            
        # Store the new results in the session
        session['results'] = results
        logger.info("Stored new results in session")
        
        # Prepare the response data
        response_data = {
            'success': True,
            'summary': results['summary'],
            'allocations': results['results']
        }
        logger.info(f"Prepared response data: {response_data}")
        
        return jsonify(response_data)
        
    except ValueError as e:
        logger.exception("ValueError in reoptimization:")
        return jsonify({
            'success': False,
            'message': f"Invalid input format: {str(e)}"
        }), 400
    except KeyError as e:
        logger.exception("KeyError in reoptimization:")
        return jsonify({
            'success': False,
            'message': f"Missing required data: {str(e)}"
        }), 400
    except Exception as e:
        logger.exception("Unexpected error in reoptimization:")
        return jsonify({
            'success': False,
            'message': f"Error: {str(e)}"
        }), 500

@app.route('/download_results')
def download_results():
    """Download optimization results as Excel file"""
    if 'results' not in session:
        flash('No results to download', 'error')
        return redirect(url_for('index'))
    
    results = session['results']
    params = session.get('params', {})
    
    # Get branch relationships if available
    branch_relationships = None
    if params.get('branch_name'):
        try:
            branch_relationships = optimizer.get_branch_relationships(params['branch_name'])
            related_banks = []
            for bank_col in optimizer.BranchRelationshipsColumns.__dict__.values():
                if isinstance(bank_col, str) and bank_col != optimizer.BranchRelationshipsColumns.BRANCH_NAME:
                    try:
                        if branch_relationships[bank_col].iloc[0] == 1:
                            bank_name = bank_col.replace('_', ' ').title()
                            related_banks.append(bank_name)
                    except KeyError:
                        continue
            # Add relationships to results
            results['branch_info'] = {
                'branch_name': params['branch_name'],
                'association_name': params.get('association_name'),
                'related_banks': related_banks
            }
        except Exception as e:
            logger.warning(f"Could not get branch relationships: {str(e)}")
    
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
    logger.exception("Internal server error occurred:")
    # Log the full error details
    import traceback
    logger.error(traceback.format_exc())
    flash('An internal error occurred. Please try again later.', 'error')
    return render_template('error.html', error=error), 500

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """Admin login page - Currently bypassing password check"""
    try:
        # Auto-login for development
        session['admin_logged_in'] = True
        session['admin_username'] = 'admin'
        
        # Redirect to next page if specified, otherwise go to admin dashboard
        next_page = request.args.get('next')
        if next_page and next_page.startswith('/'):
            return redirect(next_page)
        return redirect(url_for('admin_dashboard'))
    except Exception as e:
        logger.exception("Error in admin_login:")
        raise

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

    # Get liquidity constraints
    liquidity = pd.read_sql_query(
        "SELECT * FROM constraints WHERE category = 'liquidity' ORDER BY name",
        conn
    )
    
    # Get bank constraints
    banks = pd.read_sql_query(
        "SELECT * FROM constraints WHERE category = 'bank' ORDER BY name",
        conn
    )
    
    # Get category weights (excluding liquidity which doesn't affect scoring)
    product_category_weight = products.iloc[0]['weight'] if not products.empty else 1.0
    time_category_weight = times.iloc[0]['weight'] if not times.empty else 1.0
    weighting_category_weight = weightings.iloc[0]['weight'] if not weightings.empty else 1.0
    bank_category_weight = banks.iloc[0]['weight'] if not banks.empty else 1.0
    
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
        bank_category_weight=bank_category_weight
    )

@app.route('/admin/constraints/update', methods=['POST'])
@admin_required
def admin_update_constraint():
    """Update a single optimization constraint."""
    try:
        constraint_id = request.form.get('id')
        category = request.form.get('category')
        
        # Fix checkbox handling - checkboxes only send value when checked
        # If 'is_enabled' is in form data, it means the checkbox was checked
        is_enabled = 'is_enabled' in request.form
        
        # Convert value from percentage (0-100) to decimal (0-1) for storage
        value_input = request.form.get('value', '0')
        try:
            value = float(value_input) / 100.0  # Convert percentage to decimal
        except ValueError:
            value = 0.0
            
        weight = float(request.form.get('weight', 0))
        other_value = request.form.get('other_value')
        
        logger.info(f"Updating constraint {constraint_id}: enabled={is_enabled}, value={value}, weight={weight}")
        
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
        logger.error(f"Error updating constraint: {str(e)}")
        flash(f'Error updating constraint: {str(e)}', 'danger')
        return redirect(url_for('admin_constraints'))

# Add new route for updating individual constraint toggles and values
@app.route('/admin/constraint/update', methods=['POST'])
@admin_required
def admin_update_constraint_individual():
    """Update individual constraint toggle and value."""
    conn = None
    try:
        constraint_id = request.form.get('id')
        is_enabled = request.form.get('is_enabled') == 'true'
        value_input = request.form.get('value', '0')
        
        # Convert percentage (0-100) to decimal (0-1) for storage
        try:
            value = float(value_input) / 100.0
        except ValueError:
            value = 0.0
            
        logger.info(f"Updating constraint {constraint_id}: enabled={is_enabled}, value={value}")
        
        # Get database connection
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Update the constraint
        cursor.execute(
            """UPDATE constraints 
               SET is_enabled = ?, value = ?
               WHERE id = ?""",
            (is_enabled, value, constraint_id)
        )
        
        conn.commit()
        
        return jsonify({'success': True, 'message': 'Constraint updated successfully'})
            
    except Exception as e:
        logger.error(f"Error updating constraint: {str(e)}")
        if conn:
            conn.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500
    finally:
        if conn:
            conn.close()

# Add new route for handling category toggles
@app.route('/admin/category/toggle', methods=['POST'])
@admin_required
def admin_toggle_category():
    """Toggle a category on/off and handle cascading effects."""
    conn = None
    try:
        category = request.form.get('category')
        is_enabled = request.form.get('enabled') == 'true'
        
        if not category:
            return jsonify({'success': False, 'message': 'Category is required'}), 400
            
        logger.info(f"Toggling category {category}: enabled={is_enabled}")
        
        # Special handling for liquidity constraints (no weight, only affects fund availability)
        if category == 'liquidity':
            conn = get_db_connection()
            cursor = conn.cursor()
            
            if is_enabled:
                # For liquidity, just ensure constraints are enabled (no weight changes)
                cursor.execute(
                    """UPDATE constraints 
                       SET is_enabled = 1
                       WHERE category = ?""",
                    (category,)
                )
            else:
                # Disable liquidity constraints
                cursor.execute(
                    """UPDATE constraints 
                       SET is_enabled = 0
                       WHERE category = ?""",
                    (category,)
                )
            
            conn.commit()
            conn.close()
            
            return jsonify({'success': True, 'message': f'Liquidity category {"enabled" if is_enabled else "disabled"}'})
        
        # Handle other categories with weights
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if is_enabled:
            # Just re-enable UI interaction, don't change existing is_enabled values
            # Update category weight to a default value if it was 0
            cursor.execute(
                """UPDATE constraints 
                   SET weight = CASE 
                       WHEN weight = 0 THEN 0.33 
                       ELSE weight 
                   END
                   WHERE category = ?""",
                (category,)
            )
        else:
            # Cascade disable: set is_enabled=0 for all constraints in this category
            cursor.execute(
                """UPDATE constraints 
                   SET is_enabled = 0, weight = 0
                   WHERE category = ?""",
                (category,)
            )
        
        conn.commit()
        
        return jsonify({'success': True, 'message': f'Category {category} {"enabled" if is_enabled else "disabled"}'})
        
    except Exception as e:
        logger.error(f"Error toggling category: {str(e)}")
        if conn:
            conn.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500
    finally:
        if conn:
            conn.close()

# Add new route for updating category weights
@app.route('/admin/category/weight', methods=['POST'])
@admin_required
def admin_update_category_weight():
    """Update weights for all constraints in a category."""
    conn = None
    try:
        category = request.form.get('category')
        weight_input = request.form.get('weight', '1.0')
        
        # Validate inputs
        if not category:
            return jsonify({'success': False, 'message': 'Invalid category'}), 400
            
        try:
            weight = float(weight_input)
            if weight < 0:
                return jsonify({'success': False, 'message': 'Weight must be non-negative'}), 400
        except ValueError:
            return jsonify({'success': False, 'message': 'Invalid weight value'}), 400
            
        logger.info(f"Updating category weight for {category}: {weight}")
        
        # Get the database connection
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Update all constraints in the category with a single bulk call
        cursor.execute(
            """UPDATE constraints 
               SET weight = ?
               WHERE category = ?""",
            (weight, category)
        )
        
        conn.commit()
        
        return jsonify({'success': True, 'message': f'Category weight updated to {weight:.2f}'})
    except Exception as e:
        logger.error(f"Error updating category weight: {str(e)}")
        if conn:
            conn.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500
    finally:
        if conn:
            conn.close()

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

@app.route('/admin/analysis')
@admin_required
def admin_analysis():
    """Admin analysis dashboard"""
    try:
        # Get analysis results from session if available
        analysis_results = session.get('analysis_results')
        
        # Log the analysis results for debugging
        if analysis_results:
            logger.info("Analysis results found in session")
            logger.debug(f"Analysis results: {analysis_results}")
        else:
            logger.info("No analysis results found in session")
        
        return render_template(
            'admin/analysis.html',
            analysis_results=analysis_results
        )
        
    except Exception as e:
        logger.exception("Error in admin analysis")
        flash(f"Error: {str(e)}", 'error')
        return redirect(url_for('admin_dashboard'))

@app.route('/analyze_excel', methods=['POST'])
def analyze_excel():
    """Handle Excel file upload and perform analysis using database rates"""
    try:
        # Check if portfolio file was uploaded
        if 'portfolio_file' not in request.files:
            return jsonify({'success': False, 'message': 'Please upload a portfolio file'})
        
        portfolio_file = request.files['portfolio_file']
        
        # Check if file is selected
        if portfolio_file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'})
        
        # Check if file is an Excel file
        if not (portfolio_file.filename.endswith('.xlsx') or portfolio_file.filename.endswith('.xls')):
            return jsonify({'success': False, 'message': 'Please upload a valid Excel file'})
        
        # Create uploads directory if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save file temporarily with secure filename
        filename = secure_filename(portfolio_file.filename)
        portfolio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        portfolio_file.save(portfolio_path)
        
        try:
            # Read and analyze the file using database rates
            analysis_results = optimizer.analyze_portfolio(portfolio_path=portfolio_path)
            
            # Store results in session for display
            session['analysis_results'] = analysis_results
            
            # Log success
            logger.info(f"Successfully analyzed portfolio file: {filename}")
            logger.debug(f"Analysis results: {analysis_results}")
            
            return jsonify({'success': True})
            
        except Exception as e:
            logger.exception("Error analyzing portfolio")
            return jsonify({'success': False, 'message': f"Error analyzing portfolio: {str(e)}"})
            
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(portfolio_path):
                    os.remove(portfolio_path)
            except Exception as e:
                logger.warning(f"Error removing temporary file: {str(e)}")
        
    except Exception as e:
        logger.exception("Error in Excel analysis")
        return jsonify({'success': False, 'message': f"Error processing file: {str(e)}"})

@app.route('/download_analysis_report')
def download_analysis_report():
    """Generate and download detailed analysis report"""
    try:
        if 'analysis_results' not in session:
            flash('No analysis results available', 'error')
            return redirect(url_for('admin_analysis'))
        
        # Generate Excel report from analysis results
        analysis_results = session['analysis_results']
        excel_data = optimizer.export_analysis_report(analysis_results)
        
        return send_file(
            io.BytesIO(excel_data),
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='Portfolio_Analysis_Report.xlsx'
        )
        
    except Exception as e:
        logger.exception("Error generating analysis report")
        flash(f"Error generating report: {str(e)}", 'error')
        return redirect(url_for('admin_analysis'))

def init_db():
    """
    Initialize the database with required tables if they don't exist.
    """
    try:
        logger.info("Starting database initialization...")
        
        # Ensure data directory exists
        data_dir = Path('data')
        data_dir.mkdir(exist_ok=True)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        logger.info("Creating tables if they don't exist...")
        
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

        # Create banks table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS banks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bank_name TEXT NOT NULL UNIQUE,
            is_partner INTEGER DEFAULT 0
        )
        ''')

        # Create bank_balances table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS bank_balances (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bank_id INTEGER NOT NULL,
            balance REAL DEFAULT 0.0,
            rate REAL DEFAULT 0.0,
            FOREIGN KEY (bank_id) REFERENCES banks (id)
        )
        ''')

        # Create rates table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS rates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bank_id INTEGER NOT NULL,
            term_category TEXT NOT NULL,
            rate REAL DEFAULT 0.0,
            FOREIGN KEY (bank_id) REFERENCES banks (id)
        )
        ''')

        # Create maturities table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS maturities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bank_id INTEGER NOT NULL,
            maturity_date DATE NOT NULL,
            balance REAL DEFAULT 0.0,
            FOREIGN KEY (bank_id) REFERENCES banks (id)
        )
        ''')
        
        # Check if we need to populate with default constraints
        cursor.execute("SELECT COUNT(*) FROM constraints")
        count = cursor.fetchone()[0]
        
        if count == 0:
            logger.info("Populating default constraints...")
            
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

            # Insert sample banks
            sample_banks = [
                ('Bank United', 1),
                ('City National', 1),
                ('First Citizens Bank', 1),
                ('Pacific Premier Bank', 1),
                ('Non-Partner Bank 1', 0),
                ('Non-Partner Bank 2', 0)
            ]
            cursor.executemany(
                "INSERT OR IGNORE INTO banks (bank_name, is_partner) VALUES (?, ?)",
                sample_banks
            )

            # Insert sample balances
            cursor.execute("SELECT id, bank_name FROM banks")
            bank_ids = cursor.fetchall()
            for bank_id, bank_name in bank_ids:
                cursor.execute(
                    "INSERT INTO bank_balances (bank_id, balance, rate) VALUES (?, ?, ?)",
                    (bank_id, 1000000.0, 4.5 if 'Non-Partner' not in bank_name else 3.5)
                )

            # Insert sample rates
            term_categories = ['Short Term (1-3 months)', 'Mid Term (4-6 months)', 'Long Term (7-12 months)']
            for bank_id, bank_name in bank_ids:
                if 'Non-Partner' not in bank_name:  # Only for partner banks
                    for term in term_categories:
                        rate = 4.5 if 'Short' in term else (5.0 if 'Mid' in term else 5.5)
                        cursor.execute(
                            "INSERT INTO rates (bank_id, term_category, rate) VALUES (?, ?, ?)",
                            (bank_id, term, rate)
                        )

            # Insert sample maturities
            from datetime import datetime, timedelta
            today = datetime.now()
            for bank_id, _ in bank_ids:
                for months in range(1, 13):
                    maturity_date = today + timedelta(days=30*months)
                    cursor.execute(
                        "INSERT INTO maturities (bank_id, maturity_date, balance) VALUES (?, ?, ?)",
                        (bank_id, maturity_date.date(), 500000.0)
                    )
            
            logger.info("Successfully populated default data.")
        
        conn.commit()
        conn.close()
        logger.info("Database initialization completed successfully.")
        
    except Exception as e:
        logger.exception("Error during database initialization:")
        if 'conn' in locals():
            conn.close()
        raise

def get_db_connection():
    """Get a connection to the database."""
    try:
        # Ensure data directory exists
        data_dir = Path('data')
        data_dir.mkdir(exist_ok=True)
        
        # Create full path to database file
        db_path = data_dir / 'langston.db'
        
        # Log the database path being used
        logger.info(f"Connecting to database at: {db_path.absolute()}")
        
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.exception("Error establishing database connection:")
        raise

# Create a function to initialize the database on first request
@app.route('/init-db')
def init_db_route():
    """Route to initialize the database explicitly."""
    try:
        init_db()
        return "Database initialized successfully!"
    except Exception as e:
        return f"Error initializing database: {str(e)}", 500

def get_associations_with_branches():
    """
    Get a list of associations with their corresponding branch names using the branch_association_index table.
    
    Returns:
        list: A list of dictionaries with association and branch names
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = """
        SELECT associations.association_name, branches.branch_name
        FROM branch_association_index
        JOIN associations ON branch_association_index.association_id = associations.id
        JOIN branches ON branch_association_index.branch_id = branches.id
        ORDER BY associations.association_name
        """
        cursor.execute(query)
        associations = [{'association_name': row[0], 'branch_name': row[1]} for row in cursor.fetchall()]
        conn.close()
        return associations
    except Exception as e:
        logger.error(f"Error getting associations with branches: {str(e)}")
        return []

if __name__ == '__main__':
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Initialize database before starting
    with app.app_context():
        init_db()
        logger.info("Database initialized.")
    
    # Run app
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true') 