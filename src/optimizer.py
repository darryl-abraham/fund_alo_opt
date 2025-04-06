import pandas as pd
from ortools.linear_solver import pywraplp
from typing import Tuple, Optional, Dict, List, Any
import logging
from pathlib import Path
import io
import json
import sqlite3
import db_utils

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database Tables
TABLE_TEST_DATA = 'test_data'
TABLE_CD_RATES = 'cd_rates'
TABLE_BRANCH_RELATIONSHIPS = 'branch_relationships'
TABLE_ECR_RATES = 'ecr_rates'

# Test Data Columns
class TestDataColumns:
    BRANCH_NAME = 'branch_name'
    ASSOCIATION_NAME = 'association_name'
    HOLDER = 'holder'
    INVESTMENT_TYPE = 'investment_type'
    CURRENT_BALANCE = 'current_balance'

# CD Rates Columns
class CDRatesColumns:
    BANK_NAME = 'bank_name'
    BANK_CODE = 'bank_code'
    CD_TERM = 'cd_term'
    CD_RATE = 'cd_rate'
    CDARS_TERM = 'cdars_term'
    CDARS_RATE = 'cdars_rate'
    SPECIAL = 'special'

# Branch Relationships Columns
class BranchRelationshipsColumns:
    BRANCH_NAME = 'branch_name'
    ALLIANCE_ASSOC_BANK = 'alliance_assoc_bank'
    BANCO_POPULAR = 'banco_popular'
    BANK_UNITED = 'bank_united'
    CITY_NATIONAL = 'city_national'
    ENTERPRISE_BANK_TRUST = 'enterprise_bank_trust'
    FIRST_CITIZENS_BANK = 'first_citizens_bank'
    HARMONY_BANK = 'harmony_bank'
    PACIFIC_PREMIER_BANK = 'pacific_premier_bank'
    PACIFIC_WESTERN = 'pacific_western'
    SOUTHSTATE = 'southstate'
    SUNWEST_BANK = 'sunwest_bank'
    CAPITAL_ONE = 'capital_one'

# ECR Rates Columns
class ECRRatesColumns:
    BANK_NAME = 'bank_name'
    BANK_CODE = 'bank_code'
    ECR_RATE = 'ecr_rate'

def execute_sql_query(query: str) -> pd.DataFrame:
    """
    Execute an SQL query on the database and return results as a DataFrame.
    
    Args:
        query: SQL query string
        
    Returns:
        DataFrame containing query results
        
    Raises:
        Exception: If database connection or query execution fails
    """
    try:
        # Connect to the database
        db_path = Path(__file__).parent.parent / "data" / "langston.db"
        conn = sqlite3.connect(db_path)
        
        # Execute query and return results as DataFrame
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
        
    except Exception as e:
        logger.error(f"Error executing SQL query: {str(e)}")
        raise

def get_database_columns() -> Dict[str, List[str]]:
    """
    Connect to the langston.db database and retrieve all column names for each table.
    
    Returns:
        Dict[str, List[str]]: Dictionary mapping table names to their column names
    """
    try:
        # Connect to the database
        db_path = Path(__file__).parent.parent / "data" / "langston.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        # Dictionary to store table columns
        table_columns = {}
        
        # Get columns for each table
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = [col[1] for col in cursor.fetchall()]
            table_columns[table_name] = columns
            
        conn.close()
        return table_columns
        
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}")
        return {}

# Example SQL queries for common operations
def get_cd_rates(bank_name: Optional[str] = None) -> pd.DataFrame:
    """
    Get CD rates for all banks or a specific bank.
    
    Args:
        bank_name: Optional bank name to filter results
        
    Returns:
        DataFrame with CD rates
    """
    query = f"""
    SELECT {CDRatesColumns.BANK_NAME}, 
           {CDRatesColumns.CD_TERM}, 
           {CDRatesColumns.CD_RATE}
    FROM {TABLE_CD_RATES}
    """
    
    if bank_name:
        query += f" WHERE {CDRatesColumns.BANK_NAME} = '{bank_name}'"
    
    query += f" ORDER BY {CDRatesColumns.BANK_NAME}, {CDRatesColumns.CD_TERM}"
    
    return execute_sql_query(query)

def get_branch_relationships(branch_name: Optional[str] = None) -> pd.DataFrame:
    """
    Get branch relationships for all branches or a specific branch.
    
    Args:
        branch_name: Optional branch name to filter results
        
    Returns:
        DataFrame with branch relationships
    """
    query = f"""
    SELECT *
    FROM {TABLE_BRANCH_RELATIONSHIPS}
    """
    
    if branch_name:
        query += f" WHERE {BranchRelationshipsColumns.BRANCH_NAME} = '{branch_name}'"
    
    return execute_sql_query(query)

def get_investment_data(association_name: Optional[str] = None) -> pd.DataFrame:
    """
    Get investment data for all associations or a specific association.
    
    Args:
        association_name: Optional association name to filter results
        
    Returns:
        DataFrame with investment data
    """
    query = f"""
    SELECT {TestDataColumns.BRANCH_NAME},
           {TestDataColumns.ASSOCIATION_NAME},
           {TestDataColumns.HOLDER},
           {TestDataColumns.INVESTMENT_TYPE},
           {TestDataColumns.INVESTMENT_TERM},
           {TestDataColumns.INVESTMENT_RATE},
           {TestDataColumns.CURRENT_BALANCE}
    FROM {TABLE_TEST_DATA}
    """
    
    if association_name:
        query += f" WHERE {TestDataColumns.ASSOCIATION_NAME} = '{association_name}'"
    
    query += f" ORDER BY {TestDataColumns.ASSOCIATION_NAME}"
    
    return execute_sql_query(query)

# Constants
SHEET_BANK_RANKING = 'Bank Ranking'
SHEET_BANK_RATES = 'Bank Rates'
SHEET_FILTER = 'Filter'
DEFAULT_MAX_BANK_ALLOCATION = 250000

# Term duration mappings (in months)
SHORT_TERM_MIN, SHORT_TERM_MAX = 1, 3
MID_TERM_MIN, MID_TERM_MAX = 4, 6
LONG_TERM_MIN, LONG_TERM_MAX = 7, 12

def load_data_from_database() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and process data from the SQLite database.
        
    Returns:
        Tuple containing three DataFrames:
        - bank_ranking_df: DataFrame with bank rankings from test_data
        - bank_rates_df: DataFrame with CD rates for different terms
        - constraints_df: DataFrame with optimization constraints
    """
    try:
        # Get bank ranking data from test_data
        bank_ranking_query = f"""
        SELECT DISTINCT 
            {CDRatesColumns.BANK_NAME} as "Bank Name",
            COUNT(*) as "Priority"
        FROM {TABLE_CD_RATES}
        WHERE {CDRatesColumns.CD_RATE} IS NOT NULL
        AND {CDRatesColumns.BANK_NAME} IS NOT NULL
        AND {CDRatesColumns.BANK_NAME} != ''
        GROUP BY {CDRatesColumns.BANK_NAME}
        ORDER BY "Priority" DESC
        """
        bank_ranking_df = execute_sql_query(bank_ranking_query)
        
        # Get bank rates data
        bank_rates_query = f"""
        SELECT 
            {CDRatesColumns.BANK_NAME} as "Bank Name",
            {CDRatesColumns.CD_TERM} as "CD Term",
            {CDRatesColumns.CD_RATE} as "CD Rate"
        FROM {TABLE_CD_RATES}
        WHERE {CDRatesColumns.CD_RATE} IS NOT NULL
        AND {CDRatesColumns.BANK_NAME} IS NOT NULL
        AND {CDRatesColumns.BANK_NAME} != ''
        """
        bank_rates_df = execute_sql_query(bank_rates_query)
        
        # Clean and standardize CD terms
        bank_rates_df["CD Term"] = bank_rates_df["CD Term"].str.lower().str.strip()
        bank_rates_df["CD Term Num"] = bank_rates_df["CD Term"].str.extract(r'(\d+)').astype(float)

        # Remove any invalid terms
        bank_rates_df = bank_rates_df[bank_rates_df["CD Term Num"].notna()]
        
        # Create constraints DataFrame with default values
        constraints_data = {
            "Filter Name": ["Bank", "Term", "Rate"],
            "Min Value": [0, 1, 0],
            "Max Value": [250000, 12, 10]
        }
        constraints_df = pd.DataFrame(constraints_data)
        constraints_df.set_index("Filter Name", inplace=True)

        return bank_ranking_df, bank_rates_df, constraints_df
        
    except Exception as e:
        logger.error(f"Error loading data from database: {str(e)}")
        raise

def optimize_fund_allocation(
    bank_ranking_df: pd.DataFrame, 
    bank_rates_df: pd.DataFrame, 
    constraints: dict, 
    branch_name: str,
    association_name: str,
    time_limit_seconds: int = 30
) -> Tuple[pd.DataFrame, float]:
    """
    Optimize fund allocation across banks and terms using a composite score based on:
    1. Category weights (sum to 100%) for each constraint category
    2. Individual constraint values (0-10) within each category
    3. Weighting factors that sum to 1.0 for interest vs ECR
    4. Liquidity reserve percentage (0-100%)
    
    Args:
        bank_ranking_df: DataFrame with bank ranking information
        bank_rates_df: DataFrame with rates information
        constraints: Dictionary with optimization constraints from database
        branch_name: Name of the branch to optimize for
        association_name: Name of the association to optimize for
        time_limit_seconds: Time limit for the solver in seconds
        
    Returns:
        Tuple containing:
        - DataFrame with optimized allocation results
        - Float representing total funds available
    """
    try:
        solver = pywraplp.Solver.CreateSolver('GLOP')
        if not solver:
            raise Exception("GLOP solver is not available.")
            
        solver.set_time_limit(30000)  # 30 seconds in milliseconds
        
        # Get default max bank allocation
        max_bank_allocation = DEFAULT_MAX_BANK_ALLOCATION
        
        # Get bank relationships and filter rates
        branch_relationships = get_branch_relationships(branch_name)
        if branch_relationships.empty:
            raise ValueError(f"No relationships found for branch: {branch_name}")
            
        # Get list of banks with relationships (value = 1)
        related_banks = []
        for bank_col in BranchRelationshipsColumns.__dict__.values():
            if isinstance(bank_col, str) and bank_col != BranchRelationshipsColumns.BRANCH_NAME:
                try:
                    if branch_relationships[bank_col].iloc[0] == 1:
                        bank_name = bank_col.replace('_', ' ').title()
                        related_banks.append(bank_name)
                except KeyError:
                    continue
        
        if not related_banks:
            raise ValueError(f"No bank relationships found for branch: {branch_name}")
        
        # Filter bank rates to only include related banks
        filtered_rates = bank_rates_df[bank_rates_df["Bank Name"].isin(related_banks)].copy()
        
        if filtered_rates.empty:
            raise ValueError(f"No rates found for related banks of branch: {branch_name}")
        
        # 1. Get and normalize category weights to sum to 100%
        category_weights = {
            'product': 0.0,
            'time': 0.0,
            'weighting': 0.0,
            'bank': 0.0,
            'liquidity': 0.0
        }
        
        # First get raw weights
        total_weight = 0
        for category in category_weights:
            if category in constraints:
                category_data = constraints[category]
                if 'weight' in category_data:
                    weight = category_data['weight']  # This is already a percentage (0-100)
                    category_weights[category] = weight
                    total_weight += weight
        
        # Then normalize to sum to 100%
        if total_weight > 0:
            for category in category_weights:
                category_weights[category] = category_weights[category] / total_weight
        
        # 2. Get bank weights (0-10)
        bank_weights = {}
        if 'bank' in constraints:
            bank_constraints = constraints['bank']
            for bank_name, bank_data in bank_constraints.items():
                if bank_data['enabled']:
                    # Normalize 0-10 to 0-1
                    bank_weights[bank_name] = bank_data['value'] / 10.0
        
        # 3. Get time weights (0-10)
        filtered_rates['time_weight'] = 1.0  # Default weight
        if 'time' in constraints:
            time_constraints = constraints['time']
            time_mapping = {
                'Short Term (1-3 months)': (1, 3),
                'Mid Term (4-6 months)': (4, 6),
                'Long Term (7-12 months)': (7, 12)
            }
            
            for time_name, time_data in time_constraints.items():
                if time_data['enabled'] and time_name in time_mapping:
                    time_range = time_mapping[time_name]
                    # Normalize 0-10 to 0-1
                    weight = time_data['value'] / 10.0
                    term_mask = (filtered_rates["CD Term Num"] >= time_range[0]) & (filtered_rates["CD Term Num"] <= time_range[1])
                    filtered_rates.loc[term_mask, 'time_weight'] = weight
        
        # 4. Get interest and ECR weights (sum to 1.0)
        interest_weight = 1.0
        ecr_weight = 0.0
        if 'weighting' in constraints:
            if 'Interest Rates' in constraints['weighting']:
                interest_data = constraints['weighting']['Interest Rates']
                if interest_data['enabled']:
                    interest_weight = interest_data['value']
                    
            if 'ECR Return' in constraints['weighting']:
                ecr_data = constraints['weighting']['ECR Return']
                if ecr_data['enabled']:
                    ecr_weight = ecr_data['value']
        
        # 5. Get liquidity reserve percentage (0-100%)
        liquidity_reserve = 0.3  # Default 30%
        if 'liquidity' in constraints:
            liquidity_constraints = constraints['liquidity']
            if liquidity_constraints and liquidity_constraints[0]['enabled']:
                # Convert percentage to decimal
                liquidity_reserve = liquidity_constraints[0]['value'] / 100.0
        
        # Get total funds and apply liquidity constraint
        query = f"""
        SELECT SUM(current_balance) as total_balance
        FROM {TABLE_TEST_DATA}
        WHERE {TestDataColumns.ASSOCIATION_NAME} = '{association_name}'
        """
        balance_df = execute_sql_query(query)
        total_funds = float(balance_df['total_balance'].iloc[0] or 0) if not balance_df.empty else 0
        
        if total_funds <= 0:
            logger.warning(f"No funds available for association {association_name}")
            return pd.DataFrame(), total_funds

        available_funds = total_funds * (1 - liquidity_reserve)
        
        logger.info(f"Total funds: ${total_funds:,.2f}")
        logger.info(f"Liquidity reserve ({liquidity_reserve*100}%): ${total_funds * liquidity_reserve:,.2f}")
        logger.info(f"Available for allocation: ${available_funds:,.2f}")

        # Create allocation variables
        allocation = {}
        for bank in filtered_rates['Bank Name'].unique():
            bank_subset = filtered_rates[filtered_rates["Bank Name"] == bank]
            for _, row in bank_subset.iterrows():
                term = row["CD Term"]
                allocation[(bank, term)] = solver.IntVar(0, solver.infinity(), f'alloc_{bank}_{term}')
        
        # Add constraints
        solver.Add(sum(allocation.values()) <= available_funds)
        
        for bank in filtered_rates['Bank Name'].unique():
            vars_to_sum = [var for (b, _), var in allocation.items() if b == bank]
            if vars_to_sum:
                solver.Add(sum(vars_to_sum) <= max_bank_allocation)
        
        # Set objective function using composite score
        objective = solver.Objective()
        
        for (bank, term), var in allocation.items():
            rate_row = filtered_rates[(filtered_rates["Bank Name"] == bank) & 
                                    (filtered_rates["CD Term"] == term)]
            
            if not rate_row.empty:
                cd_rate = rate_row["CD Rate"].values[0] / 100
                bank_weight = bank_weights.get(bank, 1.0)
                time_weight = rate_row['time_weight'].values[0]
                
                # Calculate composite score
                composite_score = cd_rate * interest_weight
                
                # Add ECR component if enabled
                if ecr_weight > 0:
                    try:
                        ecr_query = f"""
                        SELECT ecr_rate
                        FROM {TABLE_ECR_RATES}
                        WHERE {ECRRatesColumns.BANK_NAME} = '{bank}'
                        """
                        ecr_df = execute_sql_query(ecr_query)
                        if not ecr_df.empty:
                            ecr_rate = ecr_df[ECRRatesColumns.ECR_RATE].values[0] / 100
                            composite_score += ecr_rate * ecr_weight
                    except Exception as e:
                        logger.warning(f"Error getting ECR rate for {bank}: {str(e)}")
                
                # Get product weight for CDs
                product_weight = 0.0
                if 'product' in constraints:
                    product_constraints = constraints['product']
                    if 'CD' in product_constraints and product_constraints['CD']['enabled']:
                        # Get the CD weight (0-100%) and multiply by category weight
                        product_weight = product_constraints['CD']['value'] * category_weights['product']
                
                # Apply category weights and individual constraint weights
                # If a category weight is 0, treat it as 1 to avoid zeroing out the score
                weighting_weight = category_weights['weighting'] if category_weights['weighting'] > 0 else 1.0
                bank_weight = bank_weight * (category_weights['bank'] if category_weights['bank'] > 0 else 1.0)
                time_weight = time_weight * (category_weights['time'] if category_weights['time'] > 0 else 1.0)
                
                composite_score *= weighting_weight  # Apply weighting category weight
                composite_score *= bank_weight  # Apply bank weights
                composite_score *= time_weight  # Apply time weights
                composite_score *= product_weight if product_weight > 0 else 1.0  # Apply product weight
                
                # Log the components of the score for debugging
                logger.info(f"Score components for {bank} {term}:")
                logger.info(f"  CD Rate: {cd_rate}, Interest Weight: {interest_weight}")
                logger.info(f"  Product Weight: {product_weight}, Product Category Weight: {category_weights['product']}")
                logger.info(f"  Bank Weight: {bank_weight}, Bank Category Weight: {category_weights['bank']}")
                logger.info(f"  Time Weight: {time_weight}, Time Category Weight: {category_weights['time']}")
                logger.info(f"  Weighting Category Weight: {category_weights['weighting']}")
                logger.info(f"  Final Composite Score: {composite_score}")
                
                objective.SetCoefficient(var, composite_score)
        
        objective.SetMaximization()
        
        # Solve and process results
        status = solver.Solve()
        
        if status != pywraplp.Solver.OPTIMAL:
            logger.warning(f"No optimal solution found. Solver status: {status}")
            return pd.DataFrame(), total_funds
        
        # Create results with actual rates for reporting
        results = []
        for (bank, term), var in allocation.items():
            if var.solution_value() > 0:
                allocated_amount = int(var.solution_value())
                cd_rate = filtered_rates[(filtered_rates["Bank Name"] == bank) & 
                                      (filtered_rates["CD Term"] == term)]["CD Rate"].values[0]
                term_months = float(term.split()[0])
                expected_return = (allocated_amount * cd_rate * term_months) / (100 * 12)
                
                results.append([
                    bank,
                    term,
                    allocated_amount,
                    cd_rate,
                    expected_return
                ])
                  
        if not results:
            logger.warning("Optimization completed but no funds were allocated")
            logger.warning("Category weights:")
            for category, weight in category_weights.items():
                logger.warning(f"  {category}: {weight}")
            logger.warning("Bank weights:")
            for bank, weight in bank_weights.items():
                logger.warning(f"  {bank}: {weight}")
            logger.warning("Time weights:")
            logger.warning(filtered_rates[['CD Term', 'time_weight']].to_string())
            return pd.DataFrame(), total_funds
            
        columns = ["Bank Name", "CD Term", "Allocated Amount", "CD Rate", "Expected Return"]
        df = pd.DataFrame(results, columns=columns)
        
        summary_row = pd.DataFrame([[
            "TOTAL",
            "",
            df["Allocated Amount"].sum(),
            None,
            df["Expected Return"].sum()
        ]], columns=columns)
        
        final_df = pd.concat([df, summary_row], ignore_index=True)
        
        return final_df, total_funds
        
    except Exception as e:
        logger.error(f"Optimization error: {str(e)}")
        raise

def run_optimization(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the optimization pipeline and return formatted results.
    
    Args:
        params: Dictionary containing optimization parameters
        
    Returns:
        Dictionary with optimization results and summary information
    """
    try:
        # Load data from database
        bank_ranking_df, bank_rates_df, _ = load_data_from_database()
        
        # Get constraints from database
        constraints = db_utils.get_constraints_for_optimizer()
        
        # Extract parameters with defaults
        branch_name = params.get('branch_name')
        association_name = params.get('association_name')
        
        if not branch_name or not association_name:
            return {
                'success': False,
                'message': 'Both branch name and association name are required for optimization.',
                'results': None
            }
        
        # Run optimization
        result_df, total_funds = optimize_fund_allocation(
            bank_ranking_df,
            bank_rates_df,
            constraints,
            branch_name,
            association_name
        )
        
        if result_df.empty:
            return {
                'success': False,
                'message': 'No allocation solution found with the given criteria.',
                'results': None
            }
            
        # Convert results to dictionary format
        results = result_df.to_dict(orient='records')
        
        # Extract summary row and make it separate
        summary = results[-1]
        results = results[:-1]
        
        return {
            'success': True,
            'message': 'Optimization completed successfully',
            'results': results,
            'summary': {
                'total_allocated': summary['Allocated Amount'],
                'total_return': summary['Expected Return'],
                'weighted_avg_rate': summary['Expected Return'] * 100 / summary['Allocated Amount'] if summary['Allocated Amount'] > 0 else 0,
                'total_funds': total_funds
            },
            'bank_count': len(set(r['Bank Name'] for r in results)),
            'term_count': len(set(r['CD Term'] for r in results))
        }
        
    except Exception as e:
        logger.exception("Error in optimization pipeline")
        return {
            'success': False,
            'message': str(e),
            'results': None
        }

def export_results_to_excel(results: Dict[str, Any]) -> bytes:
    """
    Export optimization results to Excel file.
    
    Args:
        results: Optimization results dictionary
        
    Returns:
        Excel file as bytes
    """
    if not results['success'] or not results['results']:
        df = pd.DataFrame([{'Error': results['message']}])
    else:
        # Create DataFrame from results
        df = pd.DataFrame(results['results'])
        
        # Add summary row
        summary_row = pd.DataFrame([{
            'Bank Name': 'TOTAL',
            'CD Term': '',
            'Allocated Amount': results['summary']['total_allocated'],
            'CD Rate': results['summary']['weighted_avg_rate'],
            'Expected Return': results['summary']['total_return']
        }])
        
        df = pd.concat([df, summary_row], ignore_index=True)
    
    # Write to bytes buffer
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Allocation Results', index=False)
    
    output.seek(0)
    return output.getvalue() 