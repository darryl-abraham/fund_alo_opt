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
    total_funds: float, 
    branch_name: str,
    association_name: str,
    time_limit_seconds: int = 30
) -> pd.DataFrame:
    """
    Optimize fund allocation across banks and terms to maximize interest.
    Only considers banks where the branch has a relationship (value = 1).
    Uses constraints from the database to guide optimization.
    
    Args:
        bank_ranking_df: DataFrame with bank ranking information
        bank_rates_df: DataFrame with rates information
        constraints: Dictionary with optimization constraints from database
        total_funds: Total funds available for allocation
        branch_name: Name of the branch to optimize for
        association_name: Name of the association to optimize for
        time_limit_seconds: Time limit for the solver in seconds
        
    Returns:
        DataFrame with optimized allocation results
    """
    # Create solver
    try:
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            raise Exception("SCIP solver is not available.")
            
        # Set time limit
        solver.SetTimeLimit(time_limit_seconds * 1000)  # Convert to milliseconds
        
        # Get default max bank allocation from constraints
        max_bank_allocation = DEFAULT_MAX_BANK_ALLOCATION
        
        # Apply bank constraints if available
        bank_weights = {}
        if 'bank' in constraints:
            bank_constraints = constraints['bank']
            for bank_name, bank_data in bank_constraints.items():
                if bank_data['enabled']:
                    bank_weights[bank_name] = bank_data['value'] * bank_data['weight']
        
        # Get bank relationships for the specific branch
        branch_relationships = get_branch_relationships(branch_name)
        if branch_relationships.empty:
            raise ValueError(f"No relationships found for branch: {branch_name}")
            
        # Get list of banks with relationships (value = 1)
        related_banks = []
        for bank_col in BranchRelationshipsColumns.__dict__.values():
            if isinstance(bank_col, str) and bank_col != BranchRelationshipsColumns.BRANCH_NAME:
                try:
                    if branch_relationships[bank_col].iloc[0] == 1:
                        # Convert column name to proper bank name format
                        bank_name = bank_col.replace('_', ' ').title()
                        related_banks.append(bank_name)
                except KeyError:
                    logger.warning(f"Column {bank_col} not found in branch_relationships table")
                    continue
        
        if not related_banks:
            raise ValueError(f"No bank relationships found for branch: {branch_name}")
        
        # Filter bank rates to only include related banks
        filtered_rates = bank_rates_df[bank_rates_df["Bank Name"].isin(related_banks)]
        
        if filtered_rates.empty:
            raise ValueError(f"No rates found for related banks of branch: {branch_name}")
            
        # Apply time constraints if available
        time_weights = {}
        if 'time' in constraints:
            time_constraints = constraints['time']
            
            # Map the constraint names to CD term ranges
            time_mapping = {
                'Short Term (1-3 months)': (1, 3),
                'Mid Term (4-6 months)': (4, 6),
                'Long Term (7-12 months)': (7, 12)
            }
            
            for time_name, time_data in time_constraints.items():
                if time_data['enabled'] and time_name in time_mapping:
                    time_range = time_mapping[time_name]
                    
                    # Filter terms in the appropriate range
                    term_mask = (filtered_rates["CD Term Num"] >= time_range[0]) & (filtered_rates["CD Term Num"] <= time_range[1])
                    filtered_rates.loc[term_mask, 'time_weight'] = time_data['value'] * time_data['weight']
            
            # Default weight for terms without explicit constraints
            if 'time_weight' not in filtered_rates.columns:
                filtered_rates['time_weight'] = 1.0
            else:
                filtered_rates['time_weight'].fillna(1.0, inplace=True)
        else:
            filtered_rates['time_weight'] = 1.0
            
        # Apply product constraints if available
        # Note: This implementation focuses on CD products only since that's what the optimizer handles
        if 'product' in constraints and 'CD' in constraints['product']:
            cd_constraint = constraints['product']['CD']
            if not cd_constraint['enabled']:
                raise ValueError("CD product type is disabled in constraints")
                
        # Create variables for allocation
        allocation = {}
        for bank in related_banks:
            bank_subset = filtered_rates[filtered_rates["Bank Name"] == bank]
            bank_weight = bank_weights.get(bank, 1.0)
            
            for _, row in bank_subset.iterrows():
                term = row["CD Term"]
                time_weight = row.get('time_weight', 1.0)
                allocation[(bank, term)] = solver.IntVar(0, solver.infinity(), f'alloc_{bank}_{term}')
        
        # Add total funds constraint
        solver.Add(sum(allocation.values()) <= total_funds)
        
        # Add bank maximum allocation constraint
        for bank in related_banks:
            vars_to_sum = [var for (b, _), var in allocation.items() if b == bank]
            if vars_to_sum:
                solver.Add(sum(vars_to_sum) <= max_bank_allocation)
        
        # Set objective function (maximize interest weighted by constraints)
        objective = solver.Objective()
        
        # Get interest rate and ECR weighting if available
        interest_weight = 1.0
        ecr_weight = 0.0
        if 'weighting' in constraints:
            if 'Interest Rates' in constraints['weighting']:
                interest_data = constraints['weighting']['Interest Rates']
                if interest_data['enabled']:
                    interest_weight = interest_data['value'] * interest_data['weight']
                    
            if 'ECR Return' in constraints['weighting']:
                ecr_data = constraints['weighting']['ECR Return']
                if ecr_data['enabled']:
                    ecr_weight = ecr_data['value'] * ecr_data['weight']
        
        # Normalize weights to sum to 1.0
        total_weight = interest_weight + ecr_weight
        if total_weight > 0:
            interest_weight /= total_weight
            ecr_weight /= total_weight
        else:
            interest_weight = 1.0
            ecr_weight = 0.0
        
        for (bank, term), var in allocation.items():
            # Get the CD rate
            rate_row = filtered_rates[(filtered_rates["Bank Name"] == bank) & 
                                    (filtered_rates["CD Term"] == term)]
            
            if not rate_row.empty:
                # Get CD rate
                cd_rate = rate_row["CD Rate"].values[0]
                
                # Get bank weight
                bank_weight = bank_weights.get(bank, 1.0)
                
                # Get time weight
                time_weight = rate_row.get('time_weight', 1.0).values[0]
                
                # Calculate weighted coefficient
                weighted_coef = (cd_rate / 100) * interest_weight * bank_weight * time_weight
                
                # Add ECR weight if available
                if ecr_weight > 0:
                    try:
                        ecr_query = f"""
                        SELECT ecr_rate
                        FROM {TABLE_ECR_RATES}
                        WHERE {ECRRatesColumns.BANK_NAME} = '{bank}'
                        """
                        ecr_df = execute_sql_query(ecr_query)
                        if not ecr_df.empty:
                            ecr_rate = ecr_df[ECRRatesColumns.ECR_RATE].values[0]
                            weighted_coef += (ecr_rate / 100) * ecr_weight * bank_weight
                    except Exception as e:
                        logger.warning(f"Error getting ECR rate for {bank}: {str(e)}")
                
                objective.SetCoefficient(var, weighted_coef)
        
        objective.SetMaximization()
        
        # Solve the optimization problem
        status = solver.Solve()
        
        if status != pywraplp.Solver.OPTIMAL:
            logger.warning(f"No optimal solution found. Solver status: {status}")
            return pd.DataFrame()
        
        # Collect results
        results = [(bank, term, int(var.solution_value()), 
                   filtered_rates[(filtered_rates["Bank Name"] == bank) & 
                               (filtered_rates["CD Term"] == term)]["CD Rate"].values[0])
                  for (bank, term), var in allocation.items() 
                  if var.solution_value() > 0]
                  
        if not results:
            logger.warning("Optimization completed but no funds were allocated")
            return pd.DataFrame()
            
        # Create results DataFrame
        df = pd.DataFrame(results, columns=["Bank Name", "CD Term", "Allocated Amount", "CD Rate"])
        df["Expected Return"] = (df["Allocated Amount"] * df["CD Rate"]) / 100
        
        # Add summary row
        summary = pd.DataFrame({
            "Bank Name": ["TOTAL"],
            "CD Term": [""],
            "Allocated Amount": [df["Allocated Amount"].sum()],
            "CD Rate": [None],
            "Expected Return": [df["Expected Return"].sum()]
        })
        
        return pd.concat([df, summary], ignore_index=True)
        
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
        
        # Get total current balance for the association
        query = f"""
        SELECT SUM(current_balance) as total_balance
        FROM {TABLE_TEST_DATA}
        WHERE {TestDataColumns.ASSOCIATION_NAME} = '{association_name}'
        """
        balance_df = execute_sql_query(query)
        total_funds = float(balance_df['total_balance'].iloc[0]) if not balance_df.empty else 0
        
        if total_funds <= 0:
            return {
                'success': False,
                'message': f'No funds available for association: {association_name}',
                'results': None
            }
        
        # Run optimization
        result_df = optimize_fund_allocation(
            bank_ranking_df,
            bank_rates_df,
            constraints,
            total_funds,
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