"""
Account Maturity Filter System

This module implements a comprehensive system to:
1. Scan for accounts approaching maturity in a 3-month window
2. Apply business rule filters to identify target candidates
3. Delegate optimization to the main optimizer for each candidate
4. Generate results table and summary report

The maturity filter focuses on scanning, filtering, and batch processing,
while the actual optimization is handled by the main optimizer to ensure
consistency and avoid code duplication.
"""

import pandas as pd
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import optimizer
import db_utils

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MaturityFilter:
    """
    Main class for account maturity filtering and batch processing system.
    
    This class handles scanning for maturing accounts, applying business filters,
    and coordinating batch optimization through the main optimizer.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the maturity filter.
        
        Args:
            db_path: Path to the database file. If None, uses default path.
        """
        if db_path is None:
            current_dir = Path(__file__).resolve().parent
            project_root = current_dir.parent
            self.db_path = project_root / "data" / "langston.db"
        else:
            self.db_path = Path(db_path)
            
        self.conn = None
        self.current_rates = None
        self.constraints = None
        
    def get_db_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        if self.conn is None:
            self.conn = sqlite3.connect(str(self.db_path))
            self.conn.row_factory = sqlite3.Row
        return self.conn
    
    def close_connection(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def scan_maturing_accounts(self, months_ahead: int = 3) -> pd.DataFrame:
        """
        Scan for accounts maturing within the specified time window.
        
        Args:
            months_ahead: Number of months ahead to scan for maturing accounts
            
        Returns:
            DataFrame with accounts approaching maturity
        """
        try:
            conn = self.get_db_connection()
            
            # Calculate date range
            today = datetime.now().date()
            end_date = today + timedelta(days=30 * months_ahead)
            
            # Build the scan query
            query = """
            SELECT 
                association_id,
                association_name,
                branch_name,
                summary_account,
                summary_description,
                gl_account_no,
                account_desc,
                holder as bank_name,
                investment_type,
                bank_account,
                purchase_date,
                investment_term,
                maturity_date,
                investment_rate,
                as_of_date,
                current_balance,
                association_report_name,
                -- Calculate days to maturity
                julianday(maturity_date) - julianday('now') as days_to_maturity,
                -- Calculate maturity month
                strftime('%Y-%m', maturity_date) as maturity_month
            FROM test_data 
            WHERE maturity_date IS NOT NULL 
                AND maturity_date BETWEEN ? AND ?
                AND current_balance IS NOT NULL 
                AND current_balance > 0
            ORDER BY maturity_date, current_balance DESC
            """
            
            # Execute query
            df = pd.read_sql_query(query, conn, params=[today, end_date])
            
            # Convert maturity_date to datetime for easier manipulation
            df['maturity_date'] = pd.to_datetime(df['maturity_date'])
            df['days_to_maturity'] = df['days_to_maturity'].astype(int)
            
            logger.info(f"Found {len(df)} accounts maturing in next {months_ahead} months")
            return df
            
        except Exception as e:
            logger.error(f"Error scanning maturing accounts: {str(e)}")
            raise
    
    def apply_business_filters(self, accounts_df: pd.DataFrame, 
                             min_balance: float = 100000,
                             max_balance: float = 10000000,
                             allowed_investment_types: Optional[List[str]] = None,
                             excluded_banks: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Apply business rule filters to narrow down target candidates.
        
        Args:
            accounts_df: DataFrame of accounts from scan
            min_balance: Minimum balance threshold
            max_balance: Maximum balance threshold
            allowed_investment_types: List of allowed investment types
            excluded_banks: List of banks to exclude
            
        Returns:
            Filtered DataFrame of target candidates
        """
        try:
            filtered_df = accounts_df.copy()
            
            # Apply balance filters
            filtered_df = filtered_df[
                (filtered_df['current_balance'] >= min_balance) &
                (filtered_df['current_balance'] <= max_balance)
            ]
            
            # Apply investment type filter
            if allowed_investment_types:
                filtered_df = filtered_df[
                    filtered_df['investment_type'].isin(allowed_investment_types)
                ]
            
            # Apply bank exclusion filter
            if excluded_banks:
                filtered_df = filtered_df[
                    ~filtered_df['bank_name'].isin(excluded_banks)
                ]
            
            # Add additional business logic filters
            # Note: Removed the 30-day minimum filter to allow accounts maturing soon to be included
            
            # Filter out accounts with missing critical data
            filtered_df = filtered_df.dropna(subset=['bank_name', 'current_balance', 'maturity_date'])
            
            logger.info(f"Applied business filters: {len(filtered_df)} accounts remain from {len(accounts_df)}")
            return filtered_df
            
        except Exception as e:
            logger.error(f"Error applying business filters: {str(e)}")
            raise
    
    def get_current_rates_and_constraints(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Get current rate curve and constraint set.
        
        Returns:
            Tuple of (rates_df, constraints_dict)
        """
        try:
            # Get constraints from database
            constraints = db_utils.get_constraints_for_optimizer()
            
            # Get current rates from database
            conn = self.get_db_connection()
            
            # Get CD rates
            cd_rates_query = """
            SELECT bank_name, cd_term, cd_rate, cdars_term, cdars_rate, special
            FROM cd_rates
            ORDER BY bank_name, cd_term
            """
            cd_rates = pd.read_sql_query(cd_rates_query, conn)
            
            # Get ECR rates
            ecr_rates_query = """
            SELECT bank_name, bank_code, ecr_rate
            FROM ecr_rates
            ORDER BY bank_name, bank_code
            """
            ecr_rates = pd.read_sql_query(ecr_rates_query, conn)
            
            # Get branch relationships for bank preferences
            branch_rels_query = """
            SELECT * FROM branch_relationships
            """
            branch_relationships = pd.read_sql_query(branch_rels_query, conn)
            
            rates_data = {
                'cd_rates': cd_rates,
                'ecr_rates': ecr_rates,
                'branch_relationships': branch_relationships
            }
            
            logger.info("Retrieved current rates and constraints")
            return rates_data, constraints
            
        except Exception as e:
            logger.error(f"Error getting current rates and constraints: {str(e)}")
            raise
    
    def process_single_account(self, account: pd.Series, 
                              rates_data: Dict, 
                              constraints: Dict) -> Dict[str, Any]:
        """
        Process a single account by delegating optimization to the main optimizer.
        
        This method coordinates the optimization process by:
        1. Preparing parameters for the main optimizer
        2. Calling the main optimizer to ensure consistency
        3. Processing and formatting the results
        
        Args:
            account: Account data as pandas Series
            rates_data: Current rates and bank relationship data (unused, kept for compatibility)
            constraints: Current optimization constraints (unused, kept for compatibility)
            
        Returns:
            Dictionary with optimization results
        """
        try:
            # Use the branch name from the account data
            branch_name = account.get('branch_name')
            if not branch_name:
                # Fallback to a default branch name if not available
                branch_name = "Alliance Association Management, Inc. dba Associa Hill Country"
            
            # Prepare parameters for optimization
            params = {
                'branch_name': branch_name,
                'association_name': account['association_name'],
                'allocation_amount': account['current_balance']
            }
            
            # Run optimization using the existing optimizer
            results = optimizer.run_optimization(params=params)
            
            if not results['success']:
                return {
                    'account_id': account['association_id'],
                    'association_name': account['association_name'],
                    'branch_name': branch_name,
                    'maturity_date': account['maturity_date'],
                    'current_allocation': account['current_balance'],
                    'recommended_allocation': None,
                    'projected_yield_bps': None,
                    'constraint_bindings': [],
                    'notes': f"Optimization failed: {results['message']}",
                    'success': False
                }
            
            # Extract key results
            # Investment rates are stored as decimals (0.05 = 5%), so multiply by 100 to get percentage
            current_yield = account['investment_rate'] * 100 if account['investment_rate'] else 0
            optimized_yield = results['summary']['weighted_avg_rate']
            yield_lift = optimized_yield - current_yield
            
            # Calculate financial metrics
            current_balance = account['current_balance']
            optimized_balance = results['summary']['total_allocated']
            
            # Calculate current expected return (annual)
            current_expected_return = (current_balance * current_yield / 100) if current_yield > 0 else 0
            
            # Get optimized expected return (term-period) and annualize it
            optimized_expected_return_term = results['summary']['total_return']
            
            # Calculate average term length from optimization results
            avg_term_months = 12  # Default to 12 months if no results
            if results.get('results') and len(results['results']) > 0:
                total_term_months = 0
                total_allocated = 0
                for result in results['results']:
                    if result.get('CD Term'):
                        term_months = float(result['CD Term'].split()[0])
                        allocated_amount = result.get('Allocated Amount', 0)
                        total_term_months += term_months * allocated_amount
                        total_allocated += allocated_amount
                
                if total_allocated > 0:
                    avg_term_months = total_term_months / total_allocated
            
            # Annualize the optimized return for fair comparison
            optimized_expected_return_annual = optimized_expected_return_term * (12 / avg_term_months)
            
            # Calculate dollar improvement (now both annualized)
            dollar_improvement = optimized_expected_return_annual - current_expected_return
            
            # Calculate percentage improvement
            percentage_improvement = (yield_lift / current_yield * 100) if current_yield > 0 else 0
            
            # Calculate ECR benefits
            current_ecr_monthly = results['summary'].get('current_ecr_monthly', 0)
            optimized_ecr_monthly = results['summary'].get('optimized_ecr_monthly', 0)
            ecr_benefit_monthly = optimized_ecr_monthly - current_ecr_monthly
            ecr_benefit_annual = ecr_benefit_monthly * 12
            
            # Calculate current product details
            current_product_type = account.get('investment_type', 'Unknown')
            current_amount = current_balance
            current_rate = current_yield
            current_monthly_interest = current_expected_return / 12 if current_expected_return > 0 else 0
            
            # Calculate optimized product details (use the primary allocation from results)
            optimized_product_type = "CD Portfolio"  # Default since optimization creates a portfolio
            optimized_amount = optimized_balance
            optimized_rate = optimized_yield
            optimized_monthly_interest = optimized_expected_return_annual / 12 if optimized_expected_return_annual > 0 else 0
            
            # If we have detailed results, try to get the primary product type
            if results.get('results') and len(results['results']) > 0:
                # Get the largest allocation to determine primary product
                primary_allocation = max(results['results'], key=lambda x: x.get('Allocated Amount', 0))
                if primary_allocation.get('CD Term'):
                    optimized_product_type = f"CD {primary_allocation['CD Term']}"
            
            # Identify binding constraints
            constraint_bindings = []
            if results.get('constraint_violations'):
                constraint_bindings = results['constraint_violations']
            
            return {
                'account_id': account['association_id'],
                'association_name': account['association_name'],
                'branch_name': branch_name,
                'maturity_date': account['maturity_date'],
                'current_allocation': current_balance,
                'recommended_allocation': optimized_balance,
                'projected_yield_bps': yield_lift * 100,  # Keep for backward compatibility
                'constraint_bindings': constraint_bindings,
                'notes': 'Optimization successful',
                'success': True,
                'current_yield': current_yield,
                'optimized_yield': optimized_yield,
                'bank_count': results.get('bank_count', 0),
                'term_count': results.get('term_count', 0),
                # New financial metrics
                'current_expected_return': current_expected_return,
                'optimized_expected_return': optimized_expected_return_annual,
                'dollar_improvement': dollar_improvement,
                'percentage_improvement': percentage_improvement,
                'current_ecr_monthly': current_ecr_monthly,
                'optimized_ecr_monthly': optimized_ecr_monthly,
                'ecr_benefit_monthly': ecr_benefit_monthly,
                'ecr_benefit_annual': ecr_benefit_annual,
                # Product comparison details
                'current_product_type': current_product_type,
                'current_amount': current_amount,
                'current_rate': current_rate,
                'current_monthly_interest': current_monthly_interest,
                'optimized_product_type': optimized_product_type,
                'optimized_amount': optimized_amount,
                'optimized_rate': optimized_rate,
                'optimized_monthly_interest': optimized_monthly_interest
            }
            
        except Exception as e:
            logger.error(f"Error optimizing account {account['association_id']}: {str(e)}")
            return {
                'account_id': account['association_id'],
                'association_name': account['association_name'],
                'branch_name': account.get('branch_name', 'Unknown'),
                'maturity_date': account['maturity_date'],
                'current_allocation': account['current_balance'],
                'recommended_allocation': None,
                'projected_yield_bps': None,
                'constraint_bindings': [],
                'notes': f"Error during optimization: {str(e)}",
                'success': False,
                # New financial metrics (set to None for failed optimizations)
                'current_expected_return': None,
                'optimized_expected_return': None,
                'dollar_improvement': None,
                'percentage_improvement': None,
                'current_ecr_monthly': None,
                'optimized_ecr_monthly': None,
                'ecr_benefit_monthly': None,
                'ecr_benefit_annual': None,
                # Product comparison details (set to None for failed optimizations)
                'current_product_type': account.get('investment_type', 'Unknown'),
                'current_amount': account['current_balance'],
                'current_rate': None,
                'current_monthly_interest': None,
                'optimized_product_type': None,
                'optimized_amount': None,
                'optimized_rate': None,
                'optimized_monthly_interest': None
            }
    
    def run_maturity_filtering(self, 
                              months_ahead: int = 3,
                              min_balance: float = 100000,
                              max_balance: float = 10000000,
                              allowed_investment_types: Optional[List[str]] = None,
                              excluded_banks: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run the complete maturity filtering and batch optimization pipeline.
        
        This method:
        1. Scans for accounts approaching maturity
        2. Applies business rule filters
        3. Delegates optimization to the main optimizer for each account
        4. Generates summary reports
        
        Args:
            months_ahead: Number of months ahead to scan
            min_balance: Minimum balance threshold
            max_balance: Maximum balance threshold
            allowed_investment_types: List of allowed investment types
            excluded_banks: List of banks to exclude
            
        Returns:
            Dictionary with optimization results and summary
        """
        try:
            logger.info("Starting maturity filtering and batch optimization pipeline")
            
            # Step 1: Scan for maturing accounts
            maturing_accounts = self.scan_maturing_accounts(months_ahead)
            
            # Step 2: Apply business filters
            target_candidates = self.apply_business_filters(
                maturing_accounts,
                min_balance=min_balance,
                max_balance=max_balance,
                allowed_investment_types=allowed_investment_types,
                excluded_banks=excluded_banks
            )
            
            # Step 3: Get current rates and constraints
            rates_data, constraints = self.get_current_rates_and_constraints()
            
            # Step 4: Run optimization for each candidate
            optimization_results = []
            successful_optimizations = 0
            failed_optimizations = 0
            
            for idx, account in target_candidates.iterrows():
                logger.info(f"Optimizing account {account['association_id']} ({idx + 1}/{len(target_candidates)})")
                
                result = self.process_single_account(account, rates_data, constraints)
                optimization_results.append(result)
                
                if result['success']:
                    successful_optimizations += 1
                else:
                    failed_optimizations += 1
            
            # Step 5: Create results DataFrame
            results_df = pd.DataFrame(optimization_results)
            
            # Step 6: Generate summary statistics
            summary = self.generate_summary_report(results_df, target_candidates)
            
            # Step 7: Save results to database or CSV
            timestamp = datetime.now().strftime('%Y%m%d')
            self.save_results(results_df, timestamp)
            
            logger.info(f"Maturity optimization completed: {successful_optimizations} successful, {failed_optimizations} failed")
            
            return {
                'success': True,
                'summary': summary,
                'results': results_df.to_dict('records'),
                'target_candidates_count': len(target_candidates),
                'successful_optimizations': successful_optimizations,
                'failed_optimizations': failed_optimizations,
                'timestamp': timestamp
            }
            
        except Exception as e:
            logger.error(f"Error in maturity optimization pipeline: {str(e)}")
            return {
                'success': False,
                'message': str(e),
                'results': None,
                'summary': None
            }
        finally:
            self.close_connection()
    
    def generate_summary_report(self, results_df: pd.DataFrame, 
                              target_candidates: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive summary report.
        
        Args:
            results_df: DataFrame with optimization results
            target_candidates: DataFrame with target candidate accounts
            
        Returns:
            Dictionary with summary statistics
        """
        try:
            # Basic counts
            total_accounts = len(target_candidates)
            successful_optimizations = len(results_df[results_df['success'] == True])
            failed_optimizations = len(results_df[results_df['success'] == False])
            
            # Yield analysis
            successful_results = results_df[results_df['success'] == True]
            if not successful_results.empty:
                total_current_allocation = successful_results['current_allocation'].sum()
                total_recommended_allocation = successful_results['recommended_allocation'].sum()
                
                # Calculate yield improvements
                yield_improvements = successful_results['projected_yield_bps'].dropna()
                avg_yield_lift = yield_improvements.mean() if not yield_improvements.empty else 0
                total_yield_lift_bps = yield_improvements.sum() if not yield_improvements.empty else 0
                
                # Calculate new financial metrics
                total_dollar_improvement = successful_results['dollar_improvement'].sum()
                total_optimized_expected_return = successful_results['optimized_expected_return'].sum()
                total_ecr_benefit_annual = successful_results['ecr_benefit_annual'].sum()
                avg_percentage_improvement = successful_results['percentage_improvement'].mean()
                
                # Calculate product comparison metrics
                total_monthly_interest_improvement = (successful_results['optimized_monthly_interest'] - successful_results['current_monthly_interest']).sum()
                avg_rate_improvement = (successful_results['optimized_rate'] - successful_results['current_rate']).mean()
                
                # Top 10 accounts by monthly interest improvement
                top_accounts = successful_results.nlargest(10, 'optimized_monthly_interest')[
                    ['account_id', 'association_name', 'branch_name', 'maturity_date', 'current_allocation', 
                     'dollar_improvement', 'percentage_improvement', 'optimized_expected_return', 'ecr_benefit_annual',
                     'current_product_type', 'current_amount', 'current_rate', 'current_monthly_interest',
                     'optimized_product_type', 'optimized_amount', 'optimized_rate', 'optimized_monthly_interest']
                ]
                
                # Maturity distribution
                maturity_distribution = target_candidates['maturity_month'].value_counts().sort_index()
                
            else:
                total_current_allocation = 0
                total_recommended_allocation = 0
                avg_yield_lift = 0
                total_yield_lift_bps = 0
                total_dollar_improvement = 0
                total_optimized_expected_return = 0
                total_ecr_benefit_annual = 0
                avg_percentage_improvement = 0
                total_monthly_interest_improvement = 0
                avg_rate_improvement = 0
                top_accounts = pd.DataFrame()
                maturity_distribution = pd.Series()
            
            summary = {
                'total_accounts': total_accounts,
                'successful_optimizations': successful_optimizations,
                'failed_optimizations': failed_optimizations,
                'success_rate': (successful_optimizations / total_accounts * 100) if total_accounts > 0 else 0,
                'total_current_allocation': total_current_allocation,
                'total_recommended_allocation': total_recommended_allocation,
                'avg_yield_lift_bps': avg_yield_lift,
                'total_yield_lift_bps': total_yield_lift_bps,
                'top_10_accounts': top_accounts.to_dict('records') if not top_accounts.empty else [],
                'maturity_distribution': maturity_distribution.to_dict() if not maturity_distribution.empty else {},
                'failed_accounts': results_df[results_df['success'] == False][['account_id', 'association_name', 'branch_name', 'notes']].to_dict('records') if not results_df.empty else [],
                # New financial metrics
                'total_dollar_improvement': total_dollar_improvement,
                'total_optimized_expected_return': total_optimized_expected_return,
                'total_ecr_benefit_annual': total_ecr_benefit_annual,
                'avg_percentage_improvement': avg_percentage_improvement,
                # Product comparison metrics
                'total_monthly_interest_improvement': total_monthly_interest_improvement,
                'avg_rate_improvement': avg_rate_improvement
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary report: {str(e)}")
            return {}
    
    def save_results(self, results_df: pd.DataFrame, timestamp: str):
        """
        Save optimization results to database or CSV.
        
        Args:
            results_df: DataFrame with optimization results
            timestamp: Timestamp string for filename
        """
        try:
            # Save to CSV
            output_dir = Path(__file__).parent.parent / "data" / "maturity_optimization"
            output_dir.mkdir(exist_ok=True)
            
            csv_filename = output_dir / f"maturity_optimization_results_{timestamp}.csv"
            results_df.to_csv(csv_filename, index=False)
            
            logger.info(f"Results saved to {csv_filename}")
            
            # Optionally save to database if analysis schema exists
            try:
                self.save_to_database(results_df, timestamp)
            except Exception as e:
                logger.warning(f"Could not save to database: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
    
    def save_to_database(self, results_df: pd.DataFrame, timestamp: str):
        """
        Save results to database analysis schema.
        
        Args:
            results_df: DataFrame with optimization results
            timestamp: Timestamp string for table suffix
        """
        try:
            conn = self.get_db_connection()
            
            # Create analysis results table if it doesn't exist
            table_name = f"accounts_optimizations_{timestamp}"
            
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id INTEGER,
                association_name TEXT,
                branch_name TEXT,
                maturity_date TEXT,
                current_allocation REAL,
                recommended_allocation REAL,
                projected_yield_bps REAL,
                constraint_bindings TEXT,
                notes TEXT,
                success INTEGER,
                current_yield REAL,
                optimized_yield REAL,
                bank_count INTEGER,
                term_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            
            conn.execute(create_table_sql)
            
            # Insert results
            for _, row in results_df.iterrows():
                conn.execute(f"""
                INSERT INTO {table_name} (
                    account_id, association_name, branch_name, maturity_date, current_allocation, recommended_allocation,
                    projected_yield_bps, constraint_bindings, notes, success,
                    current_yield, optimized_yield, bank_count, term_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row['account_id'],
                    row['association_name'],
                    row['branch_name'],
                    row['maturity_date'],
                    row['current_allocation'],
                    row['recommended_allocation'],
                    row['projected_yield_bps'],
                    str(row['constraint_bindings']),
                    row['notes'],
                    1 if row['success'] else 0,
                    row.get('current_yield'),
                    row.get('optimized_yield'),
                    row.get('bank_count'),
                    row.get('term_count')
                ))
            
            conn.commit()
            logger.info(f"Results saved to database table {table_name}")
            
        except Exception as e:
            logger.error(f"Error saving to database: {str(e)}")
            if conn:
                conn.rollback()
            raise
    
    def generate_markdown_summary(self, summary: Dict[str, Any]) -> str:
        """
        Generate a concise markdown summary report.
        
        Args:
            summary: Summary dictionary from generate_summary_report
            
        Returns:
            Markdown formatted summary string
        """
        try:
            md_content = f"""# Account Maturity Optimization Summary

## Overview
- **Total Accounts Scanned**: {summary.get('total_accounts', 0):,}
- **Successful Optimizations**: {summary.get('successful_optimizations', 0):,}
- **Failed Optimizations**: {summary.get('failed_optimizations', 0):,}
- **Success Rate**: {summary.get('success_rate', 0):.1f}%

## Financial Impact
- **Total Current Allocation**: ${summary.get('total_current_allocation', 0):,.0f}
- **Total Recommended Allocation**: ${summary.get('total_recommended_allocation', 0):,.0f}
- **Average Yield Lift**: {summary.get('avg_yield_lift_bps', 0):.1f} basis points
- **Total Yield Lift**: {summary.get('total_yield_lift_bps', 0):.1f} basis points

## Top 10 Accounts by Yield Lift
"""
            
            top_accounts = summary.get('top_10_accounts', [])
            if top_accounts:
                md_content += "| Association Name | Branch Name | Maturity Date | Current Allocation | Yield Lift (bps) |\n"
                md_content += "|------------------|-------------|---------------|-------------------|------------------|\n"
                
                for account in top_accounts[:10]:
                    md_content += f"| {account['association_name']} | {account['branch_name']} | {account['maturity_date']} | ${account['current_allocation']:,.0f} | {account['projected_yield_bps']:.1f} |\n"
            else:
                md_content += "*No successful optimizations to display*\n"
            
            md_content += "\n## Maturity Distribution\n"
            maturity_dist = summary.get('maturity_distribution', {})
            if maturity_dist:
                for month, count in maturity_dist.items():
                    md_content += f"- **{month}**: {count} accounts\n"
            
            md_content += "\n## Failed Optimizations\n"
            failed_accounts = summary.get('failed_accounts', [])
            if failed_accounts:
                for account in failed_accounts:
                    md_content += f"- **{account['association_name']} ({account['branch_name']})**: {account['notes']}\n"
            else:
                md_content += "*No failed optimizations*\n"
            
            md_content += f"\n---\n*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
            
            return md_content
            
        except Exception as e:
            logger.error(f"Error generating markdown summary: {str(e)}")
            return f"Error generating summary: {str(e)}"


def main():
    """
    Main function to run the maturity filtering system.
    """
    try:
        # Initialize the maturity filter
        maturity_filter = MaturityFilter()
        
        # Run filtering and batch optimization with default parameters
        results = maturity_filter.run_maturity_filtering(
            months_ahead=3,
            min_balance=100000,
            max_balance=10000000,
            allowed_investment_types=['Certificate of Deposit'],
            excluded_banks=[]
        )
        
        if results['success']:
            # Generate markdown summary
            markdown_summary = maturity_filter.generate_markdown_summary(results['summary'])
            
            # Print summary to console
            print("\n" + "="*80)
            print("MATURITY FILTERING COMPLETED SUCCESSFULLY")
            print("="*80)
            print(markdown_summary)
            
            # Save markdown summary to file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Path(__file__).parent.parent / "data" / "maturity_optimization"
            output_dir.mkdir(exist_ok=True)
            
            md_filename = output_dir / f"maturity_optimization_summary_{timestamp}.md"
            with open(md_filename, 'w') as f:
                f.write(markdown_summary)
            
            print(f"\nDetailed results saved to: {output_dir}")
            print(f"Markdown summary saved to: {md_filename}")
            
        else:
            print(f"Optimization failed: {results['message']}")
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
