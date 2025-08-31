"""
Account Maturity Optimization System

This module implements a comprehensive system to:
1. Scan for accounts approaching maturity in a 3-month window
2. Apply business rule filters to identify target candidates
3. Run OPTool optimization for each candidate with current rates and constraints
4. Generate results table and summary report
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

class MaturityOptimizer:
    """
    Main class for account maturity optimization system.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the maturity optimizer.
        
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
            # Filter out accounts with very short terms (less than 1 month)
            filtered_df = filtered_df[filtered_df['days_to_maturity'] >= 30]
            
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
    
    def optimize_single_account(self, account: pd.Series, 
                              rates_data: Dict, 
                              constraints: Dict) -> Dict[str, Any]:
        """
        Run OPTool optimization for a single account.
        
        Args:
            account: Account data as pandas Series
            rates_data: Current rates and bank relationship data
            constraints: Current optimization constraints
            
        Returns:
            Dictionary with optimization results
        """
        try:
            # Get an existing branch name from the database for optimization
            conn = self.get_db_connection()
            branch_query = "SELECT DISTINCT branch_name FROM branch_relationships LIMIT 1"
            branch_result = pd.read_sql_query(branch_query, conn)
            
            if branch_result.empty:
                # Fallback to a default branch name
                branch_name = "Alliance Association Management, Inc. dba Associa Hill Country"
            else:
                branch_name = branch_result.iloc[0]['branch_name']
            
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
                    'maturity_date': account['maturity_date'],
                    'current_allocation': account['current_balance'],
                    'recommended_allocation': None,
                    'projected_yield_bps': None,
                    'constraint_bindings': [],
                    'notes': f"Optimization failed: {results['message']}",
                    'success': False
                }
            
            # Extract key results
            current_yield = account['investment_rate'] * 100 if account['investment_rate'] else 0
            optimized_yield = results['summary']['weighted_avg_rate']
            yield_lift = optimized_yield - current_yield
            
            # Identify binding constraints
            constraint_bindings = []
            if results.get('constraint_violations'):
                constraint_bindings = results['constraint_violations']
            
            return {
                'account_id': account['association_id'],
                'maturity_date': account['maturity_date'],
                'current_allocation': account['current_balance'],
                'recommended_allocation': results['summary']['total_allocated'],
                'projected_yield_bps': yield_lift * 100,  # Convert to basis points
                'constraint_bindings': constraint_bindings,
                'notes': 'Optimization successful',
                'success': True,
                'current_yield': current_yield,
                'optimized_yield': optimized_yield,
                'bank_count': results.get('bank_count', 0),
                'term_count': results.get('term_count', 0)
            }
            
        except Exception as e:
            logger.error(f"Error optimizing account {account['association_id']}: {str(e)}")
            return {
                'account_id': account['association_id'],
                'maturity_date': account['maturity_date'],
                'current_allocation': account['current_balance'],
                'recommended_allocation': None,
                'projected_yield_bps': None,
                'constraint_bindings': [],
                'notes': f"Error during optimization: {str(e)}",
                'success': False
            }
    
    def run_maturity_optimization(self, 
                                 months_ahead: int = 3,
                                 min_balance: float = 100000,
                                 max_balance: float = 10000000,
                                 allowed_investment_types: Optional[List[str]] = None,
                                 excluded_banks: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run the complete maturity optimization pipeline.
        
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
            logger.info("Starting maturity optimization pipeline")
            
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
                
                result = self.optimize_single_account(account, rates_data, constraints)
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
                
                # Top 10 accounts by yield lift
                top_accounts = successful_results.nlargest(10, 'projected_yield_bps')[
                    ['account_id', 'maturity_date', 'current_allocation', 'projected_yield_bps']
                ]
                
                # Maturity distribution
                maturity_distribution = target_candidates['maturity_month'].value_counts().sort_index()
                
            else:
                total_current_allocation = 0
                total_recommended_allocation = 0
                avg_yield_lift = 0
                total_yield_lift_bps = 0
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
                'failed_accounts': results_df[results_df['success'] == False][['account_id', 'notes']].to_dict('records') if not results_df.empty else []
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
                    account_id, maturity_date, current_allocation, recommended_allocation,
                    projected_yield_bps, constraint_bindings, notes, success,
                    current_yield, optimized_yield, bank_count, term_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row['account_id'],
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
                md_content += "| Account ID | Maturity Date | Current Allocation | Yield Lift (bps) |\n"
                md_content += "|------------|---------------|-------------------|------------------|\n"
                
                for account in top_accounts[:10]:
                    md_content += f"| {account['account_id']} | {account['maturity_date']} | ${account['current_allocation']:,.0f} | {account['projected_yield_bps']:.1f} |\n"
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
                    md_content += f"- **Account {account['account_id']}**: {account['notes']}\n"
            else:
                md_content += "*No failed optimizations*\n"
            
            md_content += f"\n---\n*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
            
            return md_content
            
        except Exception as e:
            logger.error(f"Error generating markdown summary: {str(e)}")
            return f"Error generating summary: {str(e)}"


def main():
    """
    Main function to run the maturity optimization system.
    """
    try:
        # Initialize the optimizer
        maturity_opt = MaturityOptimizer()
        
        # Run optimization with default parameters
        results = maturity_opt.run_maturity_optimization(
            months_ahead=3,
            min_balance=100000,
            max_balance=10000000,
            allowed_investment_types=['Certificate of Deposit'],
            excluded_banks=[]
        )
        
        if results['success']:
            # Generate markdown summary
            markdown_summary = maturity_opt.generate_markdown_summary(results['summary'])
            
            # Print summary to console
            print("\n" + "="*80)
            print("MATURITY OPTIMIZATION COMPLETED SUCCESSFULLY")
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
