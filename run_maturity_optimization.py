#!/usr/bin/env python3
"""
Command-line interface for Account Maturity Optimization System

Usage:
    python run_maturity_optimization.py [options]

Examples:
    # Run with default settings (3 months ahead, $100K-$10M balance range)
    python run_maturity_optimization.py
    
    # Run with custom parameters
    python run_maturity_optimization.py --months 6 --min-balance 500000 --max-balance 5000000
    
    # Run for specific investment types only
    python run_maturity_optimization.py --investment-types "Certificate of Deposit" "Money Market"
    
    # Exclude specific banks
    python run_maturity_optimization.py --exclude-banks "Bank A" "Bank B"
"""

import argparse
import sys
from pathlib import Path
import logging

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from maturity_optimizer import MaturityOptimizer

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('maturity_optimization.log')
        ]
    )

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Account Maturity Optimization System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--months', '-m',
        type=int,
        default=3,
        help='Number of months ahead to scan for maturing accounts (default: 3)'
    )
    
    parser.add_argument(
        '--min-balance', '-min',
        type=float,
        default=100000,
        help='Minimum balance threshold in dollars (default: 100000)'
    )
    
    parser.add_argument(
        '--max-balance', '-max',
        type=float,
        default=10000000,
        help='Maximum balance threshold in dollars (default: 10000000)'
    )
    
    parser.add_argument(
        '--investment-types', '-t',
        nargs='+',
        default=['Certificate of Deposit'],
        help='Allowed investment types (default: "Certificate of Deposit")'
    )
    
    parser.add_argument(
        '--exclude-banks', '-e',
        nargs='+',
        default=[],
        help='Banks to exclude from optimization'
    )
    
    parser.add_argument(
        '--db-path', '-d',
        type=str,
        help='Custom database path (default: uses project data/langston.db)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Scan accounts without running optimization (for testing)'
    )
    
    parser.add_argument(
        '--output-format', '-f',
        choices=['csv', 'database', 'both'],
        default='both',
        help='Output format for results (default: both)'
    )
    
    return parser.parse_args()

def main():
    """Main function."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging
        setup_logging(args.verbose)
        logger = logging.getLogger(__name__)
        
        logger.info("Starting Account Maturity Optimization System")
        logger.info(f"Parameters: months={args.months}, balance_range=${args.min_balance:,.0f}-${args.max_balance:,.0f}")
        logger.info(f"Investment types: {args.investment_types}")
        if args.exclude_banks:
            logger.info(f"Excluded banks: {args.exclude_banks}")
        
        # Initialize optimizer
        maturity_opt = MaturityOptimizer(db_path=args.db_path)
        
        if args.dry_run:
            # Just scan accounts without optimization
            logger.info("DRY RUN MODE: Scanning accounts only")
            
            maturing_accounts = maturity_opt.scan_maturing_accounts(args.months)
            target_candidates = maturity_opt.apply_business_filters(
                maturing_accounts,
                min_balance=args.min_balance,
                max_balance=args.max_balance,
                allowed_investment_types=args.investment_types,
                excluded_banks=args.exclude_banks
            )
            
            print(f"\n{'='*80}")
            print("DRY RUN RESULTS - ACCOUNT SCAN ONLY")
            print(f"{'='*80}")
            print(f"Total accounts maturing in next {args.months} months: {len(maturing_accounts):,}")
            print(f"Target candidates after filtering: {len(target_candidates):,}")
            
            if not target_candidates.empty:
                print(f"\nSample target candidates:")
                sample = target_candidates.head(5)
                for _, account in sample.iterrows():
                    print(f"  - Account {account['association_id']}: ${account['current_balance']:,.0f} "
                          f"matures {account['maturity_date'].strftime('%Y-%m-%d')} "
                          f"at {account['bank_name']}")
            
            return
        
        # Run full optimization
        logger.info("Running full maturity optimization pipeline")
        
        results = maturity_opt.run_maturity_optimization(
            months_ahead=args.months,
            min_balance=args.min_balance,
            max_balance=args.max_balance,
            allowed_investment_types=args.investment_types,
            excluded_banks=args.exclude_banks
        )
        
        if results['success']:
            # Generate and display summary
            markdown_summary = maturity_opt.generate_markdown_summary(results['summary'])
            
            print(f"\n{'='*80}")
            print("MATURITY OPTIMIZATION COMPLETED SUCCESSFULLY")
            print(f"{'='*80}")
            print(markdown_summary)
            
            # Save results based on output format preference
            timestamp = results['timestamp']
            output_dir = Path(__file__).parent / "data" / "maturity_optimization"
            output_dir.mkdir(exist_ok=True)
            
            if args.output_format in ['csv', 'both']:
                csv_filename = output_dir / f"maturity_optimization_results_{timestamp}.csv"
                print(f"\nDetailed results saved to: {csv_filename}")
            
            if args.output_format in ['database', 'both']:
                print(f"Results saved to database table: accounts_optimizations_{timestamp}")
            
            # Save markdown summary
            md_filename = output_dir / f"maturity_optimization_summary_{timestamp}.md"
            with open(md_filename, 'w') as f:
                f.write(markdown_summary)
            print(f"Markdown summary saved to: {md_filename}")
            
            # Print key metrics
            summary = results['summary']
            print(f"\n{'='*50}")
            print("KEY METRICS")
            print(f"{'='*50}")
            print(f"Success Rate: {summary['success_rate']:.1f}%")
            print(f"Total Yield Lift: {summary['total_yield_lift_bps']:.1f} basis points")
            print(f"Average Yield Lift: {summary['avg_yield_lift_bps']:.1f} basis points")
            print(f"Total Allocation: ${summary['total_current_allocation']:,.0f}")
            
        else:
            logger.error(f"Optimization failed: {results['message']}")
            print(f"Optimization failed: {results['message']}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.exception("Unexpected error in main execution")
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
