# Account Maturity Optimization System

## Overview

The Account Maturity Optimization System is a comprehensive solution that identifies accounts approaching maturity, applies business rule filters, and runs each candidate through OPTool to produce the best possible allocation given current rates and constraints.

## Features

- **Automated Account Scanning**: Identifies accounts maturing in a configurable time window (default: 3 months)
- **Business Rule Filtering**: Applies configurable filters for balance thresholds, investment types, and bank exclusions
- **OPTool Integration**: Runs optimization for each candidate account using current rates and constraints
- **Comprehensive Reporting**: Generates detailed results with yield improvements and constraint analysis
- **Multiple Output Formats**: Saves results to CSV, database, and generates markdown summaries
- **Configurable Business Rules**: YAML-based configuration for easy customization

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Account Scan  │───▶│ Business Filters │───▶│ OPTool Engine  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Maturity Date │    │ Balance/Type     │    │ Optimization    │
│  Range Query   │    │ Filtering        │    │ Results         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- SQLite database with account data
- Required Python packages (see requirements.txt)

### Setup

1. **Clone or download the project files**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Ensure database is initialized**:
   ```bash
   python initialize_db.py
   ```

## Usage

### Command Line Interface

The system provides a comprehensive command-line interface for running optimizations:

#### Basic Usage

```bash
# Run with default settings (3 months ahead, $100K-$10M balance range)
python run_maturity_optimization.py

# Run with custom parameters
python run_maturity_optimization.py --months 6 --min-balance 500000 --max-balance 5000000

# Run for specific investment types only
python run_maturity_optimization.py --investment-types "Certificate of Deposit" "Money Market"

# Exclude specific banks
python run_maturity_optimization.py --exclude-banks "Bank A" "Bank B"
```

#### Advanced Options

```bash
# Dry run (scan accounts without optimization)
python run_maturity_optimization.py --dry-run

# Custom database path
python run_maturity_optimization.py --db-path /path/to/custom.db

# Verbose logging
python run_maturity_optimization.py --verbose

# Custom output format
python run_maturity_optimization.py --output-format csv
```

#### Command Line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--months` | `-m` | 3 | Months ahead to scan |
| `--min-balance` | `-min` | 100000 | Minimum balance threshold |
| `--max-balance` | `-max` | 10000000 | Maximum balance threshold |
| `--investment-types` | `-t` | ["Certificate of Deposit"] | Allowed investment types |
| `--exclude-banks` | `-e` | [] | Banks to exclude |
| `--db-path` | `-d` | auto | Custom database path |
| `--verbose` | `-v` | False | Enable verbose logging |
| `--dry-run` | `-n` | False | Scan without optimization |
| `--output-format` | `-f` | both | Output format (csv/database/both) |

### Programmatic Usage

You can also use the system programmatically:

```python
from src.maturity_optimizer import MaturityOptimizer

# Initialize optimizer
maturity_opt = MaturityOptimizer()

# Run optimization
results = maturity_opt.run_maturity_optimization(
    months_ahead=3,
    min_balance=100000,
    max_balance=10000000,
    allowed_investment_types=['Certificate of Deposit'],
    excluded_banks=[]
)

# Access results
if results['success']:
    summary = results['summary']
    print(f"Success rate: {summary['success_rate']:.1f}%")
    print(f"Total yield lift: {summary['total_yield_lift_bps']:.1f} bps")
```

## Configuration

### YAML Configuration File

The system uses `maturity_optimization_config.yaml` for business rule configuration:

```yaml
# Time Window Configuration
time_window:
  months_ahead: 3
  min_days_to_maturity: 30

# Balance Thresholds
balance_filters:
  min_balance: 100000
  max_balance: 10000000

# Investment Type Filters
investment_types:
  allowed_types:
    - "Certificate of Deposit"
    - "Money Market"
```

### Key Configuration Sections

- **Time Window**: Configure scanning horizon and maturity thresholds
- **Balance Filters**: Set minimum and maximum balance requirements
- **Investment Types**: Define allowed and excluded investment types
- **Bank Filters**: Configure bank relationships and exclusions
- **Business Rules**: Set account segmentation and maturity laddering
- **Output Configuration**: Customize report formats and content

## Output Files

### Results CSV

Contains detailed optimization results for each account:

| Column | Description |
|--------|-------------|
| `account_id` | Association ID |
| `maturity_date` | Account maturity date |
| `current_allocation` | Current balance |
| `recommended_allocation` | Optimized allocation amount |
| `projected_yield_bps` | Yield improvement in basis points |
| `constraint_bindings` | Binding constraints |
| `notes` | Optimization status/notes |
| `success` | Optimization success flag |

### Markdown Summary

Comprehensive summary report including:

- **Overview**: Total accounts, success rates
- **Financial Impact**: Yield improvements, allocation changes
- **Top Accounts**: Top 10 accounts by yield lift
- **Maturity Distribution**: Account distribution by maturity month
- **Failed Optimizations**: Details on failed optimizations

### Database Tables

Results are saved to timestamped tables:

```sql
CREATE TABLE accounts_optimizations_YYYYMMDD (
    id INTEGER PRIMARY KEY,
    account_id INTEGER,
    maturity_date TEXT,
    current_allocation REAL,
    recommended_allocation REAL,
    projected_yield_bps REAL,
    constraint_bindings TEXT,
    notes TEXT,
    success INTEGER,
    created_at TIMESTAMP
);
```

## Business Logic

### Account Filtering

1. **Maturity Scan**: Identifies accounts maturing within specified window
2. **Balance Filtering**: Applies minimum/maximum balance thresholds
3. **Type Filtering**: Restricts to allowed investment types
4. **Bank Filtering**: Excludes specified banks
5. **Data Quality**: Filters out accounts with missing critical data

### Optimization Process

1. **Rate Retrieval**: Gets current CD rates, ECR rates, and constraints
2. **Account Processing**: Runs OPTool for each candidate account
3. **Result Analysis**: Calculates yield improvements and constraint bindings
4. **Summary Generation**: Creates comprehensive reports and statistics

### Constraint Handling

The system respects existing OPTool constraints:

- **Product Constraints**: CD, Checking, Savings preferences
- **Time Constraints**: Short, Mid, Long term preferences
- **Weighting Factors**: Interest rates vs. ECR return balance
- **Bank Constraints**: Partner bank preferences
- **Liquidity Requirements**: Reserve percentage constraints

## Performance Considerations

### Batch Processing

- Processes accounts in configurable batches (default: 50)
- Configurable parallel processing (default: 4 workers)
- Timeout settings for individual optimizations and total runtime

### Memory Management

- Configurable memory usage limits
- Efficient data processing with pandas
- Database connection pooling

### Logging and Monitoring

- Comprehensive logging at multiple levels
- Performance metrics tracking
- Error handling and recovery

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   - Verify database path and permissions
   - Check if database is initialized

2. **Optimization Failures**
   - Review constraint settings
   - Check rate data availability
   - Verify account data quality

3. **Performance Issues**
   - Reduce batch size
   - Adjust timeout settings
   - Check system resources

### Debug Mode

Enable verbose logging for detailed troubleshooting:

```bash
python run_maturity_optimization.py --verbose
```

### Log Files

Check `maturity_optimization.log` for detailed execution logs.

## Examples

### Example 1: Standard 3-Month Optimization

```bash
python run_maturity_optimization.py
```

**Output**: Scans next 3 months, optimizes accounts $100K-$10M, saves results to CSV and database.

### Example 2: Extended 6-Month Scan

```bash
python run_maturity_optimization.py --months 6 --min-balance 250000
```

**Output**: Scans next 6 months, focuses on accounts $250K+, provides extended planning horizon.

### Example 3: Investment Type Specific

```bash
python run_maturity_optimization.py --investment-types "Certificate of Deposit" "Money Market"
```

**Output**: Optimizes only CD and Money Market accounts, excludes other investment types.

### Example 4: Dry Run Testing

```bash
python run_maturity_optimization.py --dry-run --months 3
```

**Output**: Shows account scan results without running optimization, useful for testing filters.

## Integration

### With Existing Systems

- **Database Integration**: Works with existing SQLite databases
- **OPTool Integration**: Leverages existing optimization engine
- **API Integration**: Can be called from other applications

### Scheduling

The system can be scheduled to run automatically:

```bash
# Cron job example (daily at 9 AM)
0 9 * * * cd /path/to/project && python run_maturity_optimization.py

# Windows Task Scheduler
# Create scheduled task to run run_maturity_optimization.py
```

## Support and Maintenance

### Regular Maintenance

- **Database Updates**: Ensure rate data is current
- **Constraint Reviews**: Periodically review business rules
- **Performance Monitoring**: Track optimization success rates

### Updates and Enhancements

- **Rate Data**: Update CD and ECR rates regularly
- **Business Rules**: Adjust filters based on changing requirements
- **System Upgrades**: Monitor for OPTool and dependency updates

## License and Attribution

This system is part of the Fund Allocation Optimization Tool (OPTool) project.

---

For additional support or questions, please refer to the main project documentation or contact the development team.
