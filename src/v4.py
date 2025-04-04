"""
import pandas as pd
from ortools.linear_solver import pywraplp

# Function to load and process data from an Excel file
# Extracts bank rankings, bank rates, and investment constraints

def load_data(file_path):
    xls = pd.ExcelFile(file_path)
    bank_ranking_df = pd.read_excel(xls, sheet_name='Bank Ranking', skiprows=0, usecols=[2, 3, 4])
    bank_ranking_df.columns = ["Bank Name", "Bank Code", "Priority"]
    bank_ranking_df.dropna(inplace=True)
    bank_ranking_df["Priority"] = pd.to_numeric(bank_ranking_df["Priority"], errors="coerce")

    raw_bank_rates_df = pd.read_excel(xls, sheet_name='Bank Rates', skiprows=0)
    bank_rates_df = raw_bank_rates_df.loc[:, ~raw_bank_rates_df.columns.str.contains('Unnamed', na=False)].copy()
    bank_rates_df.columns = ["Bank Name", "Bank Code", "CD Term", "CD Rate", "CDARS Term", "CDARS Rate", "Special"]
    bank_rates_df = bank_rates_df.dropna(subset=["Bank Name", "CD Term", "CD Rate"], how="any").copy()
    bank_rates_df.loc[:, "CD Rate"] = pd.to_numeric(bank_rates_df["CD Rate"], errors="coerce")
    bank_rates_df.loc[:, "CD Term"] = bank_rates_df["CD Term"].astype(str).str.strip()

    constraints_df = pd.read_excel(xls, sheet_name='Filter', skiprows=0, usecols=[3, 6, 7])
    constraints_df.columns = ["Filter Name", "Min Value", "Max Value"]
    constraints_df.dropna(subset=["Filter Name"], inplace=True)
    constraints_df.set_index("Filter Name", inplace=True)
    constraints_df.dropna(subset=["Min Value", "Max Value"], how='all', inplace=True)

    return bank_ranking_df, bank_rates_df, constraints_df

# Optimization Function

def optimize_fund_allocation(bank_ranking_df, bank_rates_df, constraints_df, total_funds, investment_duration, cdars_interest, single_bank=None, multi_terms=False):
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        raise Exception("SCIP solver is not available.")

    max_bank_allocation = constraints_df.loc["Bank", "Max Value"] if "Bank" in constraints_df.index else 250000
    bank_rates_filtered = bank_rates_df.copy()
    bank_rates_filtered["CD Term Num"] = bank_rates_filtered["CD Term"].str.extract(r'(\d+)').astype(float)

    if not multi_terms:
        if investment_duration == "short":
            bank_rates_filtered = bank_rates_filtered[(bank_rates_filtered["CD Term Num"] >= 1) & (bank_rates_filtered["CD Term Num"] <= 3)]
        elif investment_duration == "mid":
            bank_rates_filtered = bank_rates_filtered[(bank_rates_filtered["CD Term Num"] >= 4) & (bank_rates_filtered["CD Term Num"] <= 6)]
        elif investment_duration == "long":
            bank_rates_filtered = bank_rates_filtered[(bank_rates_filtered["CD Term Num"] >= 7) & (bank_rates_filtered["CD Term Num"] <= 12)]
    
    bank_rates_filtered.drop(columns=["CD Term Num"], inplace=True)

    if single_bank:
        bank_rates_filtered = bank_rates_filtered[bank_rates_filtered["Bank Name"] == single_bank]

    if not cdars_interest:
        bank_rates_filtered.drop(columns=["CDARS Term", "CDARS Rate"], errors='ignore', inplace=True)
        bank_rates_filtered.dropna(subset=["CD Rate"], inplace=True)

    banks = bank_ranking_df["Bank Name"].tolist() if not single_bank else [single_bank]
    allocation = {}

    for bank in banks:
        subset = bank_rates_filtered[bank_rates_filtered["Bank Name"] == bank]
        for _, row in subset.iterrows():
            term = row["CD Term"]
            allocation[(bank, term)] = solver.IntVar(0, solver.infinity(), f'alloc_{bank}_{term}')

    solver.Add(sum(allocation.values()) <= total_funds)

    if multi_terms:
        split_amount = total_funds // 3
        for bank in banks:
            short_terms = bank_rates_filtered[(bank_rates_filtered["CD Term"].str.contains("1 month|2 months|3 months")) & (bank_rates_filtered["Bank Name"] == bank)]
            mid_terms = bank_rates_filtered[(bank_rates_filtered["CD Term"].str.contains("4 months|5 months|6 months")) & (bank_rates_filtered["Bank Name"] == bank)]
            long_terms = bank_rates_filtered[(bank_rates_filtered["CD Term"].str.contains("7 months|8 months|9 months|10 months|11 months|12 months")) & (bank_rates_filtered["Bank Name"] == bank)]
            
            solver.Add(sum(allocation[(bank, term)] for term in short_terms["CD Term"]) <= split_amount)
            solver.Add(sum(allocation[(bank, term)] for term in mid_terms["CD Term"]) <= split_amount)
            solver.Add(sum(allocation[(bank, term)] for term in long_terms["CD Term"]) <= split_amount)
    
    objective = solver.Objective()
    for (bank, term), var in allocation.items():
        rate = bank_rates_filtered[(bank_rates_filtered["Bank Name"] == bank) & (bank_rates_filtered["CD Term"] == term)]["CD Rate"].values
        if len(rate) > 0:
            objective.SetCoefficient(var, rate[0] / 100)

    objective.SetMaximization()
    status = solver.Solve()

    if status != pywraplp.Solver.OPTIMAL:
        print("No optimal solution found.")
        return pd.DataFrame()

    results = [(bank, term, int(var.solution_value())) for (bank, term), var in allocation.items() if var.solution_value() > 0]
    df = pd.DataFrame(results, columns=["Bank Name", "CD Term", "Allocated Amount"])
    df = df.merge(bank_rates_filtered[["Bank Name", "CD Term", "CD Rate"]], on=["Bank Name", "CD Term"], how="left")
    df["Expected Return"] = (df["Allocated Amount"] * df["CD Rate"]) / 100
    return df

if __name__ == "__main__":
    file_path = "Associa Data Tool.xlsx"
    bank_ranking_df, bank_rates_df, constraints_df = load_data(file_path)
    total_funds = float(input("Enter the total funds available for allocation: "))
    single_bank = input("Would you like to keep it all at one bank? (yes/no): ").strip().lower() == "yes"
    if single_bank:
        print("Available Banks:", bank_ranking_df["Bank Name"].tolist())
        selected_bank = input("Select the Bank: ").strip()
    else:
        selected_bank = None

    multi_terms = input("Would you like to invest across multiple time frames? (yes/no): ").strip().lower() == "yes"
    investment_duration = input("Enter investment duration (short/mid/long): ").strip().lower()
    cdars_interest = input("Are you interested in CDARS products? (yes/no): ").strip().lower() == "yes"

    result = optimize_fund_allocation(bank_ranking_df, bank_rates_df, constraints_df, total_funds, investment_duration, cdars_interest, selected_bank, multi_terms)

    if not result.empty:
        print(result)
        result.to_excel("Optimized_Fund_Allocation.xlsx", index=False)
"""

# ------------- OPTIMIZED VERSION BELOW -------------


import pandas as pd
from ortools.linear_solver import pywraplp
from typing import Tuple, Optional, Dict, List, Any
import logging
from pathlib import Path

# Constants
SHEET_BANK_RANKING = 'Bank Ranking'
SHEET_BANK_RATES = 'Bank Rates'
SHEET_FILTER = 'Filter'
DEFAULT_MAX_BANK_ALLOCATION = 250000

# Term duration mappings (in months)
SHORT_TERM_MIN, SHORT_TERM_MAX = 1, 3
MID_TERM_MIN, MID_TERM_MAX = 4, 6
LONG_TERM_MIN, LONG_TERM_MAX = 7, 12

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and process data from the Excel file containing bank information.
    
    Args:
        file_path: Path to the Excel file with bank data
        
    Returns:
        Tuple containing three DataFrames:
        - bank_ranking_df: DataFrame with bank rankings
        - bank_rates_df: DataFrame with CD rates for different terms
        - constraints_df: DataFrame with optimization constraints
        
    Raises:
        FileNotFoundError: If the specified file doesn't exist
        ValueError: If required sheets or columns are missing
    """
    try:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Excel file not found: {file_path}")
            
        xls = pd.ExcelFile(file_path)
        
        # Validate required sheets exist
        required_sheets = [SHEET_BANK_RANKING, SHEET_BANK_RATES, SHEET_FILTER]
        missing_sheets = [sheet for sheet in required_sheets if sheet not in xls.sheet_names]
        if missing_sheets:
            raise ValueError(f"Missing required sheets: {', '.join(missing_sheets)}")
        
        # Load bank ranking data
        bank_ranking_df = pd.read_excel(xls, sheet_name=SHEET_BANK_RANKING, skiprows=0, usecols=[2, 3, 4])
        bank_ranking_df.columns = ["Bank Name", "Bank Code", "Priority"]
        bank_ranking_df.dropna(inplace=True)
        bank_ranking_df["Priority"] = pd.to_numeric(bank_ranking_df["Priority"], errors="coerce")

        # Load bank rates data
        raw_bank_rates_df = pd.read_excel(xls, sheet_name=SHEET_BANK_RATES, skiprows=0)
        bank_rates_df = raw_bank_rates_df.loc[:, ~raw_bank_rates_df.columns.str.contains('Unnamed', na=False)].copy()
        bank_rates_df.columns = ["Bank Name", "Bank Code", "CD Term", "CD Rate", "CDARS Term", "CDARS Rate", "Special"]
        bank_rates_df = bank_rates_df.dropna(subset=["Bank Name", "CD Term", "CD Rate"], how="any").copy()
        bank_rates_df.loc[:, "CD Rate"] = pd.to_numeric(bank_rates_df["CD Rate"], errors="coerce")
        bank_rates_df.loc[:, "CD Term"] = bank_rates_df["CD Term"].astype(str).str.strip()

        # Extract numeric term length for filtering
        bank_rates_df["CD Term Num"] = bank_rates_df["CD Term"].str.extract(r'(\d+)').astype(float)

        # Load constraints data
        constraints_df = pd.read_excel(xls, sheet_name=SHEET_FILTER, skiprows=0, usecols=[3, 6, 7])
        constraints_df.columns = ["Filter Name", "Min Value", "Max Value"]
        constraints_df.dropna(subset=["Filter Name"], inplace=True)
        constraints_df.set_index("Filter Name", inplace=True)
        constraints_df.dropna(subset=["Min Value", "Max Value"], how='all', inplace=True)

        return bank_ranking_df, bank_rates_df, constraints_df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def filter_by_term_duration(bank_rates_df: pd.DataFrame, investment_duration: str, multi_terms: bool) -> pd.DataFrame:
    """
    Filter bank rates by the specified investment duration.
    
    Args:
        bank_rates_df: DataFrame containing bank rate information
        investment_duration: 'short', 'mid', or 'long'
        multi_terms: If True, don't filter by duration
        
    Returns:
        Filtered DataFrame
    """
    if multi_terms:
        return bank_rates_df.copy()
        
    filtered_df = bank_rates_df.copy()
    
    if investment_duration == "short":
        return filtered_df[(filtered_df["CD Term Num"] >= SHORT_TERM_MIN) & 
                           (filtered_df["CD Term Num"] <= SHORT_TERM_MAX)]
    elif investment_duration == "mid":
        return filtered_df[(filtered_df["CD Term Num"] >= MID_TERM_MIN) & 
                           (filtered_df["CD Term Num"] <= MID_TERM_MAX)]
    elif investment_duration == "long":
        return filtered_df[(filtered_df["CD Term Num"] >= LONG_TERM_MIN) & 
                           (filtered_df["CD Term Num"] <= LONG_TERM_MAX)]
    else:
        logger.warning(f"Unknown investment duration: {investment_duration}. Using all terms.")
        return filtered_df


def optimize_fund_allocation(
    bank_ranking_df: pd.DataFrame, 
    bank_rates_df: pd.DataFrame, 
    constraints_df: pd.DataFrame, 
    total_funds: float, 
    investment_duration: str, 
    cdars_interest: bool, 
    single_bank: Optional[str] = None, 
    multi_terms: bool = False,
    time_limit_seconds: int = 30
) -> pd.DataFrame:
    """
    Optimize fund allocation across banks and CD terms to maximize interest.
    
    Args:
        bank_ranking_df: DataFrame with bank ranking information
        bank_rates_df: DataFrame with CD rates information
        constraints_df: DataFrame with optimization constraints
        total_funds: Total funds available for allocation
        investment_duration: 'short', 'mid', or 'long'
        cdars_interest: Whether to consider CDARS products
        single_bank: Restrict optimization to a single bank if specified
        multi_terms: Whether to invest across multiple term lengths
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
        
        # Get max bank allocation from constraints
        max_bank_allocation = constraints_df.loc["Bank", "Max Value"] if "Bank" in constraints_df.index else DEFAULT_MAX_BANK_ALLOCATION
        
        # Filter bank rates by term duration
        bank_rates_filtered = filter_by_term_duration(bank_rates_df, investment_duration, multi_terms)
        
        # Apply single bank filter if specified
        if single_bank:
            bank_rates_filtered = bank_rates_filtered[bank_rates_filtered["Bank Name"] == single_bank]
            if bank_rates_filtered.empty:
                raise ValueError(f"No rates found for bank: {single_bank}")
        
        # Remove CDARS info if not interested
        if not cdars_interest:
            bank_rates_filtered.drop(columns=["CDARS Term", "CDARS Rate"], errors='ignore', inplace=True)
            bank_rates_filtered.dropna(subset=["CD Rate"], inplace=True)
        
        # Get available banks
        banks = bank_ranking_df["Bank Name"].tolist() if not single_bank else [single_bank]
        banks = [bank for bank in banks if bank in bank_rates_filtered["Bank Name"].unique()]
        
        if not banks:
            raise ValueError("No banks available with the specified criteria")
            
        # Create variables for allocation
        allocation = {}
        for bank in banks:
            subset = bank_rates_filtered[bank_rates_filtered["Bank Name"] == bank]
            for _, row in subset.iterrows():
                term = row["CD Term"]
                allocation[(bank, term)] = solver.IntVar(0, solver.infinity(), f'alloc_{bank}_{term}')
        
        # Add total funds constraint
        solver.Add(sum(allocation.values()) <= total_funds)
        
        # Add constraints for multi-term allocation
        if multi_terms:
            split_amount = total_funds // 3
            term_categories = {
                "short": bank_rates_filtered[(bank_rates_filtered["CD Term Num"] >= SHORT_TERM_MIN) & 
                                            (bank_rates_filtered["CD Term Num"] <= SHORT_TERM_MAX)],
                "mid": bank_rates_filtered[(bank_rates_filtered["CD Term Num"] >= MID_TERM_MIN) & 
                                          (bank_rates_filtered["CD Term Num"] <= MID_TERM_MAX)],
                "long": bank_rates_filtered[(bank_rates_filtered["CD Term Num"] >= LONG_TERM_MIN) & 
                                           (bank_rates_filtered["CD Term Num"] <= LONG_TERM_MAX)]
            }
            
            for bank in banks:
                for category, df in term_categories.items():
                    terms = df[df["Bank Name"] == bank]["CD Term"].tolist()
                    if terms:
                        vars_to_sum = [allocation.get((bank, term), 0) for term in terms 
                                      if (bank, term) in allocation]
                        if vars_to_sum:
                            solver.Add(sum(vars_to_sum) <= split_amount)
        
        # Add bank maximum allocation constraint
        for bank in banks:
            vars_to_sum = [var for (b, _), var in allocation.items() if b == bank]
            if vars_to_sum:
                solver.Add(sum(vars_to_sum) <= max_bank_allocation)
        
        # Set objective function (maximize interest)
        objective = solver.Objective()
        for (bank, term), var in allocation.items():
            rate = bank_rates_filtered[(bank_rates_filtered["Bank Name"] == bank) & 
                                     (bank_rates_filtered["CD Term"] == term)]["CD Rate"].values
            if len(rate) > 0:
                objective.SetCoefficient(var, rate[0] / 100)
        
        objective.SetMaximization()
        
        # Solve the optimization problem
        status = solver.Solve()
        
        if status != pywraplp.Solver.OPTIMAL:
            logger.warning(f"No optimal solution found. Solver status: {status}")
            return pd.DataFrame()
        
        # Collect results
        results = [(bank, term, int(var.solution_value())) 
                  for (bank, term), var in allocation.items() 
                  if var.solution_value() > 0]
                  
        if not results:
            logger.warning("Optimization completed but no funds were allocated")
            return pd.DataFrame()
            
        # Create results DataFrame
        df = pd.DataFrame(results, columns=["Bank Name", "CD Term", "Allocated Amount"])
        df = df.merge(bank_rates_filtered[["Bank Name", "CD Term", "CD Rate"]], 
                     on=["Bank Name", "CD Term"], how="left")
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


def get_user_inputs() -> Dict[str, Any]:
    """Get and validate user inputs for the optimization."""
    try:
        inputs = {}
        
        # Get total funds
        while True:
            try:
                inputs["total_funds"] = float(input("Enter the total funds available for allocation: "))
                if inputs["total_funds"] <= 0:
                    print("Total funds must be greater than zero.")
                    continue
                break
            except ValueError:
                print("Please enter a valid number.")
        
        # Single bank preference
        inputs["single_bank_preference"] = input("Would you like to keep it all at one bank? (yes/no): ").strip().lower() == "yes"
        
        # Selected bank if single bank chosen
        inputs["selected_bank"] = None
        if inputs["single_bank_preference"]:
            print("Available Banks:", ", ".join(bank_ranking_df["Bank Name"].tolist()))
            inputs["selected_bank"] = input("Select the Bank: ").strip()
            
            # Validate bank exists
            if inputs["selected_bank"] not in bank_ranking_df["Bank Name"].tolist():
                available_banks = ", ".join(bank_ranking_df["Bank Name"].tolist())
                print(f"Warning: '{inputs['selected_bank']}' not found in bank list. Available banks: {available_banks}")
                if input("Continue anyway? (yes/no): ").strip().lower() != "yes":
                    return get_user_inputs()
        
        # Multi-term preference
        inputs["multi_terms"] = input("Would you like to invest across multiple time frames? (yes/no): ").strip().lower() == "yes"
        
        # Investment duration
        while True:
            inputs["investment_duration"] = input("Enter investment duration (short/mid/long): ").strip().lower()
            if inputs["investment_duration"] not in ["short", "mid", "long"]:
                print("Please enter 'short', 'mid', or 'long'.")
                continue
            break
        
        # CDARS interest
        inputs["cdars_interest"] = input("Are you interested in CDARS products? (yes/no): ").strip().lower() == "yes"
        
        return inputs
        
    except KeyboardInterrupt:
        print("\nInput cancelled. Exiting.")
        exit(0)


def main():
    """Main function to run the fund allocation optimization."""
    try:
        print("Fund Allocation Optimization Tool")
        print("=================================")
        
        file_path = input("Enter the path to the Excel file (default: 'Associa Data Tool.xlsx'): ").strip()
        if not file_path:
            file_path = "Associa Data Tool.xlsx"
            
        print(f"Loading data from {file_path}...")
        global bank_ranking_df, bank_rates_df, constraints_df
        bank_ranking_df, bank_rates_df, constraints_df = load_data(file_path)
        
        print(f"Found {len(bank_ranking_df)} banks and {len(bank_rates_df)} rate options.")
        
        # Get user inputs
        inputs = get_user_inputs()
        
        print("\nOptimizing fund allocation...")
        result = optimize_fund_allocation(
            bank_ranking_df, 
            bank_rates_df, 
            constraints_df, 
            inputs["total_funds"], 
            inputs["investment_duration"], 
            inputs["cdars_interest"], 
            inputs["selected_bank"], 
            inputs["multi_terms"]
        )
        
        if not result.empty:
            print("\nOptimized Allocation Results:")
            print("=============================")
            print(result.to_string(index=False))
            
            output_file = "Optimized_Fund_Allocation.xlsx"
            result.to_excel(output_file, index=False)
            print(f"\nResults saved to {output_file}")
        else:
            print("\nNo allocation solution found. Try adjusting your criteria.")
            
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
    except ValueError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        logger.exception("Unexpected error in main function")


#Uncomment the following to run the optimized version
if __name__ == "__main__":
     main()

