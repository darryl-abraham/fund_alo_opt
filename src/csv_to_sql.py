import pandas as pd
import sqlite3
import re

# Function to clean text data by removing quotes and newlines
def clean_text(text):
    if isinstance(text, str):
        # Remove quotes, newlines, and carriage returns
        return re.sub(r'["\n\r]', '', text)
    return text

# Connect to SQLite database (or create it)
print("Connecting to SQLite database...")
conn = sqlite3.connect("./data/langston.db")
cursor = conn.cursor()

# Drop existing tables
drop_all = True
if drop_all:
    cursor.execute("DROP TABLE IF EXISTS test_data")
    cursor.execute("DROP TABLE IF EXISTS cd_rates")
    cursor.execute("DROP TABLE IF EXISTS branch_relationships")
    cursor.execute("DROP TABLE IF EXISTS ecr_rates")

# Create test_data table
print("Creating test_data table...")
cursor.execute("""
CREATE TABLE IF NOT EXISTS test_data (
    association_id INTEGER PRIMARY KEY,
    association_name TEXT,
    summary_account INTEGER,
    summary_description TEXT,
    gl_account_no INTEGER,
    account_desc TEXT, 
    holder TEXT,
    investment_type TEXT,
    bank_account INTEGER,
    purchase_date DATE,
    investment_term INTEGER,
    maturity_date DATE,
    investment_rate REAL,
    as_of_date DATETIME,
    current_balance REAL,
    association_report_name TEXT
);
""")

# Create cd_rates table
cursor.execute("""
CREATE TABLE IF NOT EXISTS cd_rates (
    bank_name TEXT,
    bank_code TEXT,
    cd_term TEXT,
    cd_rate REAL,
    cdars_term TEXT,
    cdars_rate REAL,
    special TEXT
);
""")

# Create branch_relationships table
print("Creating branch_relationships table...")
cursor.execute("""
CREATE TABLE IF NOT EXISTS branch_relationships (
    branch_name TEXT PRIMARY KEY,
    alliance_assoc_bank INTEGER,
    banco_popular INTEGER,
    bank_united INTEGER,
    city_national INTEGER,
    enterprise_bank_trust INTEGER,
    first_citizens_bank INTEGER,
    harmony_bank INTEGER,
    pacific_premier_bank INTEGER, 
    pacific_western INTEGER,
    southstate INTEGER,
    sunwest_bank INTEGER,
    capital_one INTEGER
);
""")

# Create ecr_rates table
print("Creating ecr_rates table...")
cursor.execute("""
CREATE TABLE IF NOT EXISTS ecr_rates (
    bank_name TEXT,
    bank_code TEXT,
    ecr_rate REAL,
    PRIMARY KEY (bank_name, bank_code)
);
""")

# Clear existing data (but keep structure)
cursor.execute("DELETE FROM test_data")
cursor.execute("DELETE FROM cd_rates")
cursor.execute("DELETE FROM branch_relationships")
cursor.execute("DELETE FROM ecr_rates")

# Import data from test_data.csv using to_sql method
print("Importing data from test_data.csv...")
test_data_df = pd.read_csv("./data/test_data.csv")
# Using append to respect existing table structure
test_data_df.to_sql('test_data', conn, if_exists='append', index=False)

# Import data from cd_rates.csv
print("Importing data from cd_rates.csv...")
cd_rates_df = pd.read_csv("./data/cd_rates.csv")

# Clean all text columns in the dataframe
print("Cleaning cd_rates data...")
for column in cd_rates_df.columns:
    if cd_rates_df[column].dtype == 'object':
        cd_rates_df[column] = cd_rates_df[column].apply(clean_text)

# Rename columns to match the table structure (convert from "Bank Name" to "bank_name" format)
cd_rates_df.rename(columns={
    "Bank Name": "bank_name",
    "Bank Code": "bank_code",
    "CD Term": "cd_term",
    "CD Rate": "cd_rate",
    "CDARS Term": "cdars_term",
    "CDARS Rate": "cdars_rate",
    "Special": "special"
}, inplace=True)

# Using append to respect existing table structure
cd_rates_df.to_sql('cd_rates', conn, if_exists='append', index=False)

# Import data from branch_relationships.csv
print("Importing data from branch_relationships.csv...")
branch_relationships_df = pd.read_csv("./data/branch_relationships.csv")

# Clean all text columns in the dataframe
print("Cleaning branch_relationships data...")
for column in branch_relationships_df.columns:
    if branch_relationships_df[column].dtype == 'object':
        branch_relationships_df[column] = branch_relationships_df[column].apply(clean_text)

# Rename columns to match the table structure
# The CSV has headers with spaces and special characters, so we need to standardize them
branch_relationships_df.rename(columns={
    "Branch Name": "branch_name",
    "Alliance Assoc.\nBank": "alliance_assoc_bank",
    "Banco Popular": "banco_popular",
    "Bank United": "bank_united",
    "City National": "city_national",
    "Enterprise Bank \n& Trust": "enterprise_bank_trust",
    "First Citizens Bank": "first_citizens_bank",
    "Harmony Bank": "harmony_bank",
    "Pacific Premier \nBank": "pacific_premier_bank",
    "Pacific Western": "pacific_western",
    "SouthState": "southstate",
    "SunWest Bank": "sunwest_bank",
    "Capital One": "capital_one"
}, inplace=True)

# Using append to respect existing table structure
branch_relationships_df.to_sql('branch_relationships', conn, if_exists='append', index=False)

# Import data from ecr_rates.csv
print("Importing data from ecr_rates.csv...")
ecr_rates_df = pd.read_csv("./data/ecr_rates.csv")

# Clean all text columns in the dataframe
print("Cleaning ecr_rates data...")
for column in ecr_rates_df.columns:
    if ecr_rates_df[column].dtype == 'object':
        ecr_rates_df[column] = ecr_rates_df[column].apply(clean_text)

# Rename columns to match the table structure
ecr_rates_df.rename(columns={
    "Bank Name": "bank_name",
    "Bank Code": "bank_code",
    "ECR Rate": "ecr_rate"
}, inplace=True)

# Drop duplicates
print("Removing duplicates from ecr_rates...")
original_count = len(ecr_rates_df)
ecr_rates_df = ecr_rates_df.drop_duplicates(subset=['bank_name', 'bank_code'])
print(f"Removed {original_count - len(ecr_rates_df)} duplicate records")

# Replace table with cleaned data
ecr_rates_df.to_sql('ecr_rates', conn, if_exists='replace', index=False)

# Commit changes and close connection
conn.commit()
conn.close()

print("Data imported successfully!")

