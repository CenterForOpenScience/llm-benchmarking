import pandas as pd

# Load the Excel file and print the first few rows to understand its structure
def inspect_excel(file_path):
    try:
        df = pd.read_excel(file_path)
        print(df.head())
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

if __name__ == "__main__":
    inspect_excel('/app/data/1901_Day1_10-24-2020-VAT_replication_scenario.xlsx')