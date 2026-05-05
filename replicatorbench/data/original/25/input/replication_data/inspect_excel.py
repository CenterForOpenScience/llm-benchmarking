import pandas as pd
import sys
import subprocess

EXCEL_PATH = '/app/data/1901_Day1_10-24-2020-VAT_replication_scenario.xlsx'

print('Python executable:', sys.executable)
print('sys.path:', sys.path)

# Ensure openpyxl is available
try:
    import openpyxl
    print('openpyxl version:', openpyxl.__version__)
except Exception as e:
    print('openpyxl not available or import failed:', e)
    try:
        print('Attempting to install openpyxl into /opt/venv')
        subprocess.check_call(['/opt/venv/bin/python', '-m', 'pip', 'install', '--no-cache-dir', 'openpyxl'])
        import openpyxl
        print('Successfully installed openpyxl version:', openpyxl.__version__)
    except Exception as ie:
        print('Failed to install openpyxl:', ie)

# Load the Excel file and print the first few rows to understand its structure
def inspect_excel(file_path):
    try:
        df = pd.read_excel(file_path)
        print(df.head())
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

if __name__ == "__main__":
    inspect_excel(EXCEL_PATH)