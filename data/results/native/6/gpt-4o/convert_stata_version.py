import pandas as pd

# Load the original Stata file
original_file_path = './GSSreplication.dta'
converted_file_path = './GSSreplication_converted.dta'

# Read the original Stata file
try:
    data = pd.read_stata(original_file_path, convert_categoricals=False)
    # Save the data to a new Stata file with a supported version
    data.to_stata(converted_file_path, version=118)
    print('Conversion successful. File saved as:', converted_file_path)
except Exception as e:
    print('Error during conversion:', str(e))