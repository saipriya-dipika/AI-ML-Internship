import pandas as pd
import mysql.connector
import os
import json
import re

# Define the folder path containing CSV files
extract_folder = "/Users/saipriyadipika/Desktop/GTRE/venv/Chatbot integrated analysis model/datasets"
json_file_path = "/Users/saipriyadipika/Desktop/GTRE/venv/Chatbot integrated analysis model/channel-mapping.json" 

# Load the JSON file with engine_build to channel number mapping
try:
    with open(json_file_path, 'r') as f:
        engine_build_channels = json.load(f)
    # Flatten the channel lists (ignore temperature/pressure subcategories)
    for key in engine_build_channels:
        engine_build_channels[key] = engine_build_channels[key]["temperature"] + engine_build_channels[key]["pressure"]
    print("JSON file loaded successfully with flattened channels:", engine_build_channels)
except FileNotFoundError as e:
    print(f"Error: JSON file not found at {json_file_path}. Please provide the correct path.")
    exit()
except json.JSONDecodeError as e:
    print(f"Error: Invalid JSON format in {json_file_path}: {e}")
    exit()

# Get sorted list of CSV files
csv_files = sorted([os.path.join(extract_folder, f) for f in os.listdir(extract_folder) if f.endswith('.csv')])

# MySQL connection details
username = 'root'  
password = 'root'  
host = 'localhost'
database = 'engine_trend'

# Connect to MySQL
try:
    connection = mysql.connector.connect(user=username, password=password, host=host, database=database)
    cursor = connection.cursor()
    print("MySQL connection established successfully!")
except mysql.connector.Error as err:
    print(f"Error: {err}")
    exit()

def table_exists(table_name):
    cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
    return cursor.fetchone() is not None

def get_existing_columns(table_name):
    cursor.execute(f"SHOW COLUMNS FROM `{table_name}`")
    return {col[0] for col in cursor.fetchall()}

# Iterate through each CSV file
for input_file in csv_files:
    print(f"\nProcessing file: {input_file}")

    # Read metadata
    try:
        meta_data = pd.read_csv(input_file, header=None, nrows=13, skip_blank_lines=False, encoding='ISO-8859-1')
    except UnicodeDecodeError as e:
        print(f"Error reading CSV file {input_file}: {e}")
        continue

    # Extract run_id components
    engine_id = str(meta_data.iloc[2, 0]).replace("ENGINE :", "").strip() if pd.notna(meta_data.iloc[2, 0]) else "unknown"
    build_no = str(meta_data.iloc[3, 0]).replace("BUILD NO :", "").strip() if pd.notna(meta_data.iloc[3, 0]) else "unknown"
    run_no = str(meta_data.iloc[4, 0]).replace("RUN NO :", "").strip() if pd.notna(meta_data.iloc[4, 0]) else "unknown"

    run_id = f"{engine_id}B{build_no}R{run_no}"
    run_date = str(meta_data.iloc[10, 0].replace("Date: ", "")).strip()

    # Format run_date (only date, no time)
    try:
        run_date = pd.to_datetime(run_date, format='%d:%m:%Y', errors='coerce').strftime('%Y-%m-%d')
    except:
        run_date = "2000-01-01"

    print(f"run_id: {run_id}, run_date: {run_date}")

    # Parse run_id to extract engine_number, build_number, and run_number
    try:
        match = re.match(r'([A-Z])(\d+)B(\d+)R(\d+)', run_id)
        if match:
            engine_letter, engine_digits, build_number, run_number = match.groups()
            engine_number = f"{engine_letter}{engine_digits}" 
            engine_build_key = f"{engine_number}B{build_number}"  
            table_name = engine_build_key 
            print(f"Parsed: engine_number={engine_number}, build_number={build_number}, run_number={run_number}")
        else:
            raise ValueError("Invalid run_id format")
    except ValueError:
        print(f"Error: Could not parse run_id {run_id}. Skipping file.")
        continue

    # Check JSON for matching key and get channel numbers
    if engine_build_key in engine_build_channels:
        selected_channels = engine_build_channels[engine_build_key]
        print(f"Found matching key {engine_build_key} in JSON. Selected channels: {selected_channels}")
    else:
        print(f"Warning: No matching key {engine_build_key} found in JSON. Skipping file.")
        continue

    # Read header rows
    header_rows = pd.read_csv(input_file, header=None, nrows=3, skiprows=13, skip_blank_lines=False, encoding='ISO-8859-1')
    header_rows = header_rows.fillna('unknown')

    # Combine headers
    combined_header = [f"{header_rows.iloc[0, i]}_{header_rows.iloc[1, i]}_{header_rows.iloc[2, i]}" for i in range(1, header_rows.shape[1])]
    combined_header.insert(0, 'Timestamp')
    combined_header = [col.replace(" ", "_").replace(":", "_").replace("-", "_") for col in combined_header]

    # print(f"All headers: {combined_header}")

    # Filter headers based on selected channels from JSON
    selected_headers = ['Timestamp'] + [col for col in combined_header[1:] if col.split('_')[1] in selected_channels]
    print(f"Selected headers: {selected_headers}")

    # Read main data
    df = pd.read_csv(input_file, header=None, skiprows=17, encoding='ISO-8859-1')
    df.columns = combined_header

    # Keep only selected columns
    df = df[selected_headers]

    # Convert Timestamp to string (HH:MM:SS), handle NaT explicitly
    def format_time(ts):
        try:
            return pd.to_datetime(ts, format='%H:%M:%S').strftime('%H:%M:%S')
        except (ValueError, TypeError):
            return '00:00:00'  # Default for invalid timestamps

    df['Timestamp'] = df['Timestamp'].apply(format_time)

    # Convert selected columns to numeric
    for col in df.columns[1:]:  # Skip Timestamp
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Create or modify table based on engine_build_key
    if not table_exists(table_name):
        # Create new table for this engine-build combination
        create_table_sql = f"""CREATE TABLE `{table_name}` (
            `Timestamp` TIME,
            `run_id` VARCHAR(20),
            `run_date` DATE
        """

        # Add selected columns
        for col in selected_headers[1:]:  # Skip Timestamp
            create_table_sql += f", `{col}` FLOAT DEFAULT 0.0"
        create_table_sql += ");"

        print(f"Executing SQL: {create_table_sql}")
        cursor.execute(create_table_sql)
        print(f"Table '{table_name}' created successfully")
    else:
        # Alter existing table to add new columns if any
        existing_columns = get_existing_columns(table_name)
        new_columns = set(selected_headers[1:]) - existing_columns

        for col in new_columns:
            alter_sql = f"ALTER TABLE `{table_name}` ADD COLUMN `{col}` FLOAT DEFAULT 0.0"
            print(f"Executing SQL: {alter_sql}")
            cursor.execute(alter_sql)

    # Prepare and execute insert statement
    columns_to_insert = ['Timestamp', 'run_id', 'run_date'] + [col for col in selected_headers if col != 'Timestamp']
    insert_sql = f"INSERT INTO `{table_name}` ({','.join([f'`{col}`' for col in columns_to_insert])}) VALUES ({','.join(['%s'] * len(columns_to_insert))})"
    print(f"Executing SQL: {insert_sql}")

    # Insert data with debugging
    for index, row in df.iterrows():
        values = [row['Timestamp'], run_id, run_date] + [row[col] for col in selected_headers if col != 'Timestamp']
        try:
            print(values)
            cursor.execute(insert_sql, values)
        except mysql.connector.Error as err:
            print(f"Error inserting row {index}: {err}")
            print(f"Values: {values}")
            raise  # Re-raise to stop and debug

    connection.commit()
    print(f"Data from {input_file} inserted successfully into table '{table_name}'")

# Verify row count for each table
cursor.execute(f"SHOW TABLES")
tables = cursor.fetchall()
for table in tables:
    table_name = table[0]
    cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
    row_count = cursor.fetchone()[0]
    print(f"Number of rows in table '{table_name}': {row_count}")

# Clean up
cursor.close()
connection.close()

print("All files processed successfully")