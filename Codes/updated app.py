import io
import base64
import matplotlib.pyplot as plt
import mysql.connector
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import json
import logging
import os
from datetime import datetime
import re
import matplotlib
from scipy import stats
from huggingface_hub import InferenceClient

matplotlib.use('Agg')

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)

# Database credentials
username = 'root'
password = 'root'
host = 'localhost'
database = 'engine_trend'

# Hugging Face Inference API
HF_API_KEY = "api_key_abcd" 
client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.3", token=HF_API_KEY)

# Load module and channel mappings
try:
    with open('module-info.json', 'r') as f:
        module_columns_map = json.load(f)
    with open('channel-mapping.json', 'r') as f:
        channel_mapping = json.load(f)
        for key in channel_mapping:
            if key != "fixed sensors":
                channel_mapping[key]["all"] = channel_mapping[key]["temperature"] + channel_mapping[key]["pressure"]
except FileNotFoundError as e:
    logging.error(f"Config file missing: {e}")
    raise

# Temporary directory for plots
temp_dir = "/Users/saipriyadipika/Desktop/GTRE/venv/Chatbot integrated analysis model/temp_plots"
os.makedirs(temp_dir, exist_ok=True)

# Global storage for current run data
current_run_data = {
    "run_id": None,
    "selected_module": None,
    "table_name": None,
    "df": None
}

def get_db_connection():
    logging.debug("Attempting database connection")
    try:
        conn = mysql.connector.connect(user=username, password=password, host=host, database=database)
        logging.debug("Database connection successful")
        return conn
    except mysql.connector.Error as e:
        logging.error(f"Database connection failed: {e}")
        raise

def extract_table_name(run_id):
    match = re.match(r'([A-Z]\d+B\d+)', run_id)
    return match.group(1) if match else None

def get_previous_run_id(current_run_id):
    match = re.match(r'([A-Z]\d+B\d+)R(\d+)', current_run_id)
    if not match:
        return None
    table_prefix, run_num = match.groups()
    prev_run_num = int(run_num) - 1
    return f"{table_prefix}R{prev_run_num:04d}"

def generate_query_data(run_id, module_name, channel_mapping, module_info):
    table_name = extract_table_name(run_id)
    if not table_name:
        logging.error(f"Invalid run_id: {run_id}")
        return None

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(f"DESCRIBE `{table_name}`")
        all_columns = [desc[0] for desc in cursor.fetchall() if desc[0] not in ['Timestamp', 'run_id', 'run_date']]

        module_name_clean, analysis_type = module_name.rsplit(' (', 1)
        analysis_type = analysis_type[:-1]
        if module_name_clean not in module_info:
            logging.error(f"Module {module_name_clean} not in module_info")
            return None

        parameter_names = module_info[module_name_clean]
        relevant_channels = channel_mapping.get(table_name, {}).get("temperature" if analysis_type == "temp analysis" else "pressure", [])
        selected_columns = []
        for col in all_columns:
            parts = col.split('_')
            if len(parts) >= 2 and parts[0] in parameter_names and parts[1] in relevant_channels:
                selected_columns.append(col)
        fixed_columns = []
        for col in all_columns:
            parts = col.split('_')
            if len(parts) >= 2 and parts[1] in channel_mapping["fixed sensors"]:
                fixed_columns.append(col)
        columns_to_fetch = list(set(selected_columns + fixed_columns))

        if not columns_to_fetch:
            logging.error(f"No columns to fetch for {run_id}, {module_name}")
            return None

        columns_str = ', '.join([f"`{col}`" for col in columns_to_fetch])
        query = f"SELECT Timestamp, run_date, {columns_str} FROM `{table_name}` WHERE run_id = %s"
        logging.debug(f"Executing query: {query} with run_id: {run_id}")
        cursor.execute(query, (run_id,))
        rows = cursor.fetchall()
        if not rows:
            logging.error(f"No data for run {run_id}")
            return None

        base_date = rows[0][1]
        timestamps = [datetime.combine(base_date, datetime.min.time()) + row[0] for row in rows]
        data = {col: [row[i + 2] for row in rows] for i, col in enumerate(columns_to_fetch)}
        df = pd.DataFrame(data)
        df['Timestamp'] = timestamps
        df = df.reset_index(drop=True)  # Ensure consistent integer index

        # Detect outliers
        outlier_indices = []
        for col in df.columns:
            if col == 'Timestamp':
                continue
            values = df[col].dropna().astype(float)
            if len(values) < 2:
                continue
            z_scores = np.abs(stats.zscore(values))
            col_outliers = df.index[z_scores > 3].tolist()
            outlier_indices.extend(col_outliers)
        outlier_indices = sorted(list(set(outlier_indices)))

        # Sample data
        max_rows =80
        if len(df) <= max_rows:
            sampled_df = df
        else:
            sampled_indices = outlier_indices
            if len(sampled_indices) > max_rows:
                sampled_indices = sampled_indices[:max_rows]
            else:
                regular_step = max(1, len(df) // (max_rows - len(outlier_indices)))
                regular_indices = list(set(range(0, len(df), regular_step)) - set(outlier_indices))
                sampled_indices.extend(regular_indices[:max_rows - len(sampled_indices)])
            sampled_indices = sorted(sampled_indices[:max_rows])
            sampled_df = df.iloc[sampled_indices].reset_index(drop=True)

        return sampled_df
    except mysql.connector.Error as e:
        logging.error(f"Failed to fetch data: {e}")
        return None
    finally:
        cursor.close()
        conn.close()

def generate_plot_image(df, run_id, selected_module, single_col=None):
    module_name, analysis_type = selected_module.rsplit(' (', 1)
    analysis_type = analysis_type[:-1]

    if single_col:
        selected_columns = [single_col] if single_col in df.columns else []
    else:
        parameter_names = module_columns_map[module_name]
        table_name = extract_table_name(run_id)
        relevant_channels = channel_mapping.get(table_name, {}).get("temperature" if analysis_type == "temp analysis" else "pressure", [])
        selected_columns = []
        for col in df.columns:
            parts = col.split('_')
            if len(parts) >= 2 and parts[0] in parameter_names and parts[1] in relevant_channels and col != 'Timestamp':
                selected_columns.append(col)
    fixed_columns = []
    for col in df.columns:
        parts = col.split('_')
        if len(parts) >= 2 and parts[1] in channel_mapping["fixed sensors"] and col != 'Timestamp':
            fixed_columns.append(col)

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()

    if df.empty or not selected_columns:
        timestamps = pd.date_range(start="2023-01-01", periods=10, freq='T')
        test_data = np.arange(10)
        ax1.plot(timestamps, test_data, label="Test Data", color='#FF6B6B', linewidth=2)
        ax1.set_title(f"{module_name} for run: {run_id}", fontsize=14)
    else:
        timestamps = df['Timestamp']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD']
        default_fixed_color = '#333333'
        purple_fixed_color = '#800080'

        for i, column in enumerate(selected_columns):
            values = df[column].dropna()
            if not values.empty:
                ax1.plot(timestamps[values.index], values, label=column, color=colors[i % len(colors)], linewidth=2)

        for i, column in enumerate(fixed_columns):
            values = df[column].dropna()
            if not values.empty:
                # Use purple for the first fixed sensor, default color for others
                color = purple_fixed_color if i == 0 else default_fixed_color
                ax2.plot(timestamps[values.index], values, label=column, color=color, linewidth=2, linestyle=':')

        ax1.set_xlabel('Timestamp', fontsize=12)
        ax1.set_ylabel('Module Parameters', fontsize=12, color='#1E3A8A')
        ax2.set_ylabel('Fixed Sensors', fontsize=12, color=default_fixed_color)
        ax1.legend(title="Parameters", loc='upper left', bbox_to_anchor=(1.15, 0.5), fontsize=15)
        ax2.legend(title="Fixed Sensors", loc='upper left', bbox_to_anchor=(1.15, 1), fontsize=15)
        plt.title(f"{module_name} for run: {run_id}", fontsize=14)

    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.xticks(rotation=45, ha='right', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot_filename = f"temp_plot_{run_id}_{selected_module.replace(' ', '_')}_{os.urandom(4).hex()}.png"
    plot_path = os.path.join(temp_dir, plot_filename)
    plt.savefig(plot_path, format='png', dpi=150, bbox_inches='tight')
    plt.close()

    img = io.BytesIO()
    with open(plot_path, 'rb') as f:
        img.write(f.read())
    img.seek(0)
    img_b64 = base64.b64encode(img.read()).decode('utf-8')
    os.remove(plot_path)
    return img_b64

def analyze_with_mistral(df, selected_module):
    logging.debug("Starting model analysis")
    if df.empty:
        return "Error: No data to analyze."

    timestamps = df['Timestamp']
    data_columns = {col: df[col].tolist() for col in df.columns if col != 'Timestamp'}
    outlier_indices = []
    for col in data_columns:
        values = [v for v in data_columns[col] if v is not None]
        if len(values) < 2:
            continue
        z_scores = np.abs(stats.zscore(values))
        col_outliers = [i for i, z in enumerate(z_scores) if z > 3]
        outlier_indices.extend(col_outliers)
    outlier_indices = sorted(list(set(outlier_indices)))

    max_rows = 80
    if len(timestamps) <= max_rows:
        sampled_indices = list(range(len(timestamps)))
    else:
        sampled_indices = outlier_indices
        if len(sampled_indices) > max_rows:
            sampled_indices = sampled_indices[:max_rows]
        else:
            regular_step = max(1, len(timestamps) // (max_rows - len(outlier_indices)))
            regular_indices = list(set(range(0, len(timestamps), regular_step)) - set(outlier_indices))
            sampled_indices.extend(regular_indices[:max_rows - len(sampled_indices)])
        sampled_indices = sorted(sampled_indices[:max_rows])

    sampled_timestamps = timestamps.iloc[sampled_indices].tolist()
    sampled_data_columns = {col: df[col].iloc[sampled_indices].tolist() for col in data_columns}
    sampled_timestamps_str = [t.strftime('%H:%M:%S') if isinstance(t, datetime) else str(t) for t in sampled_timestamps]
    headers = ['Timestamp'] + list(sampled_data_columns.keys())
    table_rows = [','.join(str(val) if val is not None else 'None' for val in row)
                  for row in zip(sampled_timestamps_str, *[sampled_data_columns[col] for col in sampled_data_columns.keys()])]
    table_text = ','.join(headers) + '\n' + '\n'.join(table_rows)

    f1_col = next((col for col in sampled_data_columns if len(col.split('_')) >= 2 and col.split('_')[1] in channel_mapping["fixed sensors"]), None)
    f2_col = next((col for col in sampled_data_columns if len(col.split('_')) >= 2 and col.split('_')[1] in channel_mapping["fixed sensors"] and col != f1_col), None)
    module_cols = [col for col in sampled_data_columns if col not in [f1_col, f2_col]]

    if not f1_col or not f2_col:
        logging.warning("Fixed sensors not found")
        return "Error: Fixed sensors not found."

    prompt = f"""
Analyze this sampled time series data for module '{selected_module}'
Table:
{table_text}

- Fixed parameters: '{f1_col}' (f1) and '{f2_col}' (f2) set the trend.
- Module parameters: {', '.join(module_cols)} should follow f1 and f2's trend.

Analyze the data and provide comments on:
- Stagnation
- Erratic readings or not following trend properly
- Sensor malfunctions (all zeros or missing data). (in this case, just comment 'sensor_name hasn't captured data')
- Any other notable patterns, deviations or anomalies.

Make your response short, concise and to the point. Mention the value and timestamp of anomalies if they exist.
It need not be full English sentences, but it should include all important details.
Also, mention the overall trend comment at the end, e.g., if trends are normal and engine can be cleared for the next run, or engine should be inspected.

Example:
Stagnation: No significant periods of stagnation observed.
Erratic readings: S1_1_Hz and S5_9_g occasionally deviate from the trend set by f1 and f2, especially between timestamps 12:01:28-12:02:00 and 12:03:04-12:03:16.
Sensor malfunctions: Sensor S2_3_Hz hasn't captured data.
Anomalies: Sensor S3_5_psi shows an unusual drop to 1.09072 at timestamp 12:14:11, which is significantly lower than the rest of the data.

Overall trend: The data shows some deviations, but the overall trend appears to be normal. However, the engine should be inspected, particularly the S3_5_psi sensor.

For erratic readings, mention significant vibrations or negative values.
For missing data, use ranges (e.g., 'sensor hasn't captured data at 12:00:00-12:10:00') if multiple points are missing.
Leave one line before 'Overall trend'.
"""

    try:
        comments = client.text_generation(
            prompt,
            max_new_tokens=800,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            return_full_text=False
        ).strip()
        logging.debug(f"Analysis: {comments}")
        return comments or "No analysis generated."
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        return f"Error: Analysis failed. {str(e)}"

def chat_with_mistral(query):
    global current_run_data
    if not current_run_data["run_id"] or not current_run_data["selected_module"] or current_run_data["df"] is None:
        return {"response": "No current run data.", "comp_image": None}

    comp_run_id = None
    comp_type = None
    if "compare" in query.lower():
        words = query.split()
        comp_run_id = words[-1]
        comp_type = "previous" if "previous" in query.lower() else "specific"

    if comp_type:
        module_name, analysis_type = current_run_data["selected_module"].rsplit(' (', 1)
        analysis_type = analysis_type[:-1].replace(" analysis", "")
        module_display = f"{analysis_type.capitalize()} analysis"

        if comp_type == "previous":
            comp_run_id = get_previous_run_id(current_run_data["run_id"])
            comp_table_name = current_run_data["table_name"]
        else:
            comp_table_name = extract_table_name(comp_run_id)
            if not comp_table_name:
                return {"response": f"Invalid run ID: {comp_run_id}.", "comp_image": None}

        comp_df = generate_query_data(comp_run_id, current_run_data["selected_module"], channel_mapping, module_columns_map)
        if comp_df is None:
            return {"response": f"Data for run: {comp_run_id} not present in database.", "comp_image": None}

        # Prepare data for model analysis
        curr_df = current_run_data["df"]
        curr_timestamps = curr_df['Timestamp']
        comp_timestamps = comp_df['Timestamp']

        # Sample current run data
        curr_outlier_indices = []
        for col in curr_df.columns:
            if col == 'Timestamp':
                continue
            values = curr_df[col].dropna().astype(float)
            if len(values) < 2:
                continue
            z_scores = np.abs(stats.zscore(values))
            col_outliers = curr_df.index[z_scores > 3].tolist()
            curr_outlier_indices.extend(col_outliers)
        curr_outlier_indices = sorted(list(set(curr_outlier_indices)))

        max_rows = 40 
        if len(curr_timestamps) <= max_rows:
            curr_sampled_indices = list(range(len(curr_timestamps)))
        else:
            curr_sampled_indices = curr_outlier_indices
            if len(curr_sampled_indices) > max_rows:
                curr_sampled_indices = curr_sampled_indices[:max_rows]
            else:
                regular_step = max(1, len(curr_df) // (max_rows - len(curr_outlier_indices)))
                regular_indices = list(set(range(0, len(curr_df), regular_step)) - set(curr_outlier_indices))
                curr_sampled_indices.extend(regular_indices[:max_rows - len(curr_sampled_indices)])
            curr_sampled_indices = sorted(curr_sampled_indices[:max_rows])

        curr_sampled_timestamps = curr_timestamps.iloc[curr_sampled_indices].tolist()
        curr_sampled_data = {col: curr_df[col].iloc[curr_sampled_indices].tolist() for col in curr_df.columns if col != 'Timestamp'}
        curr_sampled_timestamps_str = [t.strftime('%H:%M:%S') if isinstance(t, datetime) else str(t) for t in curr_sampled_timestamps]
        curr_headers = ['Timestamp'] + list(curr_sampled_data.keys())
        curr_table_rows = [','.join(str(val) if val is not None else 'None' for val in row)
                           for row in zip(curr_sampled_timestamps_str, *[curr_sampled_data[col] for col in curr_sampled_data.keys()])]
        curr_table_text = ','.join(curr_headers) + '\n' + '\n'.join(curr_table_rows)

        
        comp_outlier_indices = []
        for col in comp_df.columns:
            if col == 'Timestamp':
                continue
            values = comp_df[col].dropna().astype(float)
            if len(values) < 2:
                continue
            z_scores = np.abs(stats.zscore(values))
            col_outliers = comp_df.index[z_scores > 3].tolist()
            comp_outlier_indices.extend(col_outliers)
        comp_outlier_indices = sorted(list(set(comp_outlier_indices)))

        if len(comp_timestamps) <= max_rows:
            comp_sampled_indices = list(range(len(comp_timestamps)))
        else:
            comp_sampled_indices = comp_outlier_indices
            if len(comp_sampled_indices) > max_rows:
                comp_sampled_indices = comp_sampled_indices[:max_rows]
            else:
                regular_step = max(1, len(comp_df) // (max_rows - len(comp_outlier_indices)))
                regular_indices = list(set(range(0, len(comp_df), regular_step)) - set(comp_outlier_indices))
                comp_sampled_indices.extend(regular_indices[:max_rows - len(comp_outlier_indices)])
            comp_sampled_indices = sorted(comp_sampled_indices[:max_rows])

        comp_sampled_timestamps = comp_timestamps.iloc[comp_sampled_indices].tolist()
        comp_sampled_data = {col: comp_df[col].iloc[comp_sampled_indices].tolist() for col in comp_df.columns if col != 'Timestamp'}
        comp_sampled_timestamps_str = [t.strftime('%H:%M:%S') if isinstance(t, datetime) else str(t) for t in comp_sampled_timestamps]
        comp_headers = ['Timestamp'] + list(comp_sampled_data.keys())
        comp_table_rows = [','.join(str(val) if val is not None else 'None' for val in row)
                           for row in zip(comp_sampled_timestamps_str, *[comp_sampled_data[col] for col in comp_sampled_data.keys()])]
        comp_table_text = ','.join(comp_headers) + '\n' + '\n'.join(comp_table_rows)

        # Identify fixed and module parameters
        curr_f1_col = next((col for col in curr_sampled_data if len(col.split('_')) >= 2 and col.split('_')[1] in channel_mapping["fixed sensors"]), None)
        curr_f2_col = next((col for col in curr_sampled_data if len(col.split('_')) >= 2 and col.split('_')[1] in channel_mapping["fixed sensors"] and col != curr_f1_col), None)
        curr_module_cols = [col for col in curr_sampled_data if col not in [curr_f1_col, curr_f2_col]]

        comp_f1_col = next((col for col in comp_sampled_data if len(col.split('_')) >= 2 and col.split('_')[1] in channel_mapping["fixed sensors"]), None)
        comp_f2_col = next((col for col in comp_sampled_data if len(col.split('_')) >= 2 and col.split('_')[1] in channel_mapping["fixed sensors"] and col != comp_f1_col), None)
        comp_module_cols = [col for col in comp_sampled_data if col not in [comp_f1_col, comp_f2_col]]

        if not (curr_f1_col and curr_f2_col and comp_f1_col and comp_f2_col):
            logging.warning("Fixed sensors not found in one or both runs")
            return {"response": "Error: Fixed sensors not found.", "comp_image": None}

        # Match parameters by sensor prefix (e.g., S2)
        curr_param_map = {col.split('_')[0]: col for col in curr_module_cols}
        comp_param_map = {col.split('_')[0]: col for col in comp_module_cols}
        common_params = set(curr_param_map.keys()) & set(comp_param_map.keys())

        if not common_params:
            logging.warning("No common parameters found for comparison")
            return {"response": "Error: No common parameters found for comparison.", "comp_image": None}

        # Model prompt for parameter-wise comparison
        param_pairs = [f"{curr_param_map[p]} in {current_run_data['run_id']} vs {comp_param_map[p]} in {comp_run_id}" for p in common_params]
        prompt = f"""
Compare two runs of time series data for module '{current_run_data["selected_module"]}':
- Current run: {current_run_data["run_id"]}
  Table:
  {curr_table_text}
  Fixed parameters: '{curr_f1_col}' (f1) and '{curr_f2_col}' (f2) set the trend.
  Module parameters: {', '.join(curr_module_cols)} should follow f1 and f2's trend.

- Comparison run: {comp_run_id}
  Table:
  {comp_table_text}
  Fixed parameters: '{comp_f1_col}' (f1) and '{comp_f2_col}' (f2) set the trend.
  Module parameters: {', '.join(comp_module_cols)} should follow f1 and f2's trend.

Compare each common sensor parameter (e.g., {', '.join(common_params)}) individually:
{'; '.join(param_pairs)}

For each parameter, analyze:
- Trend: Describe behavior (e.g., upward, stable, erratic) relative to fixed parameters in each run.
- Stagnation: Note flat periods (specify timestamps).
- Erratic readings: Identify deviations, vibrations, or negative values (specify timestamps).
- Sensor malfunctions: Report all zeros or missing data (e.g., 'sensor_name hasn't captured data', with time ranges if multiple points).
- Anomalies: Highlight significant drops, spikes, or outliers with exact values and timestamps.

Use full column names for each run (e.g., {curr_param_map[list(common_params)[0]]} for {current_run_data["run_id"]}, {comp_param_map[list(common_params)[0]]} for {comp_run_id}).
Format:
{module_display} comparison for {current_run_data["run_id"]} and {comp_run_id}
{''.join([f"\nParameter {p}:\n- Trends: <trend for {curr_param_map[p]} vs {comp_param_map[p]}>.\n- Stagnation: <periods for each>.\n- Erratic readings: <deviations for each>.\n- Sensor malfunctions: <issues for each>.\n- Anomalies: <values and timestamps for each>." for p in common_params])}

Overall trend: <normal/inspect engine, with detailed reasoning>.

Example:
Pressure analysis comparison for V12B34R1233 and P20B78R1998
Parameter S2:
- Trends: S2_234_Pa in V12B34R1233 increases steadily, aligns with S1_3_Pa, S2_9_Pa; S2_567_Psi in P20B78R1998 fluctuates, poorly tracks S1_3_Psi, S2_9_Psi.
- Stagnation: S2_234_Pa flat 12:03:00-12:04:00 in V12B34R1233; S2_567_Psi flat 12:06:00-12:08:00 in P20B78R1998.
- Erratic readings: S2_234_Pa vibrates 12:01:30-12:02:10 in V12B34R1233; S2_567_Psi negative values 12:05:00-12:05:30 in P20B78R1998.
- Sensor malfunctions: None in V12B34R1233; S2_567_Psi hasn't captured data 12:00:00-12:01:00 in P20B78R1998.
- Anomalies: S2_234_Pa spikes to 5.2 at 12:02:15 in V12B34R1233; S2_567_Psi drops to -0.50 at 12:05:00 in P20B78R1998.

Overall trend: Inspect engine, erratic behavior and anomalies in P20B78R1998 suggest potential issues.

Use 'HH:MM:SS' for timestamps. Leave one line before 'Overall trend'.
"""

        try:
            response = client.text_generation(
                prompt,
                max_new_tokens=1500,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                return_full_text=False
            ).strip()
            logging.debug(f"Comparison analysis: {response}")
        except Exception as e:
            logging.error(f"Comparison analysis failed: {e}")
            response = f"Error: Comparison analysis failed. {str(e)}"

        comp_image = generate_plot_image(comp_df, comp_run_id, current_run_data["selected_module"])
        return {"response": response or "No comparison generated.", "comp_image": comp_image}

    trend_match = re.search(r'explain the trend of (\S+)', query, re.IGNORECASE)
    if trend_match:
        col_name = trend_match.group(1)
        if current_run_data["df"] is not None and col_name in current_run_data["df"].columns:
            # Prepare data for model analysis
            curr_df = current_run_data["df"]
            timestamps = curr_df['Timestamp']
            values = curr_df[col_name].dropna()
            if values.empty:
                return {"response": f"{col_name}: No data available.", "comp_image": None}

            # Sample data to keep input manageable
            outlier_indices = []
            z_scores = np.abs(stats.zscore(values.astype(float)))
            col_outliers = values.index[z_scores > 3].tolist()
            outlier_indices.extend(col_outliers)
            outlier_indices = sorted(list(set(outlier_indices)))

            max_rows = 80
            if len(timestamps) <= max_rows:
                sampled_indices = list(range(len(timestamps)))
            else:
                sampled_indices = outlier_indices
                if len(sampled_indices) > max_rows:
                    sampled_indices = sampled_indices[:max_rows]
                else:
                    regular_step = max(1, len(timestamps) // (max_rows - len(outlier_indices)))
                    regular_indices = list(set(range(0, len(timestamps), regular_step)) - set(outlier_indices))
                    sampled_indices.extend(regular_indices[:max_rows - len(sampled_indices)])
                sampled_indices = sorted(sampled_indices[:max_rows])

            sampled_timestamps = timestamps.iloc[sampled_indices].tolist()
            sampled_data = {col_name: curr_df[col_name].iloc[sampled_indices].tolist()}
            sampled_timestamps_str = [t.strftime('%H:%M:%S') if isinstance(t, datetime) else str(t) for t in sampled_timestamps]
            headers = ['Timestamp', col_name]
            table_rows = [','.join(str(val) if val is not None else 'None' for val in row)
                          for row in zip(sampled_timestamps_str, sampled_data[col_name])]
            table_text = ','.join(headers) + '\n' + '\n'.join(table_rows)

            # Identify fixed sensors
            f1_col = next((col for col in curr_df.columns if col != 'Timestamp' and len(col.split('_')) >= 2 and col.split('_')[1] in channel_mapping["fixed sensors"]), None)
            f2_col = next((col for col in curr_df.columns if col != 'Timestamp' and len(col.split('_')) >= 2 and col.split('_')[1] in channel_mapping["fixed sensors"] and col != f1_col), None)

            if not f1_col or not f2_col:
                logging.warning("Fixed sensors not found")
                return {"response": "Error: Fixed sensors not found.", "comp_image": None}

            # Model prompt for trend explanation
            prompt = f"""
Analyze the trend of the sensor parameter '{col_name}' in the following time series data for module '{current_run_data["selected_module"]}':
Table:
{table_text}

- Fixed parameters: '{f1_col}' (f1) and '{f2_col}' (f2) set the reference trend.
- '{col_name}' should follow the trend of f1 and f2.

Provide a brief analysis:
- Describe the trend of '{col_name}' (e.g., upward, downward, stable, erratic).
- Note if it aligns with the fixed parameters f1 and f2.
- Provide an inference about the engine's condition based on the trend (e.g., normal operation, potential issues requiring inspection).
- If applicable, mention specific anomalies (e.g., spikes, drops) with values and timestamps.
- Keep the response concise, avoiding full sentences.


Format:
Trend: <trend description, alignment with f1 and f2>.
Anomalies: <specific values and timestamps, if any>.
Inference: <engine condition>.

Example:
Trend: S2_234_Pa increases steadily, aligns with S1_3_Pa, S2_9_Pa.
Anomalies: Spikes to 5.2 at 12:02:15.
Inference: Normal operation, engine can be cleared.

Use 'HH:MM:SS' for timestamps.
"""

            try:
                response = client.text_generation(
                    prompt,
                    max_new_tokens=300,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    return_full_text=False
                ).strip()
                logging.debug(f"Trend analysis for {col_name}: {response}")
            except Exception as e:
                logging.error(f"Trend analysis failed for {col_name}: {e}")
                response = f"Error: Trend analysis failed. {str(e)}"

            comp_image = generate_plot_image(current_run_data["df"], current_run_data["run_id"], current_run_data["selected_module"], single_col=col_name)
            return {"response": response or f"No trend analysis generated for {col_name}.", "comp_image": comp_image}
        return {"response": f"Data not available for {col_name}.", "comp_image": None}

    return {"response": "Query not recognized. Supported: 'compare with previous run', 'compare with run <run_id>', 'explain the trend of <col_name>'.", "comp_image": None}

@app.route('/')
def index():
    module_options = []
    for module_name in module_columns_map.keys():
        module_options.append(f"{module_name} (temp analysis)")
        module_options.append(f"{module_name} (pressure analysis)")
    return render_template('index.html', module_names=module_options)

@app.route('/generate_plot', methods=['POST'])
def generate_plot():
    global current_run_data
    logging.debug("Received POST request")
    data = request.get_json()
    run_id = data.get('run_id')
    selected_module = data.get('selected_module')
    logging.debug(f"run_id: {run_id}, selected_module: {selected_module}")

    if not selected_module or not run_id:
        return jsonify({'message': 'Please provide Run ID and module.', 'image': None, 'comments': None, 'table': None})

    table_name = extract_table_name(run_id)
    if not table_name:
        return jsonify({'message': f"Invalid run_id: {run_id}.", 'image': None, 'comments': None, 'table': None})

    df = generate_query_data(run_id, selected_module, channel_mapping, module_columns_map)
    if df is None:
        return jsonify({'message': f"No data for run {run_id}.", 'image': None, 'comments': None, 'table': None})

    current_run_data.update({
        "run_id": run_id,
        "selected_module": selected_module,
        "table_name": table_name,
        "df": df
    })

    image = generate_plot_image(df, run_id, selected_module)
    comments = analyze_with_mistral(df, selected_module)

    return jsonify({'image': image, 'comments': comments, 'table': table_name})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data.get('query', '').strip()
    if not query:
        return jsonify({'response': 'Please enter a query.', 'comp_image': None})
    result = chat_with_mistral(query)
    return jsonify({'response': result["response"], 'comp_image': result["comp_image"]})

if __name__ == '__main__':
    logging.info("Starting Flask server")
    try:
        app.run(debug=True, use_reloader=False, port=5000)
    except Exception as e:
        logging.error(f"Flask server failed: {e}")
        raise