import streamlit as st
import pandas as pd
import time
from datetime import datetime, timedelta
import os # Import the os module for path manipulation

# This script assumes the user's 'speedtest.py' is in the same directory.
# We are importing the Speedtest class from the user-provided file.
try:
    import speedtest
except ImportError:
    st.error("Error: The 'speedtest.py' file was not found. Please ensure it is in the same directory as this script.")
    st.stop()


# --- Configuration ---
TEST_INTERVAL_MINUTES = 10
TEST_INTERVAL_SECONDS = TEST_INTERVAL_MINUTES * 60

# Define the path for the history file.
# It will be stored in the same directory as the application.
# The systemd service's WorkingDirectory is /opt/speedtest-dashboard,
# so this path will resolve correctly relative to that.
HISTORY_FILE = os.path.join(os.path.dirname(__file__), 'speedtest_history.csv')

# --- Helper Functions for Persistence ---
def load_history():
    """
    Loads speed test history from the HISTORY_FILE.
    Returns a pandas DataFrame. If the file doesn't exist or is invalid,
    returns an empty DataFrame with the correct columns.
    """
    if os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_csv(HISTORY_FILE)
            # Ensure timestamp column is datetime objects
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            st.info(f"Loaded {len(df)} previous test results from disk.")
            return df
        except Exception as e:
            st.warning(f"Could not load speed test history from '{HISTORY_FILE}'. "
                       f"Starting with empty history. Error: {e}")
            return pd.DataFrame(columns=['timestamp', 'download', 'upload', 'ping'])
    else:
        st.info("No existing speed test history found. Starting fresh.")
        return pd.DataFrame(columns=['timestamp', 'download', 'upload', 'ping'])

def save_history(df):
    """
    Saves the current speed test history to the HISTORY_FILE.
    """
    try:
        df.to_csv(HISTORY_FILE, index=False)
        st.success("Speed test history saved to disk.")
    except Exception as e:
        st.error(f"Failed to save speed test history to '{HISTORY_FILE}'. Error: {e}")
        print(f"Error saving history: {e}")


# --- Page Setup ---
st.set_page_config(
    page_title="Internet Speed Dashboard",
    page_icon="⚡",
    layout="wide"
)

# --- Speed Test Function ---
def run_speed_test():
    """
    Runs a speed test using the imported speedtest.py module.
    Returns a dictionary with ping, download, and upload speeds.
    Returns None on failure.
    """
    try:
        # The spinner in the main app will provide user feedback
        s = speedtest.Speedtest()
        s.get_best_server()
        s.download()
        s.upload()
        
        results = s.results.dict()
        
        # Convert speeds from bits/s to Mbit/s
        ping = results['ping']
        download_speed = results['download'] / 1_000_000
        upload_speed = results['upload'] / 1_000_000
        
        return {
            "ping": ping,
            "download": download_speed,
            "upload": upload_speed,
            "timestamp": datetime.now()
        }
    except Exception as e:
        st.error(f"An error occurred during the speed test: {e}")
        print(f"Error during speed test: {e}")
        return None

# --- Main Application UI ---
st.title("⚡ Real-Time Internet Speed Dashboard")
st.markdown(f"This dashboard automatically tests your internet speed every **{TEST_INTERVAL_MINUTES} minutes**.")

# Initialize session state with loaded history
if 'data' not in st.session_state:
    st.session_state.data = load_history()
if 'last_run' not in st.session_state:
    # If data was loaded, set last_run to the timestamp of the latest entry
    if not st.session_state.data.empty:
        st.session_state.last_run = st.session_state.data['timestamp'].max()
    else:
        # Set to a past time to ensure the first run happens immediately
        st.session_state.last_run = datetime.min

# --- UI Layout ---
# Placeholders for metrics and button
top_cols = st.columns([3, 1])
with top_cols[1]:
    if st.button("Run Test Now"):
        # By setting last_run to the past, we force a new test on the next rerun
        st.session_state.last_run = datetime.min
        st.rerun()

# --- Main Logic ---
# Check if it's time for a new test
time_since_last_run = (datetime.now() - st.session_state.last_run)
is_first_run = st.session_state.last_run == datetime.min

if time_since_last_run.total_seconds() > TEST_INTERVAL_SECONDS or is_first_run:
    with st.spinner("Running speed test... this may take a minute."):
        new_data = run_speed_test()
    
    if new_data:
        new_df_row = pd.DataFrame([new_data])
        # FIX: Check if the main dataframe is empty to avoid the FutureWarning
        if st.session_state.data.empty:
            st.session_state.data = new_df_row
        else:
            st.session_state.data = pd.concat([st.session_state.data, new_df_row], ignore_index=True)
        
        # Save history after each successful test
        save_history(st.session_state.data)
    
    # Update the last run time regardless of success to avoid constant retries on failure
    st.session_state.last_run = datetime.now()


# Display the time of the last test and the next scheduled test
last_run_str = st.session_state.last_run.strftime('%H:%M:%S') if st.session_state.last_run != datetime.min else "Never"
next_run_time = st.session_state.last_run + timedelta(minutes=TEST_INTERVAL_MINUTES)
next_run_str = next_run_time.strftime('%H:%M:%S')

st.info(f"Last test run at: **{last_run_str}**. Next scheduled test at: **{next_run_str}**.")
st.markdown("---")

# --- Display Results ---
df = st.session_state.data

if not df.empty:
    # Get the latest results
    latest = df.iloc[-1]
    
    # Update metric cards
    cols = st.columns(3)
    cols[0].metric(label="Download Speed (Mbps)", value=f"{latest['download']:.2f}")
    cols[1].metric(label="Upload Speed (Mbps)", value=f"{latest['upload']:.2f}")
    cols[2].metric(label="Ping (ms)", value=f"{latest['ping']:.2f}")
    
    st.markdown("### Performance Over Time")
    
    # Create a DataFrame for charting with a proper index
    chart_df = df.set_index('timestamp')
    
    # Display charts
    st.markdown("#### Download and Upload Speed (Mbps)")
    st.line_chart(chart_df[['download', 'upload']])
    
    st.markdown("#### Ping (ms)")
    st.line_chart(chart_df[['ping']])

    st.markdown("### Test History")
    # Update data table, showing newest first
    st.dataframe(df.sort_values(by='timestamp', ascending=False).style.format({
        "download": "{:.2f}",
        "upload": "{:.2f}",
        "ping": "{:.2f}",
        "timestamp": "{:%Y-%m-%d %H:%M:%S}"
    }))
else:
    st.warning("No speed test data yet. The first test will run shortly.")


# --- Auto-refresh logic ---
# This will cause the script to rerun every 60 seconds,
# allowing the time check logic to trigger a new test when needed.
time.sleep(60)
st.rerun()
