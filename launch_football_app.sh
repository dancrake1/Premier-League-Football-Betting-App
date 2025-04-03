#!/bin/bash

# Configuration
PROJECT_DIR="/Users/danielcrake/Desktop/Football Betting 2025"
VENV_PATH="$PROJECT_DIR/fb_bet_venv"
APP_FILE="streamlit-dashboard.py"
LOG_FILE="$PROJECT_DIR/logs/streamlit-runner.log"
PID_FILE="/tmp/streamlit-dashboard.pid"
PORT=8501
MAX_RETRIES=3

# Create log directory if it doesn't exist
mkdir -p "$(dirname "$LOG_FILE")"

# Log function
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Clean up function - kill any existing Streamlit processes
cleanup() {
  log "Shutting down Streamlit app..."
  
  # Kill the main Streamlit process if we have its PID
  if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null; then
      log "Killing Streamlit process with PID: $PID"
      kill "$PID" 2>/dev/null
      sleep 1
      # Force kill if still running
      if ps -p "$PID" > /dev/null; then
        log "Force killing process: $PID"
        kill -9 "$PID" 2>/dev/null
      fi
    fi
    rm -f "$PID_FILE"
  fi
  
  # Find and kill any lingering Streamlit processes
  log "Checking for other Streamlit processes..."
  for PROC in $(pgrep -f "streamlit run $APP_FILE"); do
    log "Killing additional Streamlit process: $PROC"
    kill "$PROC" 2>/dev/null
    sleep 1
    if ps -p "$PROC" > /dev/null; then
      kill -9 "$PROC" 2>/dev/null
    fi
  done

  # Also kill any Python processes that might be related to Streamlit
  for PROC in $(pgrep -f "python.*streamlit"); do
    log "Killing related Python process: $PROC"
    kill "$PROC" 2>/dev/null
    sleep 1
    if ps -p "$PROC" > /dev/null; then
      kill -9 "$PROC" 2>/dev/null
    fi
  done

  log "Cleanup complete"
  exit 0
}

# Set up trap to ensure cleanup on exit
trap cleanup INT TERM EXIT

# Check if port is already in use and free it
check_and_free_port() {
  if command -v lsof >/dev/null && lsof -i :$PORT > /dev/null; then
    log "Port $PORT is already in use. Attempting to free it..."
    for PID in $(lsof -t -i :$PORT); do
      log "Killing process using port $PORT: $PID"
      kill "$PID" 2>/dev/null
      sleep 1
      if ps -p "$PID" > /dev/null; then
        kill -9 "$PID" 2>/dev/null
      fi
    done
    sleep 2
    
    # Verify port is free
    if command -v lsof >/dev/null && lsof -i :$PORT > /dev/null; then
      log "Failed to free port $PORT after cleanup"
      return 1
    fi
  fi
  return 0
}

# Clean up previous state files that might cause issues
clean_streamlit_state() {
  log "Cleaning Streamlit state files..."
  # Remove any Streamlit cache files that might be causing issues
  CACHE_DIR="$HOME/.streamlit"
  if [ -d "$CACHE_DIR" ]; then
    find "$CACHE_DIR" -type f -name "*.ldb" -delete
    find "$CACHE_DIR" -type f -name "*.log" -delete
    find "$CACHE_DIR" -type f -name "LOCK" -delete
  fi
}

# Kill any existing Streamlit processes before starting
log "Checking for existing Streamlit processes..."
for PROC in $(pgrep -f "streamlit run $APP_FILE"); do
  log "Killing existing Streamlit process: $PROC"
  kill "$PROC" 2>/dev/null
  sleep 1
  if ps -p "$PROC" > /dev/null; then
    kill -9 "$PROC" 2>/dev/null
  fi
done

# Change to project directory
log "Changing to project directory: $PROJECT_DIR"
cd "$PROJECT_DIR" || { log "Failed to change directory"; exit 1; }

# Activate virtual environment
log "Activating virtual environment"
if [ -f "$VENV_PATH/bin/activate" ]; then
  source "$VENV_PATH/bin/activate" || { log "Failed to activate virtual environment"; exit 1; }
else
  log "Virtual environment not found at $VENV_PATH"
  exit 1
fi

# Clean Streamlit state files
clean_streamlit_state

# Check and free port
check_and_free_port || { log "Failed to free port $PORT"; exit 1; }

# Give the system a moment to release resources
sleep 2

# Start Streamlit app with retry mechanism
log "Starting Streamlit app: $APP_FILE"

retry_count=0
while [ $retry_count -lt $MAX_RETRIES ]; do
  streamlit run "$APP_FILE" --server.port=$PORT > "$LOG_FILE" 2>&1 &
  
  # Save PID for later cleanup
  APP_PID=$!
  echo $APP_PID > "$PID_FILE"
  log "Streamlit started with PID: $APP_PID"
  
  # Wait briefly to see if it crashes immediately
  sleep 5
  
  # Check if process is still running
  if ps -p "$APP_PID" > /dev/null; then
    # Wait for the service to be available
    for i in {1..10}; do
      if curl -s http://localhost:$PORT > /dev/null; then
        log "Streamlit app is responding on port $PORT"
        break
      fi
      sleep 1
    done
    break
  else
    log "Streamlit crashed on startup, attempt $((retry_count+1))/$MAX_RETRIES"
    retry_count=$((retry_count+1))
    sleep 2
  fi
done

if [ $retry_count -ge $MAX_RETRIES ]; then
  log "Failed to start Streamlit after $MAX_RETRIES attempts"
  exit 1
fi

# Open browser after confirmation service is running
log "Opening browser to http://localhost:$PORT"
open "http://localhost:$PORT"

# Wait for the Streamlit process
wait $APP_PID

# If we get here, the process has exited on its own
log "Streamlit process exited naturally"