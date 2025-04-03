import os
import logging
import sys
from pathlib import Path

def find_project_root() -> str:
    """
    Find the project root directory based on the actual project structure.
    The root should be 'Football Betting 2025' directory which contains the logs folder.
    """
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    
    # Keep going up until we find the project root folder
    while current_dir and not current_dir.endswith('Football Betting 2025'):
        parent = os.path.dirname(current_dir)
        if parent == current_dir:  # Reached root without finding target
            break
        current_dir = parent
    
    return current_dir

def setup_logging(app_name: str) -> logging.Logger:
    """
    Set up logging with proper error handling and directory creation
    
    Args:
        app_name: Name of the application (e.g., 'dashboard', 'prediction_system')
        
    Returns:
        logging.Logger: Configured logger instance
    """
    try:
        # Find project root and create logs directory
        project_root = find_project_root()
        logs_dir = os.path.join(project_root, 'logs')
        Path(logs_dir).mkdir(parents=True, exist_ok=True)
        
        # Create log file path
        log_file = os.path.join(logs_dir, f'{app_name}.log')
        
        # Create a custom logger
        logger = logging.getLogger(app_name)
        logger.setLevel(logging.INFO)
        
        # Check if logger already has handlers to avoid duplicates
        if not logger.handlers:
            # Create handlers
            file_handler = logging.FileHandler(log_file)
            stream_handler = logging.StreamHandler(sys.stdout)
            
            # Create formatters and add it to handlers
            log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(log_format)
            stream_handler.setFormatter(log_format)
            
            # Add handlers to the logger
            logger.addHandler(file_handler)
            logger.addHandler(stream_handler)
            
            # Set propagate to False to prevent double logging
            logger.propagate = False
            
            # Log successful setup with absolute paths for debugging
            logger.info(f"Logging initialized for {app_name}. Log file: {log_file}")
            logger.debug(f"Project root: {project_root}")
            logger.debug(f"Logs directory: {logs_dir}")
        
        return logger
        
    except Exception as e:
        # If we can't set up logging, print to stderr
        print(f"Failed to initialize logging: {str(e)}", file=sys.stderr)
        raise

def get_log_path() -> str:
    """Get the path to the logs directory"""
    project_root = find_project_root()
    return os.path.join(project_root, 'logs')