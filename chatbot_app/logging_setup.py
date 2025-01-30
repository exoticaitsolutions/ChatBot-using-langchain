import logging
import os
from datetime import datetime


def setup_logging():
    # Create the logs_detail directory if it doesn't exist
    log_dir = "logs_detail"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Generate a log file name based on the current date
    log_file = os.path.join(log_dir, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")
    
    # Set up logging configuration
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,  # Set the logging level to INFO
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode='a'  # Append to the existing log file
    )
    logger = logging.getLogger()
    return logger