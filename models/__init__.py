import logging

# Configure module-level logging to ensure it works consistently across all module imports
def setup_logging():
    """Set up logging configuration for the models module"""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Only add handler if none exists to avoid duplicate logs
    if not root_logger.handlers:
        # Add a handler that writes to stderr and flushes immediately
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

# Set up logging when the module is imported
setup_logging()
