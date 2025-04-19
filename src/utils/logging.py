import logging
import sys
from logging import Logger, StreamHandler, Formatter

# Define a default log format
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# Define a default date format for the timestamp
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

def setup_logging(level: int = logging.INFO, log_format: str = DEFAULT_LOG_FORMAT, date_format: str = DEFAULT_DATE_FORMAT) -> Logger:
    """
    Sets up the basic logging configuration for the application.

    Configures the root logger to output messages to the console with a
    specified level and format. Prevents adding duplicate handlers if called
    multiple times.

    Args:
        level: The minimum logging level to capture (e.g., logging.INFO, logging.DEBUG).
        log_format: The format string for log messages.
        date_format: The format string for the timestamp in log messages.

    Returns:
        The configured root logger.
    """
    # Get the root logger
    root_logger = logging.getLogger()

    # Set the logging level for the root logger
    root_logger.setLevel(level)

    # Prevent adding duplicate handlers if setup_logging is called more than once
    if not root_logger.handlers:
        # Create a console handler to output logs to stdout
        console_handler = StreamHandler(sys.stdout)

        # Create a formatter for the log messages
        formatter = Formatter(fmt=log_format, datefmt=date_format)

        # Set the formatter for the console handler
        console_handler.setFormatter(formatter)

        # Add the console handler to the root logger
        root_logger.addHandler(console_handler)

        # Optional: Configure other handlers here (e.g., FileHandler for logging to a file)
        # file_handler = FileHandler("app.log")
        # file_handler.setFormatter(formatter)
        # root_logger.addHandler(file_handler)

        logger = logging.getLogger(__name__)
        logger.info("Logging configured.")

    return root_logger

# Example Usage (optional, for testing purposes)
# This block will only run if the script is executed directly
# if __name__ == "__main__":
#     # Call setup_logging to configure logging
#     setup_logging(level=logging.DEBUG)

#     # Get a logger for a specific module (e.g., this one)
#     module_logger = logging.getLogger(__name__)

#     # Log messages at different levels
#     module_logger.debug("This is a debug message.")
#     module_logger.info("This is an info message.")
#     module_logger.warning("This is a warning message.")
#     module_logger.error("This is an error message.")
#     module_logger.critical("This is a critical message.")

#     # Logging from another 'simulated' module
#     another_logger = logging.getLogger("another_module")
#     another_logger.info("This message is from another module.")
