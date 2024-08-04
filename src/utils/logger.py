import logging
import os
from src.config import Config

def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """
    Settingup logger.
    :param name: Name of the logger
    :param log_file: File to log messages to
    :param level: Logging level
    :return: Configured logger
    """
    # Ensuring log directory exists
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Defining log format
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')

    # Creating file handler to log to file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Creating stream handler to log to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # To get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Avoiding duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger
  
if __name__ == "__main__":
    # Ensuring log directory exists
    if not os.path.exists(Config.LOG_DIR):
        os.makedirs(Config.LOG_DIR)

    # To setup logger
    logger = setup_logger('example_logger', os.path.join(Config.LOG_DIR, 'example.log'))
    logger.info("This is a test log message.")
    logger.error("This is a test error message.")
