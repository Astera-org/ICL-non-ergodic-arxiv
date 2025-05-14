import logging
import logging.handlers
from omegaconf import DictConfig, OmegaConf # Ensure OmegaConf is imported
import sys
import os
from pathlib import Path

LOG_DIR = "logs"
# Ensure log directory exists when module is loaded
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

def setup_logging(cfg: DictConfig = None):
    log_level_str = "INFO"  # Default
    log_file_path_str = "logs/app_default.log" # Default log file

    if cfg:
        # Safely get logging level
        resolved_level = OmegaConf.select(cfg, "logging.level", default=None)
        if isinstance(resolved_level, str):
            log_level_str = resolved_level
        elif resolved_level is not None:
            # If it's defined but not a string, log a warning and use default
            logging.warning(f"logging.level in config is not a string ({type(resolved_level)}), using default '{log_level_str}'.")
        
        # Safely get log file path
        resolved_log_file = OmegaConf.select(cfg, "logging.log_file", default=None)
        if isinstance(resolved_log_file, str):
            log_file_path_str = resolved_log_file
        elif resolved_log_file is not None:
            logging.warning(f"logging.log_file in config is not a string ({type(resolved_log_file)}), using default '{log_file_path_str}'.")

    log_file = Path(log_file_path_str)
    # If log_file is relative, it will be relative to CWD where script is run.
    # For Hydra, Hydra typically changes CWD to the output directory.
    # If running script directly, CWD is project root usually.
    # Ensure parent dir for log file exists
    log_file.parent.mkdir(parents=True, exist_ok=True) 

    # The debug print that helped identify the issue
    # print(f"[DEBUG logging_config] Effective log_level_str: {log_level_str}, type: {type(log_level_str)}, Log file: {log_file}")
    
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)

    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Clear existing handlers to avoid duplicate logs if setup_logging is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        fmt="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (Rotating)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=5*1024*1024,  # 5 MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logging.info(f"Logging configured. Level: {log_level_str}, File: {log_file}")


def get_logger(name):
    """Gets a logger instance with the given name."""
    return logging.getLogger(name)

# Initial setup when module is imported, using defaults or simple config if available.
# This ensures that loggers obtained by `get_logger` are functional even before
# a more specific `setup_logging(cfg)` is called by an application entry point.
# However, this can be tricky if Hydra also manages CWD for log files.
# For now, let's call it with no config, so it uses hardcoded defaults.
# The main app (e.g. main_hydra_app.py) should call setup_logging(cfg) again
# to apply Hydra config if needed.
setup_logging() # Default setup on import

if __name__ == "__main__":
    # Example usage:
    setup_logging(log_level_str="DEBUG", log_file="main_app.log")
    
    logger_main = get_logger(__name__)
    logger_module_A = get_logger("module_A")
    logger_module_B_feature_X = get_logger("module_B.feature_X")

    logger_main.debug("This is a debug message from main.")
    logger_main.info("This is an info message from main.")
    logger_main.warning("This is a warning from main.")
    logger_main.error("This is an error from main.")
    logger_main.critical("This is a critical message from main.")

    logger_module_A.info("Module A logging an info event.")
    try:
        x = 1 / 0
    except ZeroDivisionError:
        logger_module_A.error("Error in Module A calculation!", exc_info=True) # exc_info=True logs stack trace

    logger_module_B_feature_X.debug("Debugging feature X in Module B.")
    logger_module_B_feature_X.info({"user_id": 123, "action": "button_click", "details": "User clicked the submit button."}) 