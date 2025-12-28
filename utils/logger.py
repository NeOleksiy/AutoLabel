import logging
from pathlib import Path


def setup_logger(name: str, log_file: str, level=logging.INFO):
    """Setup logger with both console and file handlers.
    
    Works in both terminal and Jupyter notebook environments.
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level (default: logging.INFO)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Ensure log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console/Stream handler (works in both terminal and Jupyter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    return logger


__all__ = ["setup_logger"]