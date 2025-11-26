"""Logging configuration for erucakra."""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str | Path] = None,
    log_dir: Optional[str | Path] = None,
    include_timestamp: bool = True,
    format_style: str = "detailed",
) -> logging.Logger:
    """
    Configure logging for erucakra.
    
    Parameters
    ----------
    level : str, optional
        Logging level. Default is "INFO".
    log_file : str or Path, optional
        Specific log file path.
    log_dir : str or Path, optional
        Directory for log files. If provided, creates timestamped log file.
    include_timestamp : bool, optional
        Include timestamp in log filename. Default is True.
    format_style : str, optional
        Format style: "detailed", "simple", or "minimal". Default is "detailed".
    
    Returns
    -------
    logging.Logger
        Configured root logger.
    """
    # Format strings
    formats = {
        "detailed": "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
        "simple": "%(asctime)s | %(levelname)-8s | %(message)s",
        "minimal": "%(levelname)s: %(message)s",
    }
    date_format = "%Y-%m-%d %H:%M:%S"
    
    log_format = formats.get(format_style, formats["detailed"])
    
    # Get root logger for erucakra
    logger = logging.getLogger("erucakra")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    logger.addHandler(console_handler)
    
    # File handler
    if log_file or log_dir:
        if log_file:
            log_path = Path(log_file)
        else:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            if include_timestamp:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_path = log_dir / f"erucakra_{timestamp}.log"
            else:
                log_path = log_dir / "erucakra.log"
        
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        logger.addHandler(file_handler)
        
        logger.info(f"Log file: {log_path}")
    
    return logger


def get_logger(name: str = "erucakra") -> logging.Logger:
    """
    Get a logger instance.
    
    Parameters
    ----------
    name : str, optional
        Logger name. Default is "erucakra".
    
    Returns
    -------
    logging.Logger
        Logger instance.
    """
    return logging.getLogger(name)
