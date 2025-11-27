"""
Logging utilities for erucakra with timing and error tracking.
"""

import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

# Global timing logger instance
_timing_logger: Optional["TimingLogger"] = None


class TimingLogger:
    """Tracks timing of execution steps."""
    
    def __init__(self):
        self.steps: List[Dict[str, Any]] = []
        self.current_step: Optional[Dict[str, Any]] = None
        self.start_time: float = time.time()
    
    def start_step(self, name: str) -> None:
        """Start timing a new step."""
        self.current_step = {
            "name": name,
            "start": time.time(),
            "end": None,
            "duration": None,
            "success": None,
        }
    
    def end_step(self, success: bool = True) -> float:
        """End current step and return duration."""
        if self.current_step is None:
            return 0.0
        
        self.current_step["end"] = time.time()
        self.current_step["duration"] = self.current_step["end"] - self.current_step["start"]
        self.current_step["success"] = success
        self.steps.append(self.current_step)
        
        duration = self.current_step["duration"]
        self.current_step = None
        return duration
    
    def get_summary(self) -> str:
        """Get formatted timing summary."""
        total = time.time() - self.start_time
        
        lines = [
            "",
            "═" * 60,
            "  TIMING SUMMARY",
            "─" * 60,
        ]
        
        for step in self.steps:
            status = "✓" if step["success"] else "✗"
            duration = step["duration"] or 0
            lines.append(f"  {status} {step['name']}: {duration:.2f}s")
        
        lines.extend([
            "─" * 60,
            f"  Total: {total:.2f}s",
            "═" * 60,
        ])
        
        return "\n".join(lines)


def get_timing_logger() -> Optional[TimingLogger]:
    """Get global timing logger instance."""
    return _timing_logger


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[str] = None,
    experiment_name: Optional[str] = None,
    format_style: str = "detailed",
    always_save: bool = True,
    include_timestamp: bool = False,
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Parameters
    ----------
    level : str
        Log level (DEBUG, INFO, WARNING, ERROR).
    log_dir : str, optional
        Directory for log files.
    experiment_name : str, optional
        Name for log file.
    format_style : str
        Format style: 'detailed', 'simple', 'minimal'.
    always_save : bool
        Always save to file even if no errors.
    include_timestamp : bool
        Include timestamp in log filename.
    
    Returns
    -------
    logging.Logger
        Configured logger.
    """
    global _timing_logger
    _timing_logger = TimingLogger()
    
    # Get root logger for erucakra
    logger = logging.getLogger("erucakra")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Format strings
    if format_style == "detailed":
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
    elif format_style == "simple":
        fmt = "%(levelname)s: %(message)s"
        datefmt = None
    else:  # minimal
        fmt = "%(message)s"
        datefmt = None
    
    formatter = logging.Formatter(fmt, datefmt=datefmt)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_dir and always_save:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{experiment_name}_{timestamp}.log" if experiment_name else f"erucakra_{timestamp}.log"
        else:
            filename = f"{experiment_name}.log" if experiment_name else "erucakra.log"
        
        file_handler = logging.FileHandler(log_path / filename)
        file_handler.setLevel(logging.DEBUG)  # Capture everything in file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def start_step(name: str) -> None:
    """Start timing a step (global convenience function)."""
    logger = logging.getLogger("erucakra")
    logger.info(f"Starting: {name}")
    
    if _timing_logger:
        _timing_logger.start_step(name)


def end_step(success: bool = True) -> float:
    """End timing current step (global convenience function)."""
    logger = logging.getLogger("erucakra")
    
    duration = 0.0
    if _timing_logger:
        duration = _timing_logger.end_step(success)
    
    status = "completed" if success else "FAILED"
    logger.info(f"Step {status} in {duration:.2f}s")
    
    return duration


def log_error(error: Exception, context: str = "") -> None:
    """Log an error with full traceback."""
    logger = logging.getLogger("erucakra")
    
    error_msg = f"ERROR in {context}: {type(error).__name__}: {error}"
    logger.error(error_msg)
    logger.debug(traceback.format_exc())


def log_calculation_issue(
    issue_type: str,
    description: str,
    details: Dict[str, Any],
) -> None:
    """Log a calculation issue (NaN, Inf, extreme values, etc.)."""
    logger = logging.getLogger("erucakra")
    
    logger.warning(f"Calculation issue [{issue_type}]: {description}")
    if details:
        logger.debug(f"  Details: {details}")
