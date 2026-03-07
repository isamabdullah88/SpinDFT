import logging
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to terminal output based on log level."""
    
    # ANSI escape codes for colors
    GREY = "\x1b[38;20m"
    GREEN = "\x1b[32;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"
    
    # The exact format you want your logs to follow
    FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"

    # Map each log level to a specific color
    FORMATS = {
        logging.DEBUG: GREY + FORMAT + RESET,
        logging.INFO: GREEN + FORMAT + RESET,
        logging.WARNING: YELLOW + FORMAT + RESET,
        logging.ERROR: RED + FORMAT + RESET,
        logging.CRITICAL: BOLD_RED + FORMAT + RESET
    }

    def format(self, record):
        logfmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(logfmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


def getlogger(name="SpinDFT"):
    """
    Sets up a dual-output logger:
    - Prints COLORED INFO, WARNING, and ERROR to the terminal.
    - Saves PLAIN TEXT DEBUG, INFO, WARNING, and ERROR to a log file.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.DEBUG) 

        # 1. Console Handler (What you see in the terminal - WITH COLORS)
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO) 
        console.setFormatter(ColoredFormatter()) # Apply the color injection!

        # 2. File Handler (What gets saved to the disk - PLAIN TEXT)
        plain_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        logdir = "SpinDFTLogs"
        os.makedirs(logdir, exist_ok=True)

        utcnow = datetime.now(ZoneInfo("UTC"))
        localdt = utcnow.astimezone(ZoneInfo("Asia/Karachi"))
        logfile=f"SpinDFT_{localdt.strftime('%Y%m%d_%H%M%S')}.log"
        file = logging.FileHandler(os.path.join(logdir, logfile))
        file.setLevel(logging.DEBUG) 
        file.setFormatter(plain_formatter) # Keep the file clean!

        logger.addHandler(console)
        logger.addHandler(file)

    return logger