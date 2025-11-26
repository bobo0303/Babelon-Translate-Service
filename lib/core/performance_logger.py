"""
Performance metrics logging utility for audio translation service.
Logs timing data to CSV for performance analysis.
"""

import csv
import os
import threading
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class PerformanceLogger:
    """Thread-safe CSV logger for performance metrics"""
    
    def __init__(self, log_file: str = "logs/performance_metrics.csv"):
        """
        Initialize the performance logger.
        
        Args:
            log_file: Path to the CSV log file
        """
        self.log_file = log_file
        self.lock = threading.Lock()
        self.headers = None  # Will be set on first write based on actual metrics
        self.headers_written = False
        
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Log performance metrics to CSV file (thread-safe).
        Only includes columns that have non-None values.
        Headers are determined by the first write and remain fixed.
        
        Args:
            metrics: Dictionary containing performance metrics
        """
        try:
            # Add timestamp if not provided
            if 'timestamp' not in metrics:
                metrics['timestamp'] = datetime.now().isoformat()
            
            # Write to CSV with thread safety
            with self.lock:
                # Determine headers on first write based on non-None values
                if self.headers is None:
                    self.headers = [key for key, value in metrics.items() if value is not None]
                    logger.info(f" | Performance log headers initialized: {self.headers} | ")
                
                # Build row with current headers (fill missing with empty string)
                row = {header: metrics.get(header, '') for header in self.headers}
                
                # Check if we need to write headers
                file_exists = os.path.exists(self.log_file)
                write_header = not file_exists or os.path.getsize(self.log_file) == 0
                
                with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.headers, extrasaction='ignore')
                    if write_header:
                        writer.writeheader()
                    writer.writerow(row)
            
            logger.debug(f" | Performance metrics logged for audio_uid: {metrics.get('audio_uid', 'unknown')} | ")
        
        except Exception as e:
            logger.error(f" | Failed to log performance metrics: {e} | ")

# Global performance logger instance
_performance_logger: Optional[PerformanceLogger] = None

def get_performance_logger() -> PerformanceLogger:
    """Get or create the global performance logger instance."""
    global _performance_logger
    if _performance_logger is None:
        _performance_logger = PerformanceLogger()
    return _performance_logger
