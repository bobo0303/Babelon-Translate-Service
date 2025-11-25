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
        self.headers = [
            'timestamp',
            'meeting_id',
            'audio_uid',
            'audio_duration',
            'file_read_time',
            'file_save_time',
            'silence_padding_time',
            'transcription_time',
            'post_processing_time',
            'translation_time',
            'get_results_time',
            'total_inference_time',
            'total_request_time',
            'rtf',
            'translate_manager_dispatch_time',
            'translate_manager_max_llm_time',
            'translate_manager_overhead'
        ]
        
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Initialize CSV file with headers if it doesn't exist
        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writeheader()
            logger.info(f" | Performance log initialized: {log_file} | ")
    
    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Log performance metrics to CSV file (thread-safe).
        Only includes columns that have non-None values in the metrics.
        
        Args:
            metrics: Dictionary containing performance metrics
        """
        try:
            # Add timestamp if not provided
            if 'timestamp' not in metrics:
                metrics['timestamp'] = datetime.now().isoformat()
            
            # Determine which headers to use based on non-None values
            active_headers = [header for header in self.headers if metrics.get(header) is not None]
            
            # Build row with only active headers
            row = {header: metrics[header] for header in active_headers}
            
            # Write to CSV with thread safety
            with self.lock:
                # Check if file exists and is empty to write headers
                file_exists = os.path.exists(self.log_file)
                file_is_empty = not file_exists or os.path.getsize(self.log_file) == 0
                
                with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=active_headers, extrasaction='ignore')
                    if file_is_empty:
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
