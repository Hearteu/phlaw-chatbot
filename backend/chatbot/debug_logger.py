# debug_logger.py — Custom logging for debug output capture
import logging
import threading
from typing import List, Optional
from contextlib import contextmanager


class DebugBufferHandler(logging.Handler):
    """Custom logging handler that buffers debug messages"""
    
    def __init__(self, max_messages: int = 50):
        super().__init__()
        self.max_messages = max_messages
        self.debug_messages: List[str] = []
        self.lock = threading.Lock()
        
    def emit(self, record):
        """Emit a log record to the buffer"""
        try:
            # Only capture debug messages with specific markers
            message = self.format(record)
            if any(marker in message for marker in ['🔍', '🎯', '⚠️', '📡', '✅', '❌']):
                with self.lock:
                    if len(self.debug_messages) < self.max_messages:
                        self.debug_messages.append(message)
        except Exception:
            # Ignore errors in debug capture
            pass
    
    def get_messages(self) -> List[str]:
        """Get captured debug messages"""
        with self.lock:
            return self.debug_messages.copy()
    
    def clear(self):
        """Clear captured debug messages"""
        with self.lock:
            self.debug_messages.clear()


class DebugFormatter(logging.Formatter):
    """Custom formatter for debug messages"""
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    
    def format(self, record):
        """Format the log record"""
        # Add thread info for debugging
        record.thread_name = threading.current_thread().name
        return super().format(record)


class DebugLogger:
    """Centralized debug logger management"""
    
    def __init__(self, logger_name: str = 'chatbot.debug'):
        self.logger_name = logger_name
        self.logger = logging.getLogger(logger_name)
        self.handler: Optional[DebugBufferHandler] = None
        self.original_level = None
        
    def setup_debug_capture(self, max_messages: int = 50):
        """Setup debug message capture"""
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Create and add our custom handler
        self.handler = DebugBufferHandler(max_messages)
        self.handler.setFormatter(DebugFormatter())
        self.handler.setLevel(logging.DEBUG)
        
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.DEBUG)
        
        # Store original level for restoration
        self.original_level = self.logger.level
        
    def get_debug_messages(self) -> List[str]:
        """Get captured debug messages"""
        if self.handler:
            return self.handler.get_messages()
        return []
    
    def clear_debug_messages(self):
        """Clear captured debug messages"""
        if self.handler:
            self.handler.clear()
    
    def cleanup(self):
        """Cleanup debug capture"""
        if self.handler:
            self.logger.removeHandler(self.handler)
            self.handler = None
        
        if self.original_level is not None:
            self.logger.setLevel(self.original_level)
            self.original_level = None


# Global debug logger instance
_debug_logger = DebugLogger()


def get_debug_logger() -> DebugLogger:
    """Get the global debug logger instance"""
    return _debug_logger


@contextmanager
def debug_capture(max_messages: int = 50):
    """Context manager for capturing debug messages"""
    debug_logger = get_debug_logger()
    
    try:
        # Setup debug capture
        debug_logger.setup_debug_capture(max_messages)
        yield debug_logger
    finally:
        # Cleanup
        debug_logger.cleanup()


def log_debug(message: str, **kwargs):
    """Log a debug message with the debug logger"""
    debug_logger = get_debug_logger()
    debug_logger.logger.debug(message, **kwargs)


def log_info(message: str, **kwargs):
    """Log an info message with the debug logger"""
    debug_logger = get_debug_logger()
    debug_logger.logger.info(message, **kwargs)


def log_warning(message: str, **kwargs):
    """Log a warning message with the debug logger"""
    debug_logger = get_debug_logger()
    debug_logger.logger.warning(message, **kwargs)


def log_error(message: str, **kwargs):
    """Log an error message with the debug logger"""
    debug_logger = get_debug_logger()
    debug_logger.logger.error(message, **kwargs)
