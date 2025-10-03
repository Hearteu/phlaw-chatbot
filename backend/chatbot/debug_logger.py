# chatbot/debug_logger.py
import logging
from contextlib import contextmanager
from typing import List


class DebugCapture:
    """Context manager to capture debug log messages"""
    
    def __init__(self, max_messages: int = 50):
        self.max_messages = max_messages
        self.messages: List[str] = []
        self.handler = None
        self.logger = logging.getLogger('chatbot')
        
    def __enter__(self):
        """Start capturing debug messages"""
        # Create a custom handler that captures messages
        self.handler = DebugHandler(self.messages, self.max_messages)
        self.handler.setLevel(logging.DEBUG)
        
        # Add handler to the logger
        self.logger.addHandler(self.handler)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop capturing debug messages"""
        if self.handler:
            self.logger.removeHandler(self.handler)
        return False
    
    def get_debug_messages(self) -> List[str]:
        """Get captured debug messages"""
        return self.messages


class DebugHandler(logging.Handler):
    """Custom logging handler that captures messages to a list"""
    
    def __init__(self, messages: List[str], max_messages: int):
        super().__init__()
        self.messages = messages
        self.max_messages = max_messages
        
    def emit(self, record):
        """Emit a record by appending to the messages list"""
        try:
            msg = self.format(record)
            if len(self.messages) < self.max_messages:
                self.messages.append(msg)
        except Exception:
            self.handleError(record)


@contextmanager
def debug_capture(max_messages: int = 50):
    """
    Context manager to capture debug log messages
    
    Usage:
        with debug_capture(max_messages=50) as debug_logger:
            # Your code here
            messages = debug_logger.get_debug_messages()
    """
    capture = DebugCapture(max_messages=max_messages)
    try:
        yield capture.__enter__()
    finally:
        capture.__exit__(None, None, None)
