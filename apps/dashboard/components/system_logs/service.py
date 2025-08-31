"""
System Logs Service
CRITICAL: This preserves EXACT business logic from monolithic system
All calculations and data structures must match original implementation
"""

from datetime import datetime


class SystemLogsService:
    """Service for System Logs component
    
    CRITICAL: This accesses the SAME global state as the monolithic system
    DO NOT create new data structures - use existing global state
    """
    
    def __init__(self):
        # CRITICAL: Import global state from main app
        # This ensures we use the SAME data that the monolithic system uses
        pass
    
    def get_logs(self, level_filter='ALL', limit=50):
        """Get system logs with filtering
        
        CRITICAL: This MUST produce the EXACT same data structure as the original
        Original function: api_logs() endpoint logic
        """
        # Import global state from core module (using proven working pattern)
        from core import system_logs
        
        # Convert deque to list for processing
        logs = list(system_logs)
        
        # Apply level filter if not 'ALL' (same logic as original)
        if level_filter != 'ALL':
            logs = [log for log in logs if log.get('level') == level_filter]
        
        # Apply limit (most recent logs) 
        if limit and len(logs) > limit:
            logs = logs[-limit:]
        
        # Return in the same format as original
        return logs