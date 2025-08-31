"""
Debug Images Service  
CRITICAL: This preserves EXACT business logic from monolithic system
All file operations and data structures must match original implementation
"""

import os
import json
from datetime import datetime
from pathlib import Path


class DebugImagesService:
    """Service for Debug Images component
    
    CRITICAL: This accesses the SAME global state as the monolithic system
    DO NOT create new data structures - use existing global state
    """
    
    def __init__(self):
        # CRITICAL: Import global state from core module
        # This ensures we use the SAME data that the monolithic system uses
        pass
    
    def get_debug_images_list(self):
        """Get list of debug images
        
        CRITICAL: This scans the filesystem since the original appears to be incomplete
        The global debug_images list is meant to be populated from filesystem
        """
        from pathlib import Path
        
        # Get debug images directory relative to project root  
        project_root = Path(__file__).parent.parent.parent.parent.parent
        debug_dir = project_root / 'data' / 'debug'
        
        images = []
        
        if debug_dir.exists():
            # Scan for image files and create the expected data structure
            for filepath in sorted(debug_dir.glob('*'), reverse=True):
                if filepath.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
                    # Get file modification time
                    mod_time = datetime.fromtimestamp(filepath.stat().st_mtime)
                    
                    # Extract emotion from filename (try different patterns)
                    filename = filepath.name
                    parts = filename.split('_')
                    
                    # Try to extract emotion - could be different formats
                    emotion = 'unknown'
                    if len(parts) > 1:
                        emotion = parts[1] if parts[1].isalpha() else 'unknown'
                    
                    images.append({
                        'filename': filename,
                        'timestamp': mod_time.isoformat(),
                        'emotion': emotion
                    })
        
        # Also update the global state to keep the count accurate
        from core import debug_images
        debug_images.clear()
        debug_images.extend(images)
        
        return images
    
    def serve_debug_image(self, filename):
        """Get path to debug image file for serving
        
        CRITICAL: This MUST serve from the EXACT same directory as expected
        Original expects images at /debug-images/${img.filename}
        """
        # Use proper configuration instead of hardcoded path
        from pathlib import Path
        
        # Get debug images directory relative to project root
        project_root = Path(__file__).parent.parent.parent.parent.parent
        debug_dir = project_root / 'data' / 'debug'
        
        file_path = debug_dir / filename
        
        if file_path.exists():
            return str(debug_dir), filename
        else:
            return None, None