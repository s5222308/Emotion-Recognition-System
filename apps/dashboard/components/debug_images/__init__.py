"""
Debug Images Component
CRITICAL: This follows the EXACT same pattern as other successful components
"""

from .routes import debug_images_bp, init_debug_images
from .service import DebugImagesService

__all__ = ['debug_images_bp', 'init_debug_images', 'DebugImagesService']