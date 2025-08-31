"""
Annotation Setup Component
Handles Label Studio project setup and configuration
"""
from .routes import annotation_setup_bp, init_annotation_setup
from .service import AnnotationSetupService

__all__ = ['annotation_setup_bp', 'init_annotation_setup', 'AnnotationSetupService']