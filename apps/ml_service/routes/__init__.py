"""
ML Service Routes
Modular route definitions for the ML inference service
"""
from .emotion_routes import emotion_bp
from .health_routes import health_bp

__all__ = ['emotion_bp', 'health_bp']