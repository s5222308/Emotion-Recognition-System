"""
ML Service Business Logic
Core service classes and processing functions
"""
from .ml_service import get_emotion_service, process_video, SimpleTwoStageService

__all__ = ['get_emotion_service', 'process_video', 'SimpleTwoStageService']