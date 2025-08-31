"""
System Logs Component
"""
from flask import Blueprint
from .routes import system_logs_bp
from .service import SystemLogsService

def init_system_logs(app):
    """Initialize System Logs component with Flask app"""
    app.register_blueprint(system_logs_bp)
    return SystemLogsService()

__all__ = ['system_logs_bp', 'SystemLogsService', 'init_system_logs']