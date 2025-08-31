"""
System Overview Component
Displays high-level system metrics and health indicators
"""

from .routes import system_overview_bp, init_system_overview
from .service import SystemOverviewService

__all__ = ['system_overview_bp', 'init_system_overview', 'SystemOverviewService']