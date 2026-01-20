"""
Evaluation module for Privacy-Preserving Federated Learning.

This module provides comprehensive metrics computation and tracking.
"""

from .metrics import MetricsComputer
from .metrics_tracker import MetricsTracker

__all__ = ['MetricsComputer', 'MetricsTracker']
