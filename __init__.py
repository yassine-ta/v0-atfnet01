# Main package initialization
"""
ATFNet Python Implementation
"""
# Import main components for easier access
from . import core
from . import strategies
from . import data
from . import models
from . import mt5_connector
from . import utils
from . import app
from . import backtest
from . import config

__version__ = '0.1.0'

__all__ = [
    'core',
    'strategies',
    'data',
    'models',
    'mt5_connector',
    'utils',
    'app',
    'backtest',
    'config'
]
