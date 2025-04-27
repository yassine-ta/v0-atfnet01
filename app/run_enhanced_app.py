#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run the enhanced ATFNet application.
"""

import sys
import os

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the enhanced application
from enhanced_atfnet_app import main

if __name__ == "__main__":
    main()
