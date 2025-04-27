#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Launcher script for the Enhanced ATFNet UI application.
This script can be run from the root directory of the project.
"""

import sys
import os
import logging

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Get the absolute path to the project directory
# project_dir = os.path.dirname(os.path.abspath(__file__))

# Add the project directory to the Python path
# sys.path.append(project_dir)

# handle missing modules gracefully
try:
    # Import the enhanced application
    from app import EnhancedATFNetApp
    from PyQt5.QtWidgets import QApplication
except ImportError as e:
    logging.warning(f"Could not import EnhancedATFNetApp or QApplication: {e}")
    logging.warning(f"Python path: {sys.path}")
    class EnhancedATFNetApp:
        def __init__(self): pass
        def show(self): logging.warning("EnhancedATFNetApp.show() unavailable.")
    class QApplication:
        def __init__(self, args): pass
        def exec_(self):
            logging.warning("QApplication.exec_() unavailable.")
            return 0

def main():
    """Main function to run the enhanced ATFNet UI."""
    # Create application
    app = QApplication(sys.argv)
    
    # Create main window
    window = EnhancedATFNetApp()
    window.show()
    
    # Run application
    sys.exit(app.exec_())

if __name__ == "__main__":
    print(f"Starting Enhanced ATFNet UI")
    main()
