#!/usr/bin/env python
"""
Run script for the YAICON Dashboard
"""
import os
import sys
from app import app

if __name__ == "__main__":
    # Add parent directory to path for imports
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5002)
