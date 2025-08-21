"""
Setup and dependency check script
Run this first to verify all dependencies are installed
"""

import sys
import subprocess
import os

def check_and_install_dependencies():
    """Check and install required dependencies"""
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'scikit-learn': 'sklearn',
        'tensorflow': 'tensorflow'
    }

    missing_packages = []

    for pip_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✅ {pip_name} - OK")
        except ImportError:
            print(f"❌ {pip_name} - MISSING")
            missing_packages.append(pip_name)

    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    print("\n✅ All dependencies are installed!")
    return True


def create_directories():
    """Create necessary directories"""
    
    directories = [
        'data/raw',
        'data/processed', 
        'reports',
        'artifacts/model'
    ]
    
    print("\nCreating directories...")
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created: {directory}")


def test_imports():
    """Test all imports used in the scripts"""
    
    print("\nTesting imports...")
    
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from datetime import datetime, timedelta
        import sklearn
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        from sklearn.utils.class_weight import compute_class_weight
        import tensorflow as tf
        import json
        import os
        import time
        
        print("✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


if __name__ == "__main__":
    print("=== SETUP CHECK ===\n")
    
    # Check dependencies
    deps_ok = check_and_install_dependencies()
    
    if not deps_ok:
        print("\n❌ Please install missing dependencies first!")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Test imports
    imports_ok = test_imports()
    
    if imports_ok:
        print("\n✅ Setup complete! You can now run the other scripts.")
    else:
        print("\n❌ Setup failed. Please check error messages above.")
