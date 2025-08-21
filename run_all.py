"""
Master script to run the entire churn prediction pipeline
"""

import subprocess
import sys
import os

def run_script(script_name):
    """Run a Python script and handle errors"""
    
    print(f"\n{'='*50}")
    print(f"Running: {script_name}")
    print('='*50)
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, 
                              text=True, 
                              cwd=os.getcwd())
        
        if result.returncode == 0:
            print(f"✅ {script_name} completed successfully!")
            return True
        else:
            print(f"❌ {script_name} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return False

def main():
    """Run the complete pipeline"""
    
    print("=== CHURN PREDICTION PIPELINE ===")
    print("Running complete ML pipeline...")
    
    scripts = [
        "C:\\Users\\ambie\\OneDrive\\Desktop\\tema_proiect_predictie\\scripts\\00_setup_check.py",
        "C:\\Users\\ambie\OneDrive\Desktop\\tema_proiect_predictie\\scripts\\generate_sample_data.py", 
        "C:\\Users\\ambie\OneDrive\Desktop\\tema_proiect_predictie\\scripts\\01_eda_analysis.py",
        "C:\\Users\\ambie\OneDrive\Desktop\\tema_proiect_predictie\\scripts\\02_feature_engineering.py",
        "C:\\Users\\ambie\OneDrive\Desktop\\tema_proiect_predictie\\scripts\\03_model_training.py"
    ]
    
    for script in scripts:
        if not os.path.exists(script):
            print(f"❌ Script not found: {script}")
            continue
            
        success = run_script(script)
        
        if not success:
            print(f"\n❌ Pipeline failed at: {script}")
            print("Please check the error messages above.")
            return False
    
    print(f"\n{'='*50}")
    print("✅ PIPELINE COMPLETE!")
    print('='*50)
    print("\nGenerated files:")
    print("- data/raw/transactions.csv")
    print("- data/processed/features.csv") 
    print("- reports/EDA_REPORT.md")
    print("- reports/FEATURES.md")
    print("- artifacts/metrics.json")
    print("- artifacts/model/churn_model.h5")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
