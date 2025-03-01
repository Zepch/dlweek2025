# run_enhanced_system.py
import os
import subprocess
import sys

def check_and_install_dependencies():
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'torch', 'scikit-learn',
        'seaborn', 'yfinance', 'joblib'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing required package: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    print("========== AI Trading System ==========")
    print("1. Checking dependencies...")
    check_and_install_dependencies()
    
    print("\n2. Applying fixes...")
    if os.path.exists('fix_rl_error.py'):
        subprocess.call([sys.executable, 'fix_rl_error.py'])
    else:
        print("Fix script not found. Creating it...")
        # Here you'd create the file with the content from above
    
    print("\n3. Adding enhanced RL features...")
    if os.path.exists('enhanced_rl.py'):
        subprocess.call([sys.executable, 'enhanced_rl.py'])
    
    print("\n4. Running main trading system...")
    subprocess.call([sys.executable, 'main.py'])
    
    print("\n========== System Run Complete ==========")

if __name__ == "__main__":
    main()