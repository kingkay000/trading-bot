
import sys
from pathlib import Path

def verify_imports():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    python_version_file = root / ".python-version"
    expected = python_version_file.read_text().strip() if python_version_file.exists() else "unknown"
    current = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"[INFO] Expected Python from .python-version: {expected}")
    print(f"[INFO] Current Python interpreter: {current}")

    modules_to_test = [
        "ccxt", "pandas", "numpy", "dotenv", "yaml", "aiohttp",
        "websockets", "requests", "vectorbt", "plotly", "rich",
        "telegram", "fastapi", "pytest"
    ]
    
    local_modules = [
        "modules.ai_signal_engine", "modules.data_engine", 
        "modules.indicator_engine", "modules.risk_manager"
    ]
    
    print("--- Testing Core Dependencies ---")
    for mod in modules_to_test:
        try:
            __import__(mod)
            print(f"[OK] {mod}")
        except ImportError as e:
            print(f"[FAIL] {mod}: {e}")
            if mod in ("pandas", "fastapi", "pytest"):
                print("       hint: run scripts/setup_test_env.sh")
            
    print("\n--- Testing Local Project Modules ---")
    for mod in local_modules:
        try:
            __import__(mod)
            print(f"[OK] {mod}")
        except ImportError as e:
            print(f"[FAIL] {mod}: {e}")

if __name__ == "__main__":
    verify_imports()
