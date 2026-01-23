import sys
import os

sys.path.append(os.getcwd())

print("Importing src.api...")
try:
    import src.api
    print("Import successful!")
except Exception as e:
    print(f"Import failed: {e}")
