import sys
import os
import time

sys.path.append(os.getcwd())

print("Starting verbose import...")

def log(msg):
    print(f"[{time.strftime('%X')}] {msg}")

try:
    log("Importing fastapi...")
    import fastapi
    log("Importing pandas...")
    import pandas
    log("Importing torch...")
    import torch
    log("Importing nba_api...")
    import nba_api.stats.endpoints.commonteamroster
    log("Importing src.data_fetch...")
    import src.data_fetch
    log("Importing src.feature_engineer...")
    import src.feature_engineer
    log("Importing src.train_models...")
    import src.train_models
    log("Importing src.daily_predict...")
    import src.daily_predict
    log("Importing src.api...")
    import src.api
    log("SUCCESS")
except Exception as e:
    log(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
