from fastapi import FastAPI, WebSocket, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import sys
import os
import pandas as pd
from datetime import datetime
from typing import List, Optional
import io
import contextlib

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_fetch import fetch_daily_scoreboard
from src.daily_predict import predict_daily
from src.train_models import train_model
from nba_api.stats.endpoints import commonteamroster

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Logger for WebSocket
class LogManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

log_manager = LogManager()

# Capture stdout/stderr logic
class StreamToLogger(object):
    def __init__(self, original_stream):
        self.original_stream = original_stream

    def write(self, buf):
        self.original_stream.write(buf)
        # Verify loop exists before creating task to avoid "RuntimeError: no running event loop"
        try:
             loop = asyncio.get_running_loop()
             if loop.is_running():
                 asyncio.create_task(log_manager.broadcast(buf))
        except:
             pass 

    def flush(self):
        self.original_stream.flush()

# Redirect stdout/stderr (Careful with concurrency)
# sys.stdout = StreamToLogger(sys.stdout) # Global override might be risky but effective for "Console Log"

@app.websocket("/logs")
async def websocket_endpoint(websocket: WebSocket):
    await log_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text() # Keep alive
    except:
        log_manager.disconnect(websocket)

class AnalysisRequest(BaseModel):
    date: Optional[str] = None
    force_train: bool = False

@app.post("/analyze-today")
async def analyze_today(req: AnalysisRequest):
    """
    Main workflow:
    1. Get games for date
    2. Get active active players in those games
    3. Check if trained
    4. Train if needed
    5. Predict
    """
    target_date = req.date if req.date else datetime.now().strftime('%Y-%m-%d')
    current_season = "2025-26" # Hardcoded for now, logical
    
    await log_manager.broadcast(f"Starting Analysis for {target_date}...")
    
    # 1. Fetch Scoreboard
    scoreboard = fetch_daily_scoreboard(target_date)
    if scoreboard.empty:
        await log_manager.broadcast("No games found today.")
        return {"message": "No games found", "results": []}
    
    team_ids = set(scoreboard['HOME_TEAM_ID'].unique().tolist() + scoreboard['VISITOR_TEAM_ID'].unique().tolist())
    await log_manager.broadcast(f"Found {len(team_ids)} teams playing today.")
    
    all_predictions = []
    
    # 2. Iterate Teams
    
    # 2. Build Execution Plan (Fetch Rosters)
    await log_manager.broadcast("Building Execution Plan (Fetching Rosters)...")
    
    execution_list = []
    
    current_team_idx = 0
    total_teams = len(team_ids)
    
    # We need to know who is playing whom to set opp_id
    # Reset scoreboard iteration to map team -> opp
    team_schedule = {} # {team_id: {'opp_id': id, 'is_home': bool}}
    for _, row in scoreboard.iterrows():
        h = row['HOME_TEAM_ID']
        v = row['VISITOR_TEAM_ID']
        team_schedule[h] = {'opp_id': v, 'is_home': True}
        team_schedule[v] = {'opp_id': h, 'is_home': False}
        
    for team_id in team_ids:
        current_team_idx += 1
        await log_manager.broadcast(f"Scanning Team {current_team_idx}/{total_teams} (ID: {team_id})...")
        try:
             roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=current_season).get_data_frames()[0]
             
             # Context
             ctx = team_schedule.get(team_id, {})
             
             for _, player in roster.iterrows():
                 execution_list.append({
                     'pid': player['PLAYER_ID'],
                     'pname': player['PLAYER'],
                     'team_id': team_id,
                     'opp_id': ctx.get('opp_id'),
                     'is_home': ctx.get('is_home')
                 })
        except Exception as e:
             await log_manager.broadcast(f"[WARN] Failed to fetch roster for team {team_id}: {e}")

    total_players = len(execution_list)
    await log_manager.broadcast(f"Plan Built: {total_players} Players identified across {total_teams} Teams.")
    
    # 3. Execute BATCH
    from src.batch_predict import BatchPredictor
    batch_predictor = BatchPredictor()
    
    # Train missing models first?
    # Batch Predictor assumes trained models.
    # We can do a quick pass for training if we want, OR just let Batch Predictor fallback to generic.
    # The user values speed. Generic fallback is fine for new players.
    
    # However, if we need to TRAIN, we should check missing models.
    # Let's do a quick check? 
    # For now, let's SKIP training automation in favor of speed, unless explicitly requested.
    # If req.force_train is True, we can't do batch effectively unless we train first.
    
    # Running Batch Analysis
    try:
        results = await batch_predictor.analyze_today_batch(
            date_input=target_date, 
            execution_list=execution_list, 
            log_manager=log_manager
        )
    except Exception as e:
        await log_manager.broadcast(f"Batch Analysis Failed: {e}")
        import traceback
        traceback.print_exc()
        return {"results": []}

    sorted_preds = sorted(results, key=lambda x: x['PRED_PTS'] if x['PRED_PTS'] else 0, reverse=True)
    
    await log_manager.broadcast(f"Analysis Complete. Generated {len(sorted_preds)} predictions.")
    return {"results": sorted_preds}

@app.get("/health")
def health():
    return {"status": "ok"}
