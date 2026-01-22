import pandas as pd
import numpy as np
import os
import torch
import joblib
from datetime import datetime
from src.data_fetch import fetch_daily_scoreboard
from src.feature_engineer import FeatureEngineer
from src.train_models import NBAPredictor
from nba_api.stats.static import teams
import asyncio

# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BatchPredictor:
    def __init__(self):
        self.fe = FeatureEngineer()
        self.resources = {} # Cache resources
        self.models_cache = {} # Cache loaded PyTorch models

    def load_common_resources(self):
        print("Loading common resources...")
        self.resources['p_enc'] = joblib.load(os.path.join(MODELS_DIR, 'player_encoder.joblib'))
        self.resources['t_enc'] = joblib.load(os.path.join(MODELS_DIR, 'team_encoder.joblib'))
        self.resources['scaler'] = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
        self.resources['feature_cols'] = joblib.load(os.path.join(MODELS_DIR, 'feature_cols.joblib'))
        
    def _get_model(self, player_id: int):
        # Determine model path
        specific_path = os.path.join(MODELS_DIR, f'pytorch_player_{player_id}.pth')
        # Corrected generic model name
        generic_path = os.path.join(MODELS_DIR, 'pytorch_nba_global.pth')
        
        path_to_load = specific_path if os.path.exists(specific_path) else generic_path
        
        # Check cache
        if path_to_load in self.models_cache:
            return self.models_cache[path_to_load]
            
        # Load
        p_enc = self.resources['p_enc']
        t_enc = self.resources['t_enc']
        f_cols = self.resources['feature_cols']
        
        num_players = len(p_enc.classes_) + 1
        num_teams = len(t_enc.classes_)
        num_cont = len(f_cols)
        
        model = NBAPredictor(num_players, num_teams, num_cont)
        
        # Safely load weights
        try:
            model.load_state_dict(torch.load(path_to_load, map_location=device))
        except Exception as e:
            print(f"Error loading model {path_to_load}: {e}")
            # Fallback to generic if specific failed?
            if path_to_load != generic_path:
                model.load_state_dict(torch.load(generic_path, map_location=device))
            
        model.to(device)
        model.eval()
        
        self.models_cache[path_to_load] = model
        self.models_cache[path_to_load] = model
        return model

    def _get_metrics(self, player_id: int):
        """Loads MAE metrics for confidence intervals."""
        specific_path = os.path.join(MODELS_DIR, f'metrics_player_{player_id}.json')
        generic_path = os.path.join(MODELS_DIR, 'metrics_global.json')
        
        path_to_load = specific_path if os.path.exists(specific_path) else generic_path
        
        # Default fallback
        metrics = {
            'mae_pts': 6.0, 'mae_reb': 2.5, 'mae_ast': 2.0,
            'baseline_pts': 7.0, 'baseline_reb': 3.0, 'baseline_ast': 2.5
        }
        
        try:
            import json
            if os.path.exists(path_to_load):
                with open(path_to_load, 'r') as f:
                    metrics = json.load(f)
        except:
             pass
             
        return metrics

    async def analyze_today_batch(self, date_input: str = None, execution_list: list = None, log_manager=None):
        """
        Main Batch Function.
        1. Loads History ONCE.
        2. Appends 'Phantom Rows' for ALL players in execution_list.
        3. Processes Features ONCE.
        4. Predicts in loop.
        """
        target_date = date_input if date_input else datetime.now().strftime('%Y-%m-%d')
        
        if log_manager: await log_manager.broadcast("Loading historical data from disk...")
        
        # 1. Load All Data (Heavy IO - done once)
        df_history = await asyncio.to_thread(self.fe.load_all_data)
        
        if log_manager: await log_manager.broadcast("Generating Prediction Rows...")
        
        # 2. Create Phantom Rows
        phantom_rows = []
        
        # Fetch team info cache
        team_abbr_cache = {}
        
        for item in execution_list:
            pid = item['pid']
            tid = item['team_id']
            # We need opponent ID. 
            # execution_list came from api.py which scanned the scoreboard.
            # But api.py only sent pid/tid. We need the GAME info (Opponent).
            # Let's pass the Full execution Item which might need to include Opponent Info.
            # OR we re-fetch scoreboard here? 
            # API.py has the scoreboard logic. Let's make API pass 'opp_id' and 'is_home' in execution_list.
            # Assuming api.py is updated to pass 'opp_id' and 'is_home'.
            
            # Fallback if keys missing (for safety during refactor)
            if 'opp_id' not in item:
                 continue
                 
            opp_id = item['opp_id']
            is_home = item['is_home']
            
            # Resolve Abbr
            if tid not in team_abbr_cache:
                try: team_abbr_cache[tid] = teams.find_team_name_by_id(tid)['abbreviation']
                except: team_abbr_cache[tid] = "UNK"
                
            if opp_id not in team_abbr_cache:
                try: team_abbr_cache[opp_id] = teams.find_team_name_by_id(opp_id)['abbreviation']
                except: team_abbr_cache[opp_id] = "UNK"
                
            own_abbr = team_abbr_cache[tid]
            opp_abbr = team_abbr_cache[opp_id]
            matchup = f"{own_abbr} vs. {opp_abbr}"
            
            new_row = {
                'PLAYER_ID': pid,
                'GAME_DATE': target_date,
                'MATCHUP': matchup,
                'TEAM_ID': tid,
                'PTS': np.nan, 'REB': np.nan, 'AST': np.nan,
                'MIN': 0, 'FGA': 0,
                'SEASON_YEAR': 2026
            }
            phantom_rows.append(new_row)
            
        if not phantom_rows:
            return []
            
        # Append all
        df_batch = pd.concat([df_history, pd.DataFrame(phantom_rows)], ignore_index=True)
        
        if log_manager: await log_manager.broadcast("Running Bulk Feature Engineering (This takes ~15s)...")
        
        # 3. Process Features (Heavy CPU - done once)
        df_processed = self.fe.process(df_batch, is_training=False)
        
        # Load resources for prediction
        self.load_common_resources()
        
        # 4. Predict Loop
        results_list = []
        target_dt = pd.to_datetime(target_date)
        
        # Extract just today's rows
        today_data = df_processed[df_processed['GAME_DATE'] == target_dt].copy()
        
        # Create map for fast lookup
        # today_data.set_index('PLAYER_ID', inplace=True) 
        # Actually duplicate PLAYER_IDs unlikely today but possible if double header? No.
        
        total_p = len(execution_list)
        
        # Pre-calculate Last Played Dates for Recency Filter
        # Ensure datetime
        df_history['GAME_DATE'] = pd.to_datetime(df_history['GAME_DATE'])
        last_played_map = df_history.groupby('PLAYER_ID')['GAME_DATE'].max().to_dict()
        
        target_dt = pd.to_datetime(target_date)
        
        if log_manager: await log_manager.broadcast("Inferencing Models...")
        
        processed_count = 0
        
        for item in execution_list:
            pid = item['pid']
            pname = item['pname']
            
            # RECENTLY ACTIVE CHECK
            # If player hasn't played in > 30 days, skip.
            last_date = last_played_map.get(pid)
            if last_date:
                days_inactive = (target_dt - last_date).days
                if days_inactive > 30:
                    # print(f"Skipping {pname} - Inactive for {days_inactive} days")
                    continue
            else:
                 # No history at all? Skip or keep?
                 # If no history, they are probably a rookie or new. Keep them or skip?
                 # If we have no history, we can't predict well anyway (using generic).
                 # Let's keep them if they are in the roster, maybe?
                 # Actually, if they have NO games in df_history (which covers multiple years), they are likely irrelevant or very new.
                 pass
            
            # Get row
            # mask = today_data['PLAYER_ID'] == pid ... fast enough?
            # optimization: use dictionary mapping
            # But let's just assume DataFrame lookup is fast enough for 200 items.
            
            player_row = today_data[today_data['PLAYER_ID'] == pid]
            if player_row.empty:
                continue
                
            processed_count += 1
            if processed_count % 10 == 0:
                 if log_manager: await log_manager.broadcast(f"Inference Progress: {processed_count}/{total_p}")
            
            # Load Model
            model = self._get_model(pid)
            
            # Prepare Tensor
            # We can batch predict if we use a GENERIC model for everyone.
            # But since we switch models per player, we must loop inference.
            # (Inference is fast: 0.01s on CPU).
            
            p_idx = torch.LongTensor(player_row['PLAYER_IDX'].values).to(device)
            t_idx = torch.LongTensor(player_row['TEAM_IDX'].values).to(device)
            x_cont = torch.FloatTensor(player_row[self.resources['feature_cols']].values).to(device)
            
            # Missing IDs embedding
            m_indices = []
            pad_idx = len(self.resources['p_enc'].classes_)
            # Logic for 'missing_ids' passed from API? 
            # For now assume None/Empty
            m_indices += [pad_idx] * 3
            batch_size = len(player_row)
            m_idx_tensor = torch.LongTensor([m_indices] * batch_size).to(device)
            
            with torch.no_grad():
                preds = model(p_idx, t_idx, x_cont, m_idx_tensor)
                
            preds_np = preds.cpu().numpy()
            
            # Get Metrics for confidence
            metrics = self._get_metrics(pid)
            mae_pts = metrics.get('mae_pts', 6.0)
            mae_reb = metrics.get('mae_reb', 2.5)
            mae_ast = metrics.get('mae_ast', 2.0)
            
            # 1. Combos
            pred_pts = float(preds_np[0, 0])
            pred_reb = float(preds_np[0, 1])
            pred_ast = float(preds_np[0, 2])
            
            pred_pra = pred_pts + pred_reb + pred_ast
            pred_pr = pred_pts + pred_reb
            pred_pa = pred_pts + pred_ast
            pred_ra = pred_reb + pred_ast
            
            # Combos MAE (Sum for conservative estimate, or Sqrt for variance)
            # User wants "Expected error". Sum is safe upper bound. Sqrt is statistical.
            # Let's use Sum to be conservative for "Good Lines".
            mae_pra = mae_pts + mae_reb + mae_ast
            mae_pr = mae_pts + mae_reb
            mae_pa = mae_pts + mae_ast
            mae_ra = mae_reb + mae_ast

            # FILTER: Exclude Inactive Players
            # If the model predicts ~0 stats, the player is likely injured/out.
            if pred_pra < 1.0:
                # print(f"Skipping {pname} (PRA: {pred_pra:.2f}) - Likely Inactive")
                continue

            res = {
                'PLAYER_ID': pid,
                'PLAYER_NAME': pname,
                'GAME_DATE': target_date,
                'MATCHUP': player_row['MATCHUP'].iloc[0],
                
                # Base Preds
                'PRED_PTS': pred_pts,
                'PRED_REB': pred_reb,
                'PRED_AST': pred_ast,
                
                # Combos
                'PRED_PRA': pred_pra,
                'PRED_PR': pred_pr,
                'PRED_PA': pred_pa,
                'PRED_RA': pred_ra,
                
                # Betting Lines (The "Good" zone)
                # If Bookmaker Line is < Low, Bet Over.
                # If Bookmaker Line is > High, Bet Under.
                
                # PTS
                'LINE_PTS_LOW': pred_pts - mae_pts,
                'LINE_PTS_HIGH': pred_pts + mae_pts,
                'MAE_PTS': mae_pts,
                
                # REB
                'LINE_REB_LOW': pred_reb - mae_reb,
                'LINE_REB_HIGH': pred_reb + mae_reb,
                'MAE_REB': mae_reb,
                
                # AST
                'LINE_AST_LOW': pred_ast - mae_ast,
                'LINE_AST_HIGH': pred_ast + mae_ast,
                'MAE_AST': mae_ast,
                
                # PRA
                'LINE_PRA_LOW': pred_pra - mae_pra,
                'LINE_PRA_HIGH': pred_pra + mae_pra,
                'MAE_PRA': mae_pra,
                
                'OPPONENT': player_row['OPP_TEAM_ABBREVIATION'].iloc[0],
                'IS_HOME': item.get('is_home', False)
            }
            results_list.append(res)
            
        return results_list
