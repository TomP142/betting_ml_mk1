import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict

# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(DATA_DIR, 'models')

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import List, Dict, Tuple

# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(DATA_DIR, 'models')

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

class FeatureEngineer:
    def __init__(self):
        self.player_encoder = LabelEncoder()
        self.team_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def load_all_data(self) -> pd.DataFrame:
        """Loads and concatenates all available season logs."""
        files = [f for f in os.listdir(RAW_DIR) if f.startswith('game_logs_')]
        dfs = []
        for f in files:
            path = os.path.join(RAW_DIR, f)
            print(f"Loading {path}...")
            df = pd.read_csv(path)
            # Parse Season Year from filename (e.g. game_logs_2023-24.csv -> 2024)
            try:
                # filename is just 'f'
                # game_logs_2023-24.csv
                s_str = f.split('_')[-1].replace('.csv', '') # 2023-24
                # We want 2024
                s_year = int(s_str.split('-')[0]) + 1
                df['SEASON_YEAR'] = s_year
            except:
                print(f"Failed to parse season from {f}")
                df['SEASON_YEAR'] = 2024 # Fallback
            dfs.append(df)
            
        if not dfs:
             raise FileNotFoundError("No game logs found in raw directory.")
             
        full_df = pd.concat(dfs, ignore_index=True)
        
        # Filter Data (User Request: Just 24-25 and 25-26)
        # 2024-25 -> SEASON_YEAR 2025
        # 2025-26 -> SEASON_YEAR 2026
        full_df = full_df[full_df['SEASON_YEAR'] >= 2025].copy()
        print(f"Loaded {len(full_df)} rows (Filtered for 2025+).")
        
        return full_df

    def load_player_positions(self) -> pd.DataFrame:
        """Loads and combines player position data for all seasons."""
        dfs = []
        # Get all position files
        import glob
        files = glob.glob(os.path.join(RAW_DIR, 'player_positions_*.csv'))
        for f in files:
            df = pd.read_csv(f)
            # Add season year from filename
            season = os.path.basename(f).split('_')[-1].replace('.csv', '')
            df['SEASON_YEAR'] = int(season.split('-')[0]) + 1 # 2023-24 -> 2024
            
            # Key: Player + Season mapping
            dfs.append(df)
            
        if not dfs:
            print("Warning: No position files found.")
            return pd.DataFrame()
            
        return pd.concat(dfs)

    def calculate_dvp(self, df: pd.DataFrame, pos_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Defense vs Position (DvP).
        1. Merge Player positions into Game Logs.
        2. Group by Opponent Team + Opponent Position + Date.
        3. Calculate Rolling Avg of stats allowed.
        """
        print("Calculating Defense vs. Position (DvP)...")
        
        # 1. Merge Position to Game Logs
        # We need to match Player + Season. 
        # DF has SEASON_YEAR (e.g. 2024 for 2023-24). 
        # Position DF also has SEASON_YEAR.
        # But Position DF might have duplicates if traded. Take last?
        pos_df = pos_df.drop_duplicates(subset=['PLAYER_ID', 'SEASON_YEAR'], keep='last')
        
        df = df.merge(pos_df[['PLAYER_ID', 'SEASON_YEAR', 'POSITION_SIMPLE']], 
                      on=['PLAYER_ID', 'SEASON_YEAR'], 
                      how='left')
                      
        df['POSITION_SIMPLE'] = df['POSITION_SIMPLE'].fillna('F') # Default to F if missing
        
        # 2. Aggregating: What did Team X allow to Position Y?
        # Use OPP_TEAM_ABBREVIATION instead of ID
        agg_key = 'OPP_TEAM_ABBREVIATION'
        
        defensive_stats = df.groupby(['GAME_DATE', agg_key, 'POSITION_SIMPLE'])[['PTS', 'REB', 'AST']].sum().reset_index()
        defensive_stats.columns = ['GAME_DATE', agg_key, 'POSITION_SIMPLE', 'def_PTS', 'def_REB', 'def_AST']
        
        # Now calculating rolling averages for each Team + Position combo
        defensive_stats = defensive_stats.sort_values('GAME_DATE')
        
        cols = ['def_PTS', 'def_REB', 'def_AST']
        for col in cols:
            defensive_stats[f'ROLL_{col}'] = defensive_stats.groupby([agg_key, 'POSITION_SIMPLE'])[col].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
            
        # 3. Merge back
        df = df.merge(defensive_stats[['GAME_DATE', agg_key, 'POSITION_SIMPLE', 'ROLL_def_PTS', 'ROLL_def_REB', 'ROLL_def_AST']],
                      on=['GAME_DATE', agg_key, 'POSITION_SIMPLE'],
                      how='left')
                      
        # Fill NaNs (start of season)
        for col in ['ROLL_def_PTS', 'ROLL_def_REB', 'ROLL_def_AST']:
            df[col] = df[col].fillna(df[col].mean()) # Fill with global average
            
        print("DvP Features Added.")
        return df

    def calculate_h2h(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Calculating Head-to-Head (H2H)...")
        # Rolling avg against specific opponent
        # Group by Player, Opponent
        cols = ['PTS', 'REB', 'AST', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS']
                
        # Ensure cols exist
        for c in cols:
            if c not in df.columns:
                df[c] = 0.0
        
        # Sort by date to ensure rolling window is correct
        df = df.sort_values('GAME_DATE')
        
        # Optimize: Group ONCE for all columns instead of looping
        # This avoids re-sorting/hashing for every column (speedup 10x+)
        grouped = df.groupby(['PLAYER_ID', 'OPP_TEAM_ABBREVIATION'])[cols]
        
        # Apply transformation to all columns at once
        h2h_features = grouped.transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        
        # Rename columns to H2H_ prefix
        h2h_features.columns = [f'H2H_{c}' for c in cols]
        
        # Merge back (concat is safe because index is preserved by transform)
        df = pd.concat([df, h2h_features], axis=1)
             
        # Fill NaNs with global mean
        for col in cols:
            val = df[col].mean()
            df[f'H2H_{col}'] = df[f'H2H_{col}'].fillna(val)
            
        print("H2H Features Added.")
        return df

    def _identify_stars(self, df: pd.DataFrame) -> dict:
        """
        Identifies top 2 scorers per team per season.
        Returns: Dict { (TeamID, Season) : [PlayerID1, PlayerID2] }
        """
        print("Identifying Star Players...")
        summary = df.groupby(['SEASON_YEAR', 'TEAM_ID', 'PLAYER_ID'])['PTS'].mean().reset_index()
        
        stars_map = {}
        
        for (season, team), group in summary.groupby(['SEASON_YEAR', 'TEAM_ID']):
            top_scorers = group.sort_values('PTS', ascending=False).head(2)
            stars_map[(team, season)] = top_scorers['PLAYER_ID'].tolist()
            
        return stars_map

    def calculate_teammate_absence(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Calculating Teammate Impact (Stars Out)...")
        stars_map = self._identify_stars(df)
        
        # Optimize: Group by Team/Date to get roster
        present_players = df.groupby(['TEAM_ID', 'GAME_DATE'])['PLAYER_ID'].apply(set).to_dict()
        
        def count_stars_out(row):
            team = row['TEAM_ID']
            season = row['SEASON_YEAR']
            date = row['GAME_DATE']
            me = row['PLAYER_ID']
            
            stars = stars_map.get((team, season), [])
            if not stars: return 0
                
            present = present_players.get((team, date), set())
            
            missing = 0
            for star_id in stars:
                if star_id == me: continue 
                if star_id not in present:
                    missing += 1
            return missing

        df['STARS_OUT'] = df.apply(count_stars_out, axis=1)
        print("Teammate Impact Features Added.")
        return df

    def _identify_rotation_players(self, df: pd.DataFrame, min_minutes: float = 15.0) -> dict:
        """
        Identifies rotation players (avg > min_minutes) per team per season.
        Returns: Dict { (TeamID, Season) : Set(PlayerIDs) }
        """
        print("Identifying Rotation Players...")
        summary = df.groupby(['SEASON_YEAR', 'TEAM_ID', 'PLAYER_ID'])['MIN'].mean().reset_index()
        
        rotation_map = {}
        for (season, team), group in summary.groupby(['SEASON_YEAR', 'TEAM_ID']):
            rotation_players = group[group['MIN'] > min_minutes]['PLAYER_ID'].unique()
            rotation_map[(team, season)] = set(rotation_players)
            
        return rotation_map

    def calculate_missing_players(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Calculating Specific Missing Players...")
        rotation_map = self._identify_rotation_players(df)
        present_players = df.groupby(['TEAM_ID', 'GAME_DATE'])['PLAYER_ID'].apply(set).to_dict()
        
        def get_missing(row):
            team = row['TEAM_ID']
            season = row['SEASON_YEAR']
            date = row['GAME_DATE']
            me = row['PLAYER_ID']
            
            # Expected roster
            expected = rotation_map.get((team, season), set())
            if not expected: return []
            
            # Who actually played
            present = present_players.get((team, date), set())
            
            missing = []
            for pid in expected:
                if pid == me: continue # Don't count myself as missing
                if pid not in present:
                    missing.append(pid)
            
            return missing

        # Apply and store as list
        # Using list in DataFrame cell is possible but requires care with saving/loading (CSV converts to string).
        # We should serialize to string "ID1_ID2_ID3" for CSV safety.
        df['MISSING_PLAYER_IDS'] = df.apply(get_missing, axis=1)
        # Convert to string for CSV compatibility
        df['MISSING_PLAYER_IDS'] = df['MISSING_PLAYER_IDS'].apply(lambda x: "_".join(map(str, x)) if x else "NONE")
        
        print("Missing Players Identified.")
        return df

    def _derive_opponent(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parses MATCHUP to get Opponent Abbreviation."""
        # MATCHUP: "LAL vs. TOR" or "LAL @ TOR"
        # Split by ' ' and take last element.
        df['OPP_TEAM_ABBREVIATION'] = df['MATCHUP'].apply(lambda x: x.split(' ')[-1])
        return df

    def calculate_team_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Team-Level Advanced Stats (Off Rtg, Def Rtg, Pace)
        and merges them as 'Opponent Stats' for the player.
        """
        print("Calculating Team Advanced Stats...")
        
        # 1. Aggregate Player Stats to Team-Game Level
        # Sum targets: PTS, FGA, FTA, OREB, TOV, MIN
        # We need DREB? Total REB - OREB = DREB usually.
        # Poss formula: 0.96 * (FGA + 0.008*FTA - 1.07*(OREB/ (OREB + OppDREB))) ... too complex.
        # Simple NBA Poss: FGA + 0.44*FTA - OREB + TOV
        
        team_game_stats = df.groupby(['GAME_ID', 'TEAM_ID', 'GAME_DATE', 'TEAM_ABBREVIATION'])[['PTS', 'FGA', 'FTA', 'OREB', 'TOV', 'MIN']].sum().reset_index()
        
        # 2. Calculate Basic Team Metrics
        team_game_stats['POSS'] = team_game_stats['FGA'] + 0.44 * team_game_stats['FTA'] - team_game_stats['OREB'] + team_game_stats['TOV']
        # Normalize Pace to 48 mins: Poss / (MIN / 5) * 48. (5 players on court)
        # Avoid div by zero
        team_game_stats['PACE'] = 48 * (team_game_stats['POSS'] / (team_game_stats['MIN'] / 5))
        team_game_stats['OFF_RTG'] = 100 * (team_game_stats['PTS'] / team_game_stats['POSS'])
        
        # 3. Find Opponent to get DEF_RTG (Opponent's Off Rtg)
        # Self-join on GAME_ID
        game_opps = team_game_stats.merge(team_game_stats[['GAME_ID', 'TEAM_ID', 'OFF_RTG', 'PACE']], 
                                          on='GAME_ID', suffixes=('', '_OPP'))
        
        # Filter out self-matches (Team A vs Team A) - usually game has 2 rows.
        game_opps = game_opps[game_opps['TEAM_ID'] != game_opps['TEAM_ID_OPP']]
        
        # Now we have: For Team A, DEF_RTG = OPP_OFF_RTG
        game_opps['DEF_RTG'] = game_opps['OFF_RTG_OPP']
        
        # We want to store these stats FOR THE TEAM, so we can look them up as an OPPONENT later.
        # Identify the Team we are tracking metrics FOR.
        # We want "How good is Team X?".
        # So we keep rows for Team X.
        
        cols_to_keep = ['GAME_DATE', 'TEAM_ABBREVIATION', 'OFF_RTG', 'DEF_RTG', 'PACE']
        team_history = game_opps[cols_to_keep].copy()
        team_history = team_history.sort_values('GAME_DATE')
        
        # 4. Calculate Rolling Stats
        # Group by Team
        roll_cols = ['OFF_RTG', 'DEF_RTG', 'PACE']
        for col in roll_cols:
            # Shift 1 to use PAST games only
            team_history[f'TEAM_ROLL_{col}'] = team_history.groupby('TEAM_ABBREVIATION')[col].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
            
        # Fill NaNs with league average
        for col in roll_cols:
            mean_val = team_history[col].mean()
            team_history[f'TEAM_ROLL_{col}'] = team_history[f'TEAM_ROLL_{col}'].fillna(mean_val)
            
        # 5. Merge back to Player DF as "OPP_TEAM_ROLL_..."
        # df has 'OPP_TEAM_ABBREVIATION'.
        # We join team_history on [TEAM_ABBREV, GAME_DATE] -> [OPP_TEAM_ABBREV, GAME_DATE]
        
        # Prepare merge source
        merge_src = team_history[['GAME_DATE', 'TEAM_ABBREVIATION', 'TEAM_ROLL_OFF_RTG', 'TEAM_ROLL_DEF_RTG', 'TEAM_ROLL_PACE']]
        merge_src.columns = ['GAME_DATE', 'OPP_TEAM_ABBREVIATION', 'OPP_ROLL_OFF_RTG', 'OPP_ROLL_DEF_RTG', 'OPP_ROLL_PACE']
        
        # Merge
        df = df.merge(merge_src, on=['GAME_DATE', 'OPP_TEAM_ABBREVIATION'], how='left')
        
        # Fill missing (e.g. first game of season or no history)
        # Fill with global means from the merge_src (approx league avg)
        for col in ['OPP_ROLL_OFF_RTG', 'OPP_ROLL_DEF_RTG', 'OPP_ROLL_PACE']:
             df[col] = df[col].fillna(merge_src[col].mean())
             
        print("Team Advanced Stats Added.")
        return df


    def process(self, df: pd.DataFrame, is_training: bool = True, overrides: dict = None) -> pd.DataFrame:
        """
        overrides: Dict of {(PlayerID, Date_Str) : {'Col': Val}} to manually set feature values before scaling.
        """
        print("Starting Feature Engineering...")
        if is_training:
            # Filter garbage
            df = df.dropna(subset=['PTS', 'REB', 'AST'])
            
        # Ensure Date Format Early
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        
        # Derive Opponent Info EARLY
        df = self._derive_opponent(df)
        
        # Load Positions & Add DvP
        pos_df = self.load_player_positions()
        if not pos_df.empty:
            df = self.calculate_dvp(df, pos_df)
        else:
             print("Skipping DvP (No position data)")
             
        # Add H2H
        df = self.calculate_h2h(df)
        
        # Add Team Stats (Defense/Pace)
        df = self.calculate_team_stats(df)
        
        # Add Teammate Impact
        df = self.calculate_missing_players(df)

        # APPLY OVERRIDES (Pre-Scaling)
        if overrides:
            print(f"Applying manual overrides: {overrides}")
            # Ensure Date column is string or matching format
            # df['GAME_DATE'] is datetime by now (Line 196 converts it, wait)
            # Line 196: df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            # So keys in overrides should use Timestamp or we convert.
            
            for (pid, date_str), vals in overrides.items():
                target_dt = pd.to_datetime(date_str)
                mask = (df['PLAYER_ID'] == pid) & (df['GAME_DATE'] == target_dt)
                if mask.any():
                    for col, val in vals.items():
                        # If overriding MISSING_PLAYER_IDS, ensure string format?
                        # Caller (daily_predict) will pass list of ints?
                        if col == 'MISSING_PLAYER_IDS' and isinstance(val, list):
                            val = "_".join(map(str, val))
                        
                        df.loc[mask, col] = val
                        print(f"Overrode {col}={val} for {pid} on {date_str}")

        # Sort
        # df['GAME_DATE'] is already datetime
        df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])
        
        # 1. Rolling Averages (Player Form)
        # Shift by 1 so we don't cheat using today's stats
        cols = ['PTS', 'REB', 'AST', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS', 'MIN']
        for col in cols:
             if col in df.columns:
                 df[f'ROLL_{col}'] = df.groupby('PLAYER_ID')[col].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
             
        # 2. Rest Days
        df['GAME_DATE_DT'] = df['GAME_DATE']
        df['PREV_GAME_DATE'] = df.groupby('PLAYER_ID')['GAME_DATE_DT'].shift(1)
        df['DAYS_REST'] = (df['GAME_DATE_DT'] - df['PREV_GAME_DATE']).dt.days
        df['DAYS_REST'] = df['DAYS_REST'].fillna(7)
        df['DAYS_REST'] = df['DAYS_REST'].clip(upper=7)
        
        # 3. Apply Scaling / Encoding
        if is_training:
            df = self._fit_artifacts(df, is_training=True) 
        else:
            # For inference, use saved scalers
            df = self._apply_transformations(df, is_training=False)
        
        # Drop NaNs
        # Important: Rolling stats take 1 game to start. First row is NaN.
        # If inference key row is the first row (impossible if history loaded), it might be dropped.
        if is_training:
            df = df.dropna(subset=['ROLL_PTS']) # Drop rows without features
        else:
            # Keep everything, let caller extract the last row
            pass
        
        return df

    def _add_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Home/Away
        df['IS_HOME'] = df['MATCHUP'].apply(lambda x: 1 if ' vs. ' in x else 0)
        
        # Rest Days
        df['REST_DAYS'] = df.groupby('PLAYER_ID')['GAME_DATE'].diff().dt.days
        df['REST_DAYS'] = df['REST_DAYS'].fillna(3) # Assume 3 days rest for first game
        df['IS_B2B'] = (df['REST_DAYS'] == 1).astype(int)
        
        # Parse Season Year (for weighting)
        # SEASON_ID is usually like 22023 -> 2023.
        df['SEASON_YEAR'] = df['SEASON_ID'].astype(str).str[-4:].astype(int)
        
        return df

    def _add_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        rolling_metrics = ['PTS', 'REB', 'AST', 'MIN', 'FGM', 'FGA', 'FG3M', 'FG3A']
        windows = [5, 10, 20]
        
        # Vectorized Rolling would be faster but GroupBy is safer for strict player isolation
        # Using a simplistic loop for clarity/safety.
        
        # Pre-sort (already done)
        for metric in rolling_metrics:
            if metric not in df.columns: continue
            
            # Group shift first to prevent data leak
            shifted = df.groupby('PLAYER_ID')[metric].shift(1)
            
            for w in windows:
                col_name = f'ROLL_{metric}_{w}'
                # Calculate rolling on shifted data
                df[col_name] = shifted.groupby(df['PLAYER_ID']).rolling(window=w, min_periods=1).mean().reset_index(0, drop=True)
                
            # Season Average
            col_name_season = f'AVG_{metric}_SEASON'
        return df

    def _fit_artifacts(self, df: pd.DataFrame, is_training: bool = True):
        if not is_training:
             # Just load and transform? actually _fit_artifacts implies fitting.
             # For inference we should use separate method.
             # But let's just ignore for now or return df.
             # Actually, if is_training=False, we shouldn't be here?
             # 'process' calls it.
             # Let's just allow the arg.
             pass
             
        print("Fitting scalers and encoders...")
        
        # Categoricals
        player_encoder = LabelEncoder()
        df['PLAYER_IDX'] = player_encoder.fit_transform(df['PLAYER_ID'].astype(str))
        
        team_encoder = LabelEncoder()
        df['TEAM_IDX'] = team_encoder.fit_transform(df['OPP_TEAM_ABBREVIATION'].astype(str))
        
        # Continuous Features
        # Exclude ID, Date, Target, and categorical strings
        # Keep ROLL_*, H2H_*, DAYS_REST
        exclude = ['PLAYER_ID', 'SEASON_ID', 'DEV_SEASON_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 
                   'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL', 'MIN', 'VIDEO_AVAILABLE', 
                   'PLAYER_NAME', 'OPP_TEAM_ABBREVIATION', 'SEASON_YEAR', 
                   'PLAYER_IDX', 'TEAM_IDX', 'GAME_DATE_DT', 'PREV_GAME_DATE',
                   'POSITION_SIMPLE', 'def_PTS', 'def_REB', 'def_AST',
                   'POSS', 'OFF_RTG', 'DEF_RTG', 'PACE', 'TEAM_ABBREVIATION_OPP'] # Exclude intermediate calc
                   
        # Also exclude Targets
        exclude += ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'FGM', 'FGA', 'FG_PCT', 
                    'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'PLUS_MINUS', 'FANTASY_PTS']
        
        feature_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
        print(f"Identified {len(feature_cols)} feature columns: {feature_cols}")
        
        # Fit SELF.scaler
        df[feature_cols] = self.scaler.fit_transform(df[feature_cols].fillna(0))
        
        # Save Artifacts
        joblib.dump(player_encoder, os.path.join(MODELS_DIR, 'player_encoder.joblib'))
        joblib.dump(team_encoder, os.path.join(MODELS_DIR, 'team_encoder.joblib'))
        joblib.dump(self.scaler, os.path.join(MODELS_DIR, 'scaler.joblib')) # Save self.scaler
        joblib.dump(feature_cols, os.path.join(MODELS_DIR, 'feature_cols.joblib'))
        
        return df

    def _load_artifacts(self):
        self.player_encoder = joblib.load(os.path.join(MODELS_DIR, 'player_encoder.joblib'))
        self.team_encoder = joblib.load(os.path.join(MODELS_DIR, 'team_encoder.joblib'))
        self.scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))

    def _apply_transformations(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        if is_training:
            return df # Already transformed in _fit_artifacts
            
        # 1. Feature Columns
        self._load_artifacts() # Ensure scaler/encoders are loaded
        feature_cols = joblib.load(os.path.join(MODELS_DIR, 'feature_cols.joblib'))
        
        # Scale continuous
        df[feature_cols] = self.scaler.transform(df[feature_cols].fillna(0))
        
        # 2. Embeddings Indices
        # Player (Handle unknown)
        df['PLAYER_ID_STR'] = df['PLAYER_ID'].astype(str)
        # Map unknown to 'UNKNOWN'
        # Map unknown to a known class (fallback) to avoid crash
        known_players = set(self.player_encoder.classes_)
        # Use first class as generic fallback if current ID is unknown
        fallback = self.player_encoder.classes_[0] 
        df.loc[~df['PLAYER_ID_STR'].isin(known_players), 'PLAYER_ID_STR'] = fallback
        
        df['PLAYER_IDX'] = self.player_encoder.transform(df['PLAYER_ID_STR'])
        
        # Team
        df['OPP_TEAM_ABBREVIATION'] = df['MATCHUP'].apply(lambda x: x.split(' ')[-1])
        known_teams = set(self.team_encoder.classes_)
        fallback_team = self.team_encoder.classes_[0]
        df.loc[~df['OPP_TEAM_ABBREVIATION'].isin(known_teams), 'OPP_TEAM_ABBREVIATION'] = fallback_team
        
        df['TEAM_IDX'] = self.team_encoder.transform(df['OPP_TEAM_ABBREVIATION'])
        
        return df

    def save_features(self, df: pd.DataFrame, filename: str = 'processed_features.csv'):
        # Save minimal dataset for Training to save space/IO
        # But we need targets (PTS, REB, AST) and Season Year
        # We must include the new columns!
        # feature_cols joblib already has the names? 
        # Yes, _fit_artifacts logic below needs to include them in the 'continuous_cols' list.
        # But here we explicitly list non-feature metadata key columns.
        cols = ['PLAYER_ID', 'PLAYER_IDX', 'TEAM_IDX', 'SEASON_YEAR', 'GAME_DATE', 'PTS', 'REB', 'AST', 'MISSING_PLAYER_IDS']
        feature_cols = joblib.load(os.path.join(MODELS_DIR, 'feature_cols.joblib'))
        
        # Ensure all columns exist
        final_cols = cols + feature_cols
        final_df = df[final_cols]
        final_df.to_csv(os.path.join(PROCESSED_DIR, filename), index=False)
        print(f"Saved processed features to {filename}")

if __name__ == "__main__":
    fe = FeatureEngineer()
    try:
        df = fe.load_all_data()
        df_processed = fe.process(df)
        fe.save_features(df_processed)
        print(f"Processed {len(df_processed)} rows.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
