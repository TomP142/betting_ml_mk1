import pandas as pd
import time
from nba_api.stats.endpoints import playergamelogs, scoreboardv2, commonallplayers
from nba_api.stats.static import players, teams
from typing import List, Optional
import os
from datetime import datetime

# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
os.makedirs(RAW_DIR, exist_ok=True)

def fetch_active_players(season: str = '2023-24') -> pd.DataFrame:
    """
    Fetches list of all active players for the given season.
    """
    print(f"Fetching active players for season {season}...")
    try:
        # commonallplayers returns Roster info including POSITION
        resp = commonallplayers.CommonAllPlayers(is_only_current_season=1, season=season)
        df = resp.get_data_frames()[0]
        
        # Save essential columns: PERSON_ID, DISPLAY_FIRST_LAST, TEAM_ID, POSITION
        # Note: API column name for Position is 'POSITION' usually?
        # Let's check columns or just save raw first to be safe, then process.
        # Actually CommonAllPlayers might NOT have position in all versions.
        # 'commonplayerinfo' endpoint has it per player.
        # But 'commonallplayers' does not have 'POSITION' column in standard output usually? 
        # Actually it does not. It has TEAM_ID, GAMES_PLAYED_FLAG etc.
        # We might need to fetch rosters (commonteamroster) for every team? That's slow.
        # ALTERNATIVE: 'playerprofilev2'? No.
        # Let's try to get it from the 'players.get_active_players()' static if possible?
        # Static file won't have current season dynamic updates.
        
        # OPTIOIN B: Loop through all teams and get CommonTeamRoster.
        # There are 30 teams. 30 calls. totally fine.
        
        # Save to raw
        df.to_csv(os.path.join(RAW_DIR, f'active_players_{season}.csv'), index=False)
        return df # Placeholder, logic moved to main fetch_player_metadata
    except Exception as e:
        print(f"Error fetching active players: {e}")
        return pd.DataFrame()

def fetch_player_positions(season: str) -> pd.DataFrame:
    """
    Fetches position info for all players in a season by querying team rosters.
    """
    print(f"Fetching rosters (positions) for season {season}...")
    from nba_api.stats.endpoints import commonteamroster
    from nba_api.stats.static import teams
    
    nba_teams = teams.get_teams()
    all_rosters = []
    
    for team in nba_teams:
        try:
            time.sleep(0.6) # Rate limit
            tid = team['id']
            roster = commonteamroster.CommonTeamRoster(team_id=tid, season=season).get_data_frames()[0]
            # Keep PLAYER_ID and POSITION
            all_rosters.append(roster[['PLAYER_ID', 'POSITION', 'TeamID']])
        except Exception as e:
            print(f"Error fetching roster for {team['abbreviation']}: {e}")
            
    if all_rosters:
        full_roster = pd.concat(all_rosters)
        # Normalize Position: "G-F" -> "G", "F-C" -> "F" for simplified DvP
        full_roster['POSITION_SIMPLE'] = full_roster['POSITION'].str[0] # Take first letter
        
        path = os.path.join(RAW_DIR, f'player_positions_{season}.csv')
        full_roster.to_csv(path, index=False)
        print(f"Saved Metadata to {path}")
        return full_roster
    return pd.DataFrame()

def fetch_player_game_logs(season: str = '2023-24', player_id: Optional[int] = None) -> pd.DataFrame:
    """
    Fetches game logs for a specific player or all players if player_id is None (if API supports batch, otherwise it might be heavy).
    Actually, playergamelogs endpoint can fetch for all players if we don't specify player_id? 
    Let's check documentation or assume we iterate if needed.
    
    For efficiency, nbi_api 'PlayerGameLogs' gets data for ALL players if PlayerID is not set? 
    Let's try to get all players data in one go to be efficient.
    """
    print(f"Fetching game logs for season {season}...")
    try:
        # PlayerGameLogs (plural) usually gets league-wide logs if parameters allow, or we use LeagueGameLog
        from nba_api.stats.endpoints import leaguegamelog
        gamelog = leaguegamelog.LeagueGameLog(season=season, player_or_team_abbreviation='P')
        df = gamelog.get_data_frames()[0]
        
        # Save
        output_path = os.path.join(RAW_DIR, f'game_logs_{season}.csv')
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} records to {output_path}")
        return df
    except Exception as e:
        print(f"Error fetching game logs: {e}")
        return pd.DataFrame()

def fetch_schedule(season: str = '2023-24') -> pd.DataFrame:
    """
    Fetches the schedule. NBA API doesn't have a simple 'Schedule' endpoint that covers the whole future schedule easily in one go 
    without iterating dates or using specific endpoints. 
    However, we can likely find a workaround or use `LeagueGameLog` for past games.
    For future games, we might need to iterate `ScoreboardV2` or use an external source/static file if the API is limited.
    
    Actually, `scoreboardv2` is daily. 
    Let's use a simpler approach for now: rely on `LeagueGameLog` for historical data (training).
    For 'daily_predict', we will fetch 'today's' scoreboard specifically.
    """
    print("For full season schedule, we primarily rely on Game Logs for past data.")
    return pd.DataFrame()

def fetch_daily_scoreboard(game_date: str) -> pd.DataFrame:
    """
    Fetches games for a specific date (YYYY-MM-DD).
    """
    print(f"Fetching scoreboard for {game_date}...")
    try:
        board = scoreboardv2.ScoreboardV2(game_date=game_date)
        games = board.game_header.get_data_frame()
        return games
    except Exception as e:
        print(f"Error fetching scoreboard: {e}")
        return pd.DataFrame()

def fetch_all_seasons(start_year: int = 2021, end_year: int = 2025):
    """
    Fetches game logs for a range of seasons.
    start_year: e.g. 2021 for '2021-22' season.
    """
    seasons = []
    for year in range(start_year, end_year + 1):
        suffix = (year + 1) % 100
        season_str = f"{year}-{suffix:02d}"
        seasons.append(season_str)
        
    print(f"Fetching data for seasons: {seasons}")
    
    for season in seasons:
        # fetch_active_players(season) # Deprecated/Redundant if we get full roster
        fetch_player_positions(season)
        fetch_player_game_logs(season)
        time.sleep(1) # Be nice to API

if __name__ == "__main__":
    print("Starting Multi-Season Data Fetch...")
    try:
        # Fetch seasons 2021-22, 22-23, 23-24, 24-25, 25-26
        fetch_all_seasons(start_year=2021, end_year=2025) 
        print("Data fetch successful.")
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
