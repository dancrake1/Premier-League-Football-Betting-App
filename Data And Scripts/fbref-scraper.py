import sys
print(f"Using Python from: {sys.executable}")

import pandas as pd
from bs4 import BeautifulSoup
import requests
import time
import pickle
import os
from datetime import datetime
import random
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import logging
from io import StringIO
import urllib3

LOGPATH = '/Users/danielcrake/Desktop/Football Betting 2025/logs'

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{LOGPATH}/fbref_scraper.log'),
        logging.StreamHandler()
    ]
)

def create_session():
    """Create a requests session with enhanced headers and retry strategy"""
    session = requests.Session()
    
    # Enhanced retry strategy
    retry_strategy = Retry(
        total=5,  # Increased from 3
        backoff_factor=10,  # Increased from 5
        status_forcelist=[403, 429, 500, 502, 503, 504],  # Added 403
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    # Disable SSL verification for corporate environments
    # session.verify = False
    # import urllib3
    # urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # Enhanced headers to appear more like a real browser
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0',
    })
    
    return session

def safe_read_html(html_content):
    """Safely read HTML content using StringIO to avoid FutureWarning"""
    return pd.read_html(StringIO(html_content))  # Added verify=False

def get_fixture(session, link, year):
    """Get fixture data from FBRef URL"""
    try:
        logging.info(f"Fetching fixtures for {year} season from {link}")
        
        response = session.get(link)
        response.raise_for_status()
        
        time.sleep(random.uniform(3, 5))
        
        fixtures = safe_read_html(response.text)
        fixtures = fixtures[0]
        fixtures = fixtures.dropna(how='all').reset_index(drop=True)
        fixtures['Game'] = fixtures.Home + ' vs ' + fixtures.Away
        fixtures['Season'] = year
        
        return fixtures
    
    except Exception as e:
        logging.error(f"Error fetching fixtures for {year}: {str(e)}")
        raise

def should_update_data(year, fixtures, file_path):
    """Determine if data needs to be updated based on new completed matches"""
    if not os.path.exists(file_path):
        return True
        
    try:
        with open(file_path, 'rb') as f:
            existing_data = pickle.load(f)
        
        # Get completed match IDs from fixtures
        completed_matches = set(
            idx for idx, row in fixtures.iterrows() 
            if row['Match Report'] == 'Match Report'
        )
        
        # Get existing match IDs
        existing_matches = set(
            i for i, data in enumerate(existing_data) 
            if data is not None
        )
        
        logging.info(f"Completed matches for {year}: {len(completed_matches)}")
        logging.info(f"Existing matches in pickle for {year}: {len(existing_matches)}")
        
        # Return True if there are any new completed matches
        return not completed_matches.issubset(existing_matches)
                
    except Exception as e:
        logging.error(f"Error in should_update_data for {year}: {str(e)}")
        return True
    

def scrape_func(session, link, year):
    fixtures = get_fixture(session, link, year)
    
    yearly_file_info = f'all_game_detail_{year}.pkl'
    file_path = f'Data And Scripts/Data/{yearly_file_info}'
    
    # Load existing pickle file
    existing_info = []
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            existing_info = pickle.load(f)
            
    # Get length of actual data (non-None entries)
    current_length = len([x for x in existing_info if x is not None])
    logging.info(f"Current number of matches in pickle: {current_length}")
    
    # Get indices of completed matches from new fixtures that are BEYOND our current data
    to_scrape = [
        idx for idx, row in fixtures.iterrows() 
        if row['Match Report'] == 'Match Report' and idx >= current_length
    ]
    
    if not to_scrape:
        logging.info(f"No new matches to scrape for {year}")
        return fixtures, existing_info
        
    logging.info(f"Need to scrape {len(to_scrape)} new matches for {year}")
    
    # Clear any None padding to start fresh
    existing_info = [x for x in existing_info if x is not None]
    
    # Extend existing_info to accommodate new indices
    while len(existing_info) < max(to_scrape) + 1:
        existing_info.append(None)
    
    # Get match report links
    response = session.get(link)
    soup = BeautifulSoup(response.text, "html.parser")
    links_dict = {}
    
    for row in soup.find_all('tr'):
        for cell in row.find_all('td'):
            for link in cell.find_all('a', href=True):
                if link.get_text() == 'Match Report':
                    idx = len(links_dict)
                    links_dict[idx] = link['href']
    
    # Scrape new matches
    for i, idx in enumerate(to_scrape, 1):
        try:
            if idx in links_dict:
                link = 'https://fbref.com' + links_dict[idx]
                logging.info(f"Scraping match {i}/{len(to_scrape)} for {year} (index {idx})")
                
                response = session.get(link)
                response.raise_for_status()
                
                game_detail = safe_read_html(response.text)
                existing_info[idx] = game_detail
                
                # Save after each scrape
                with open(file_path, 'wb') as f:
                    pickle.dump(existing_info, f)
                
                logging.info(f"Successfully scraped match {idx}")
                time.sleep(random.uniform(3, 6))
                
        except Exception as e:
            logging.error(f"Error scraping match {idx}: {str(e)}")
            
    return fixtures, existing_info

def shots_clean(df, game, season):
    """Clean shots data"""
    try:
        df = df.copy()
        
        # Clean column names, removing Unnamed prefixes and level_0
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[1] if 'Unnamed:' in col[0] else col[0] 
                         for col in df.columns]
        
        df = df.dropna(how='all')
        df = df.iloc[:,:-4]  # Remove last 4 columns
        
        # Handle any remaining duplicate column names
        df.columns = [f"{col}_{i}" if df.columns.tolist().count(col) > 1 else col 
                     for i, col in enumerate(df.columns)]
        
        df = df.rename(columns={'Squad': 'team'})
        df['game'] = game
        df['Season'] = int(season)  # Ensure Season is int
        
        df = df.reset_index(drop=True)
        df = df.drop_duplicates()
        
        return df
    except Exception as e:
        logging.error(f"Error in shots_clean for game {game}: {str(e)}")
        raise

def line_up_clean(df, game, season):
    """Clean lineup data"""
    df = df.copy()
    bench_loc = df[df.iloc[:,0] == 'Bench'].index[0]
    clean = df.iloc[:bench_loc, [1]]
    clean['starting_11'] = 'Y'
    clean['team'] = clean.columns[0].split('(')[0].strip()
    clean['game'] = game
    clean.columns.values[0] = 'Player'
    clean['Season'] = int(season)  # Ensure Season is int
    return clean

def summary_clean(df, game='', team='', season='', drop=False):
    """Clean summary statistics data"""
    df = df.copy()
    
    # Clean column names and remove any existing _x or _y suffixes
    df.columns = [f'{x[0]} - {x[1]}' if 'Unnamed' not in x[0] else x[1] 
                 for x in df.columns]
    df.columns = [col.replace('_x', '').replace('_y', '') for col in df.columns]
    
    # Define core columns that should always be present
    subset_cols = ['#', 'Nation', 'Pos', 'Age']
    
    # Only process rows that have the core player information
    df = df.dropna(subset=[x for x in subset_cols if x in df.columns])
    
    if drop:
        # When merging additional stats, drop the identifier columns
        df = df.drop([x for x in subset_cols if x in df.columns] + ['Min'], axis=1)
        
        # Rename any duplicate columns to avoid _x, _y suffixes in merge
        rename_dict = {}
        for col in df.columns:
            if col.endswith('_x') or col.endswith('_y'):
                new_col = col[:-2]  # Remove the suffix
                rename_dict[col] = new_col
        if rename_dict:
            df = df.rename(columns=rename_dict)
            
    else:
        # For the main player data, clean position and add metadata
        df.Pos = df.Pos.str.split(',').str[0]
        df['team'] = team
        df['game'] = game
        df['Season'] = int(season)
        
        # Ensure Player column exists and is unique index for merging
        if 'Player' in df.columns:
            df = df.drop_duplicates(subset=['Player'])
    
    return df

def process_new_games(fixtures_df, detailed_info, year, start_idx):
    """Process games from pickle files into DataFrames"""
    line_ups = []
    summaries = []
    shot_info = []
    
    logging.info(f"\nProcessing year {year}")
    logging.info(f"Fixtures dataframe shape: {fixtures_df.shape}")
    logging.info(f"Total games in detailed info: {len(detailed_info)}")
    logging.info(f"Starting from index: {start_idx}")
    
    games_processed = 0
    for idx, game_info in enumerate(detailed_info):
        try:
            if game_info is None:
                continue
                
            # Get game details from fixtures
            if idx >= len(fixtures_df):
                logging.warning(f"Index {idx} out of range for fixtures dataframe")
                continue
                
            game = fixtures_df.loc[idx, 'Game']
            home = fixtures_df.loc[idx, 'Home']
            away = fixtures_df.loc[idx, 'Away']
            note = fixtures_df.loc[idx, 'Notes']
            
            # Skip games with notes
            if not pd.isnull(note):
                continue
                
            # Process lineups
            try:
                if len(game_info) >= 2:
                    home_line_up = game_info[0]
                    away_line_up = game_info[1]
                    
                    if isinstance(home_line_up, pd.DataFrame) and isinstance(away_line_up, pd.DataFrame):
                        home_lineup_clean = line_up_clean(home_line_up, game, year)
                        away_lineup_clean = line_up_clean(away_line_up, game, year)
                        line_ups.extend([home_lineup_clean, away_lineup_clean])
                    else:
                        logging.warning(f"Invalid lineup data format for game {game}")
            except Exception as e:
                logging.error(f"Error processing lineups for game {game}: {str(e)}")
                
            # Process summaries
            try:
                if len(game_info) >= 11:
                    home_team_summary = game_info[3]
                    away_team_summary = game_info[10]
                    
                    if isinstance(home_team_summary, pd.DataFrame) and isinstance(away_team_summary, pd.DataFrame):
                        home_summary = summary_clean(home_team_summary, game, home, year)
                        away_summary = summary_clean(away_team_summary, game, away, year)
                        
                        # Process additional stats
                        for x in range(1, 7):
                            if len(game_info) >= (10 + x) and len(game_info) >= (3 + x):
                                new_home_summary = summary_clean(game_info[3 + x], drop=True)
                                new_away_summary = summary_clean(game_info[10 + x], drop=True)
                                
                                # Handle goalkeeper passing column rename
                                if 'Passes - Att (GK)' in new_home_summary.columns:
                                    new_home_summary = new_home_summary.rename(columns={'Passes - Att (GK)': 'Passes - Att'})
                                if 'Passes - Att (GK)' in new_away_summary.columns:
                                    new_away_summary = new_away_summary.rename(columns={'Passes - Att (GK)': 'Passes - Att'})

                                # Ensure all column names are strings and properly escaped
                                new_home_summary.columns = [str(col) for col in new_home_summary.columns]
                                new_away_summary.columns = [str(col) for col in new_away_summary.columns]
                                
                                home_summary = home_summary.merge(
                                    new_home_summary, 
                                    how='left', 
                                    on='Player',
                                    suffixes=('', '_drop')
                                )
                                away_summary = away_summary.merge(
                                    new_away_summary,
                                    how='left',
                                    on='Player',
                                    suffixes=('', '_drop')
                                )
                                
                                # Drop any columns with _drop suffix
                                home_summary = home_summary.loc[:, ~home_summary.columns.str.endswith('_drop')]
                                away_summary = away_summary.loc[:, ~away_summary.columns.str.endswith('_drop')]
                        
                        summaries.extend([home_summary, away_summary])
                    else:
                        logging.warning(f"Invalid summary data format for game {game}")
            except Exception as e:
                logging.error(f"Error processing summaries for game {game}: {str(e)}")
                
            # Process shots
            try:
                if len(game_info) >= 19:
                    shots = game_info[-3]
                    if isinstance(shots, pd.DataFrame):
                        clean_shots = shots_clean(shots, game, year)
                        if not clean_shots.empty:
                            shot_info.append(clean_shots)
                    else:
                        logging.warning(f"Invalid shots data format for game {game}")
            except Exception as e:
                logging.error(f"Error processing shots for game {game}: {str(e)}")
            
            games_processed += 1
            
        except Exception as e:
            logging.error(f"Unexpected error processing game at index {idx}: {str(e)}")
            continue
    
    logging.info(f"Successfully processed {games_processed} games for year {year}")
    
    # Log summary of processed data
    logging.info(f"Generated {len(line_ups)} lineup DataFrames")
    logging.info(f"Generated {len(summaries)} summary DataFrames")
    logging.info(f"Generated {len(shot_info)} shot DataFrames")
    
    if line_ups and summaries and shot_info:
        # Log sample of first DataFrame from each category
        logging.info("\nSample of processed data:")
        logging.info(f"\nLineup columns: {line_ups[0].columns.tolist()}")
        logging.info(f"Summary columns: {summaries[0].columns.tolist()}")
        logging.info(f"Shot columns: {shot_info[0].columns.tolist()}")
    
    return line_ups, summaries, shot_info

def verify_season_completeness(data_dir, year, detailed_info=None):
    """Verify if the season data has been fully processed"""
    print(f"---- Verifying completeness for {year} ----")
    
    pkl_path = os.path.join(data_dir, f'all_game_detail_{year}.pkl')
    
    # Load pickle file if not provided
    if detailed_info is None:
        if not os.path.exists(pkl_path):
            print(f"No pickle file found for {year}")
            return True
            
        try:
            with open(pkl_path, 'rb') as f:
                detailed_info = pickle.load(f)
        except Exception as e:
            logging.error(f"Error loading pickle file for {year}: {str(e)}")
            return True
    
    # Count total non-None entries in detailed_info
    total_games = len([game for game in detailed_info if game is not None])
    print(f"Total games found in pickle for {year}: {total_games}")
    
    if total_games == 0:
        return True
        
    # Check if CSV files exist
    csv_files = ['lineups.csv', 'all_players.csv', 'shots.csv']
    csv_paths = {file: os.path.join(data_dir, file) for file in csv_files}
    
    # If any CSV doesn't exist, we need to process
    if not all(os.path.exists(path) for path in csv_paths.values()):
        missing_files = [file for file, path in csv_paths.items() if not os.path.exists(path)]
        print(f"Missing CSV files: {missing_files}")
        return True
        
    # If files exist, check game counts
    try:
        dfs = {
            name: pd.read_csv(path, low_memory=False) 
            for name, path in csv_paths.items()
        }
        
        # Filter by year and get game counts
        game_counts = {
            name: df[df['Season'] == int(year)]['game'].nunique()
            for name, df in dfs.items()
        }
        
        print(f"Game counts for {year}:")
        for name, count in game_counts.items():
            print(f"{name.split('.')[0]}: {count}")
            
        # Compare counts to pickle games
        counts_match = all(count == total_games for count in game_counts.values())
        
        if not counts_match:
            logging.info(
                f"Mismatch in game counts for {year}. "
                f"Pickle games: {total_games}, "
                f"CSV games: {game_counts}"
            )
            
        return not counts_match
        
    except Exception as e:
        logging.error(f"Error checking CSV completeness for {year}: {str(e)}")
        return True        

def main():
    session = create_session()
    
    yearly_data = {
        '2024': 'https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures',
        '2023': 'https://fbref.com/en/comps/9/2023-2024/schedule/2023-2024-Premier-League-Scores-and-Fixtures',
        '2022': 'https://fbref.com/en/comps/9/2022-2023/schedule/2022-2023-Premier-League-Scores-and-Fixtures',
        '2021': 'https://fbref.com/en/comps/9/2021-2022/schedule/2021-2022-Premier-League-Scores-and-Fixtures',
        '2020': 'https://fbref.com/en/comps/9/2020-2021/schedule/2020-2021-Premier-League-Scores-and-Fixtures',
        '2019': 'https://fbref.com/en/comps/9/2019-2020/schedule/2019-2020-Premier-League-Scores-and-Fixtures',
        '2018': 'https://fbref.com/en/comps/9/2018-2019/schedule/2018-2019-Premier-League-Scores-and-Fixtures'
    }

    data_dir = 'Data And Scripts/Data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # --- SCRAPE FIXTURES AND CHECK FOR UPDATES ---
    all_fixtures = []
    seasons_needing_scraping = set()
    seasons_needing_processing = set()
    detailed_info_cache = {}  # Cache for pickle files

    all_fixtures = []
    seasons_needing_scraping = set()
    seasons_needing_processing = set()
    detailed_info_cache = {}  # Cache for pickle files

    # Process each season
    for year, link in yearly_data.items():
        try:
            fixtures = get_fixture(session, link, year)
            if fixtures is not None and not fixtures.empty:
                all_fixtures.append(fixtures)
                logging.info(f"Successfully added fixtures for {year}")
                
                # Check if we need to scrape new games
                yearly_file_info = f'all_game_detail_{int(year)}.pkl'
                file_path = os.path.join(data_dir, yearly_file_info)
                
                if should_update_data(year, fixtures, file_path):
                    seasons_needing_scraping.add(year)
                
                # Load and cache pickle file if it exists
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        detailed_info = pickle.load(f)
                    detailed_info_cache[year] = detailed_info
                    
                    # Check if season needs processing
                    if verify_season_completeness(data_dir, year, detailed_info):
                        seasons_needing_processing.add(year)
            else:
                logging.warning(f"No fixtures data obtained for {year}")
        
        except Exception as e:
            logging.error(f"Error processing {year} season: {str(e)}")
            continue
    # Save fixtures
    fixtures_df = pd.concat(all_fixtures, ignore_index=True)
    fixtures_df['Season'] = fixtures_df['Season'].astype(int)
    fixtures_df.to_csv(os.path.join(data_dir, 'fixtures.csv'), index=False)

    # Save team goals in a game
    fixtures_df[['homeGoals', 'awayGoals']] = fixtures_df['Score'].str.split('–', expand=True)
    home = fixtures_df[['Wk', 'Date', 'Home', 'Game', 'Season', 'homeGoals']].rename(columns={'Home': 'Team', 'homeGoals': 'teamGoals'})
    away = fixtures_df[['Wk', 'Date', 'Away', 'Game', 'Season', 'awayGoals']].rename(columns={'Away': 'Team', 'awayGoals': 'teamGoals'})

    gameGoals = pd.concat([home, away], ignore_index=True)
    gameGoals = gameGoals[gameGoals['teamGoals'].notna()]

    gameGoals['teamGoals'] = gameGoals['teamGoals'].astype(int)
    gameGoals['Season'] = gameGoals['Season'].astype(int)
    gameGoals.to_csv(os.path.join(data_dir, 'game_goals.csv'), index=False)

    # --- SCRAPE NEW GAMES FOR SEASONS THAT NEED IT ---
    for year in seasons_needing_scraping:
        try:
            logging.info(f"Scraping new games for {year}")
            _, detailed_info = scrape_func(session, yearly_data[year], year)
            detailed_info_cache[year] = detailed_info
            
            # Save updated pickle file
            with open(os.path.join(data_dir, f'all_game_detail_{year}.pkl'), 'wb') as f:
                pickle.dump(detailed_info, f)
                
        except Exception as e:
            logging.error(f"Error scraping new games for {year}: {str(e)}")

# --- PROCESS ALL SEASONS THAT NEED PROCESSING ---
    # --- PROCESS ALL SEASONS THAT NEED PROCESSING ---
    if seasons_needing_processing:
        start = time.time()
        logging.info(f"Processing seasons: {sorted(seasons_needing_processing)}")
        
        # Load existing data if available
        existing_data = {}
        for file in ['lineups.csv', 'all_players.csv', 'shots.csv']:
            path = os.path.join(data_dir, file)
            if os.path.exists(path):
                existing_data[file] = pd.read_csv(path, low_memory=False)
                existing_data[file]['Season'] = existing_data[file]['Season'].astype(int)
                # Remove any existing data for seasons we're reprocessing
                for year in seasons_needing_processing:
                    existing_data[file] = existing_data[file][
                        existing_data[file]['Season'] != int(year)
                    ]
                logging.info(f"\nLoaded existing {file} with shape after filtering: {existing_data[file].shape}")

        all_lineups = []
        all_players = []
        all_shots = []

        for year in sorted(seasons_needing_processing):
            try:
                detailed_info = detailed_info_cache.get(year)
                if detailed_info is None:
                    continue
                    
                # Get fixtures for this year
                year_fixtures_df = fixtures_df[fixtures_df['Season'] == int(year)].copy()
                year_fixtures_df.reset_index(drop=True, inplace=True)
                
                # Process the entire season
                lineups, summaries, shots = process_new_games(
                    year_fixtures_df, detailed_info, year, 0
                )
                
                all_lineups.extend(lineups)
                all_players.extend(summaries)
                all_shots.extend(shots)
                
            except Exception as e:
                logging.error(f"Error processing year {year}: {str(e)}")
        
        # Save all processed data
        try:
            if all_lineups:
                lineups_df = pd.concat(all_lineups, ignore_index=True)
                if 'lineups.csv' in existing_data:
                    old_data = existing_data['lineups.csv']
                    lineups_df = pd.concat([old_data, lineups_df], ignore_index=True)
                lineups_df = lineups_df.drop_duplicates()  # Add final deduplication
                lineups_df.to_csv(os.path.join(data_dir, 'lineups.csv'), index=False)
            
            if all_players:
                players_df = pd.concat(all_players, ignore_index=True)
                players_df = players_df.fillna(0)
                if 'all_players.csv' in existing_data:
                    old_data = existing_data['all_players.csv']
                    players_df = pd.concat([old_data, players_df], ignore_index=True)
                players_df = players_df.drop_duplicates()  # Add final deduplication
                players_df.to_csv(os.path.join(data_dir, 'all_players.csv'), index=False)
            
            if all_shots:
                shots_df = pd.concat(all_shots, ignore_index=True)
                if 'shots.csv' in existing_data:
                    old_data = existing_data['shots.csv']
                    shots_df = pd.concat([old_data, shots_df], ignore_index=True)
                shots_df = shots_df.drop_duplicates()  # Add final deduplication
                shots_df.to_csv(os.path.join(data_dir, 'shots.csv'), index=False)
                
        except Exception as e:
            logging.error(f"Error saving processed data: {str(e)}")
            logging.error("Full traceback:", exc_info=True)
        
        end = time.time()
        logging.info(f"Data processing complete in: {(end - start)/60:.2f} minutes")
    
    else:
        logging.info("No seasons require processing")
    
    print("\nFinal verification of all seasons:")
    for year in yearly_data.keys():
        # Load pickle file
        pkl_path = os.path.join(data_dir, f'all_game_detail_{year}.pkl')
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                detailed_info = pickle.load(f)
            verify_season_completeness(data_dir, year, detailed_info)
            

if __name__ == "__main__":
    main()