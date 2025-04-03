import sys
import os
import subprocess
import pandas as pd
from datetime import datetime
import logging

LOGPATH = '/Users/danielcrake/Desktop/Football Betting 2025/logs'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{LOGPATH}/scraper_runner.log'),
        logging.StreamHandler()
    ]
)

def run_fbref_scraper():
    try:
        logging.info("Starting FBRef scraper...")
        result = subprocess.run(
            [sys.executable, "Data And Scripts/fbref-scraper.py"], 
            check=True,
            capture_output=True,
            text=True
        )
        logging.info("FBRef scraper completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"FBRef scraper failed: {e}\nOutput: {e.output}")
        return False

def run_sky_scraper(date_str, season, gameweek):
    try:
        logging.info(f"Starting Sky Sports scraper for GW{gameweek}...")
        os.makedirs('Data And Scripts/Data', exist_ok=True)
        
        result = subprocess.run([
            sys.executable, 
            "Data And Scripts/skysports-scraper.py",
            "--date", date_str,
            "--season", str(season),
            "--gameweek", str(gameweek)
        ], check=True, capture_output=True, text=True)
        
        logging.info("Sky Sports scraper completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Sky Sports scraper failed: {e}\nOutput: {e.output}")
        return False

def get_current_gameweek():
    try:
        fixtures = pd.read_csv('Data And Scripts/Data/fixtures.csv')
        current_season = fixtures['Season'].max()
        fixtures = fixtures[fixtures.Season == current_season]
        fixtures['Date'] = pd.to_datetime(fixtures['Date'])
        
        fixtures = fixtures[fixtures.Score.isna()].sort_values('Date')
        if len(fixtures) > 0:
            return int(fixtures.iloc[0]['Wk'])
        
        fixtures = fixtures[fixtures.Season == current_season].sort_values('Wk', ascending=False)
        if len(fixtures) > 0:
            return int(fixtures.iloc[0]['Wk'])
        return None
    except Exception as e:
        logging.error(f"Error determining gameweek: {e}")
        return None

def get_next_matchday():
    try:
        fixtures = pd.read_csv('Data And Scripts/Data/fixtures.csv')
        current_season = fixtures['Season'].max()
        fixtures = fixtures[fixtures.Season == current_season]
        fixtures = fixtures[fixtures.Score.isna()].sort_values('Date')
        if len(fixtures) > 0:
            date = pd.to_datetime(fixtures.iloc[0]['Date'])
            return date.strftime("%d-%B-%Y")
        return None
    except Exception as e:
        logging.error(f"Error determining next matchday: {e}")
        return None

def main():
    logging.info("Starting scraper runner sequence...")
    
    # Step 1: Always run FBRef scraper first
    if not run_fbref_scraper():
        logging.error("FBRef scraper failed - stopping process")
        sys.exit(1)
    
    # Step 2: Only proceed with Sky scraper if FBRef was successful
    try:
        fixtures = pd.read_csv('Data And Scripts/Data/fixtures.csv')
        current_season = fixtures['Season'].max()
        
        gameweek = get_current_gameweek()
        next_matchday = get_next_matchday()
        
        if gameweek and next_matchday:
            if not run_sky_scraper(next_matchday, current_season, gameweek):
                logging.error("Sky Sports scraper failed")
                # Don't exit here as odds can still run
        
    except Exception as e:
        logging.error(f"Error in main process: {e}")
        # Continue to allow odds to run
    
    logging.info("Scraper sequence completed")

if __name__ == "__main__":
    main()