import os
import pandas as pd
import streamlit as st
import subprocess
import sys
import logging

class DataLoader:
    def __init__(self, data_dir: str, script_dir: str):
        self.DATA_DIR = data_dir
        self.SCRIPT_DIR = script_dir

    def initialize_data(self) -> bool:
        """Initialize data by running scraper-runner first"""
        try:
            logging.info("Initializing data with FBRef scraper...")
            scraper_path = os.path.join(self.SCRIPT_DIR, 'scraper-runner.py')
            
            if not os.path.exists(scraper_path):
                msg = f"Scraper runner not found at: {scraper_path}"
                logging.error(msg)
                st.error(msg)
                return False
                
            result = subprocess.run(
                [sys.executable, scraper_path], 
                check=True,
                capture_output=True,
                text=True
            )
            logging.info("Initial data scraping completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            msg = f"Error initializing data: {str(e)}\nOutput: {e.output}"
            logging.error(msg)
            st.error(msg)
            return False

    def validate_data_directory(self) -> bool:
        """Validate that required directories and files exist"""
        if not os.path.exists(self.DATA_DIR):
            msg = f"Data directory not found at: {self.DATA_DIR}"
            logging.error(msg)
            st.error(msg)
            return False
        return True

    def load_fixtures(self) -> pd.DataFrame:
        """Load fixtures data and cache it in session state"""
        if st.session_state.fixtures_data is None:
            try:
                fixtures_path = os.path.join(self.DATA_DIR, 'fixtures.csv')
                if not os.path.exists(fixtures_path):
                    msg = f"Fixtures file not found at: {fixtures_path}"
                    logging.error(msg)
                    st.error(msg)
                    return pd.DataFrame()
                    
                df = pd.read_csv(fixtures_path)
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                st.session_state.fixtures_data = df.sort_values(['Date'])
                
            except Exception as e:
                logging.error(f"Error loading fixtures: {str(e)}")
                st.error(f"Error loading fixtures: {str(e)}")
                return pd.DataFrame()
        
        return st.session_state.fixtures_data

    def load_lineup(self, team: str, gameweek: int) -> pd.DataFrame:
        """Load lineup data for a specific team and gameweek"""
        try:
            logging.info(f"Attempting to load lineup for team: {team}, gameweek: {gameweek}")
            
            lineup_file = os.path.join(self.DATA_DIR, f'sky_lineups_gw{int(gameweek)}.csv')
            if os.path.exists(lineup_file):
                logging.info(f"Found sky_lineups file: {lineup_file}")
                df = pd.read_csv(lineup_file)
                
                unique_teams = df['team'].unique()
                logging.info(f"Teams in lineup file: {unique_teams}")
                logging.info(f"Looking for team: {team}")
                
                team_data = df[df['team'] == team]
                if not team_data.empty:
                    logging.info(f"Found {len(team_data)} players for {team}")
                    return team_data
                    
                logging.warning(f"No lineup found for team {team}")
                return None
                
            else:
                logging.info(f"Sky lineups file not found: {lineup_file}")
                return None
            
        except Exception as e:
            logging.error(f"Error loading lineup data: {str(e)}")
            st.error(f"Error loading lineup data: {str(e)}")
            return None

    def run_scrapers(self) -> None:
        """Run both scrapers using scraper-runner and clear lineup cache"""
        try:
            st.info("Running scrapers...")
            scraper_path = os.path.join(self.SCRIPT_DIR, 'scraper-runner.py')
            
            if not os.path.exists(scraper_path):
                msg = f"Scraper runner not found at: {scraper_path}"
                logging.error(msg)
                st.error(msg)
                return
                
            subprocess.run(['python', scraper_path], check=True)
            
            # Clear the lineup cache after running scrapers
            if 'lineup_cache' in st.session_state:
                st.session_state.lineup_cache = {}
                
            st.success("Data updated successfully!")
        except subprocess.CalledProcessError as e:
            msg = f"Error updating data: {str(e)}"
            logging.error(msg)
            st.error(msg)