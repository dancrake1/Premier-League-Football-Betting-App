import streamlit as st
import pandas as pd
import os
from typing import List
import logging

class FixtureManager:
    def __init__(self, fixtures_df: pd.DataFrame, data_loader):
        self.fixtures_df = fixtures_df
        self.data_loader = data_loader
        self.current_season = fixtures_df['Season'].max()
        self.season_fixtures = fixtures_df[fixtures_df['Season'] == self.current_season]
        self.gameweek_status = self._analyze_gameweeks()
        self.active_gameweeks = self._get_current_gameweek()
        
        # Initialize lineup cache in session state if it doesn't exist
        if 'lineup_cache' not in st.session_state:
            st.session_state.lineup_cache = {}

    def _analyze_gameweeks(self) -> dict:
        """Analyze gameweeks to determine their status"""
        try:
            # Group by gameweek and analyze each
            gameweek_status = {}
            for gw in self.season_fixtures['Wk'].unique():
                gw_fixtures = self.season_fixtures[self.season_fixtures['Wk'] == gw]
                total_matches = len(gw_fixtures)
                played_matches = gw_fixtures['Score'].notna().sum()
                
                gameweek_status[gw] = {
                    'total_matches': total_matches,
                    'played_matches': played_matches,
                    'is_complete': total_matches == played_matches and total_matches == 10,
                    'has_unplayed': played_matches < total_matches,
                    'first_match_date': gw_fixtures['Date'].min(),
                    'last_match_date': gw_fixtures['Date'].max()
                }
            
            return gameweek_status
        except Exception as e:
            logging.error(f"Error analyzing gameweeks: {str(e)}")
            return {}

    def _get_current_gameweek(self) -> List[int]:
        """Determine active gameweeks based on comprehensive analysis"""
        try:
            today = pd.Timestamp.now()
            
            if not self.gameweek_status:
                return [1]
                
            active_gameweeks = set()
            
            # Find earliest gameweek with unplayed matches
            earliest_unplayed = None
            for gw in sorted(self.gameweek_status.keys()):
                status = self.gameweek_status[gw]
                if status['has_unplayed']:
                    earliest_unplayed = gw
                    break
            
            # Find current or next active gameweek
            current_gw = None
            for gw, status in self.gameweek_status.items():
                if (status['first_match_date'] <= today <= status['last_match_date'] or 
                    (today < status['first_match_date'] and not status['is_complete'])):
                    current_gw = gw
                    break
            
            # If no current gameweek found, find next upcoming
            if current_gw is None:
                for gw, status in sorted(self.gameweek_status.items()):
                    if today < status['first_match_date'] and not status['is_complete']:
                        current_gw = gw
                        break
            
            # Build final set of gameweeks to display
            if earliest_unplayed is not None:
                active_gameweeks.add(earliest_unplayed)
                
            if current_gw is not None and current_gw != earliest_unplayed:
                active_gameweeks.add(current_gw)
                
                # Include next gameweek if current is almost complete
                if current_gw in self.gameweek_status:
                    current_status = self.gameweek_status[current_gw]
                    if current_status['played_matches'] >= 8:
                        next_gw = current_gw + 1
                        if next_gw in self.gameweek_status and not self.gameweek_status[next_gw]['is_complete']:
                            active_gameweeks.add(next_gw)
            
            # If no gameweeks selected yet, find next unplayed
            if not active_gameweeks:
                for gw, status in sorted(self.gameweek_status.items()):
                    if not status['is_complete']:
                        active_gameweeks.add(gw)
                        break
            
            result = sorted(list(active_gameweeks))
            logging.info(f"Selected gameweeks {result} for display")
            return result
            
        except Exception as e:
            logging.error(f"Error determining gameweeks: {str(e)}")
            return [1]

    def get_available_gameweeks(self, show_unplayed_only: bool) -> list:
        """Get available gameweeks based on filter settings"""
        if show_unplayed_only:
            return [gw for gw, status in self.gameweek_status.items() 
                   if status['has_unplayed']]
        return sorted(self.gameweek_status.keys())

    def get_default_gameweeks(self, available_gw: list) -> list:
        """Get default gameweeks ensuring they're available"""
        default_gw = [gw for gw in self.active_gameweeks if gw in available_gw]
        if not default_gw and available_gw:
            default_gw = [min(available_gw)]
        return default_gw

    def format_gameweek(self, gw: int) -> str:
        """Format gameweek display string"""
        status = self.gameweek_status[gw]
        return f"GW{gw} ({status['played_matches']}/{status['total_matches']} played)"

    def filter_fixtures(self, selected_gw: list, show_unplayed_only: bool, 
                    show_completed: bool) -> pd.DataFrame:
        """Apply all filters to fixtures"""
        # Clear lineup cache whenever filters change
        self.clear_lineup_cache()
        
        filtered_df = self.fixtures_df[
            (self.fixtures_df['Wk'].isin(selected_gw)) & 
            (self.fixtures_df['Season'] == self.current_season)
        ]
        
        if show_unplayed_only or not show_completed:
            filtered_df = filtered_df[filtered_df['Score'].isna()]
            
        return filtered_df
    
    def has_lineups(self, gameweek, game) -> bool:
        """Check if lineup data exists for a gameweek"""
        try:
            # Use the DATA_DIR constant instead of data_loader.data_dir
            DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data And Scripts/Data")
            lineup_path = os.path.join(DATA_DIR, f'sky_lineups_gw{int(gameweek)}.csv')
            
            # First check if file exists at all
            if not os.path.exists(lineup_path):
                return False
                
            # Then check if this specific game is in the file
            lineups = pd.read_csv(lineup_path)
            games = lineups['game'].unique()
            return game in games
        except Exception as e:
            logging.error(f"Error checking lineup availability for game {game} in GW{gameweek}: {str(e)}")
            return False  # Return False on any error to gracefully handle issues
    
    
    def create_fixture_card(self, fixture: pd.Series, odds_manager) -> None:
        """Create a fixture card with all relevant information"""
        try:
            st.markdown(f"### {fixture['Home']} vs {fixture['Away']}")
            
            if pd.notna(fixture['Notes']):
                st.markdown(f"*{fixture['Notes']}*")
            
            match_date = fixture['Date'].strftime('%d %B')
            match_time = fixture['Time'] if pd.notna(fixture['Time']) else 'TBD'
            st.write(f"*{match_date}, {match_time}*")
            
            if pd.notna(fixture['Score']):
                st.markdown(f"**Score: {fixture['Score']}**")
            
            # Show lineup availability indicator
            gameweek = fixture['Wk']
            # Needs to be fixed when lineups not available
            if self.has_lineups(gameweek, fixture['Game']):
                st.caption("📋 Lineups Available")
            else:
                st.caption("⏳ Lineups Not Available Yet")
            
            lineup_tab, goals_tab, cards_tab = st.tabs(["Lineups", "Goals Markets", "Cards Markets"])
            
            with lineup_tab:
                self._display_lineups(fixture)
            
            with goals_tab:
                odds_manager.display_odds_card(fixture, 'goals')
            
            with cards_tab:
                odds_manager.display_odds_card(fixture, 'cards')
        except Exception as e:
            # Fallback display if there's an error
            st.markdown(f"### {fixture['Home']} vs {fixture['Away']}")
            st.write(f"Date: {fixture['Date']}")
            st.caption("⚠️ Error displaying complete fixture details")
            logging.error(f"Error displaying fixture card: {str(e)}")

    def _display_lineups(self, fixture: pd.Series) -> None:
        """Display lineups for both teams using cached data"""
        try:
            gameweek = fixture['Wk']
            
            # First check if lineup file exists at all
            if not self.has_lineups(gameweek, fixture['Game']):
                st.info("Lineups not yet available. Use the 'Update Lineups Only' button to fetch the latest lineups.")
                return
            
            # Create cache keys
            home_cache_key = f"{fixture['Home']}_{gameweek}"
            away_cache_key = f"{fixture['Away']}_{gameweek}"
            
            # Get or create home lineup cache
            if home_cache_key not in st.session_state.lineup_cache:
                home_lineup = self.data_loader.load_lineup(fixture['Home'], gameweek)
                st.session_state.lineup_cache[home_cache_key] = home_lineup
            else:
                home_lineup = st.session_state.lineup_cache[home_cache_key]

            # Get or create away lineup cache
            if away_cache_key not in st.session_state.lineup_cache:
                away_lineup = self.data_loader.load_lineup(fixture['Away'], gameweek)
                st.session_state.lineup_cache[away_cache_key] = away_lineup
            else:
                away_lineup = st.session_state.lineup_cache[away_cache_key]

            # Display home team lineup
            with st.expander(f"{fixture['Home']} Lineup"):
                if home_lineup is not None and not home_lineup.empty:
                    for _, player in home_lineup.iterrows():
                        st.write(f"• {player['Player']}")
                else:
                    st.write(f"No lineup available for {fixture['Home']}")
            
            # Display away team lineup
            with st.expander(f"{fixture['Away']} Lineup"):
                if away_lineup is not None and not away_lineup.empty:
                    for _, player in away_lineup.iterrows():
                        st.write(f"• {player['Player']}")
                else:
                    st.write(f"No lineup available for {fixture['Away']}")
        except Exception as e:
            st.error(f"Error displaying lineups: {str(e)}")
            logging.error(f"Error displaying lineups: {str(e)}")

    def clear_lineup_cache(self):
        """Clear the lineup cache when needed (e.g., after running scrapers)"""
        if 'lineup_cache' in st.session_state:
            st.session_state.lineup_cache = {}