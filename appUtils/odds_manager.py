import os
import pandas as pd
import streamlit as st
import logging
from typing import Optional, Tuple
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from .utils import save_historical_odds
from .betfair_api import BetfairAPI
import numpy as np
import time

# At the top of prediction_system.py
from .logging_setup import setup_logging

# Set up logging
odds_logger = setup_logging('odds_manager')

class OddsManager:
    def __init__(self):
        self.api = BetfairAPI(
            username=st.secrets["betfair_username"],
            password=st.secrets["betfair_password"],
            app_key=st.secrets["betfair_app_key"],
            cert_path=st.secrets["betfair_cert_path"],
            key_path=st.secrets["betfair_key_path"]
        )
        self.DATA_DIR = os.path.join("Data And Scripts", "Data")
        self._last_update_attempt = None
        self.scheduler = None
        self.UPDATE_INTERVAL = 400  # seconds
        self.HEALTH_THRESHOLD = self.UPDATE_INTERVAL * 1.5

    def start_scheduler(self):
        """Start the background scheduler if it's not already running"""
        if not self.scheduler:
            self.scheduler = BackgroundScheduler()
            self.scheduler.add_job(self.update_odds, 'interval', seconds=self.UPDATE_INTERVAL)
            self.scheduler.start()
            odds_logger.info("Started background odds scheduler")

    def check_health(self) -> bool:
        """Check if odds data is being updated regularly"""
        try:
            if self._last_update_attempt is None:
                odds_logger.info("No last_odds_refresh in session state")
                return False
                
            now = pd.Timestamp.now()
            time_since_update = now - self._last_update_attempt 
            
            is_healthy = time_since_update.total_seconds() < self.HEALTH_THRESHOLD
            
            odds_logger.info(f"Last odds refresh: {self._last_update_attempt }")
            odds_logger.info(f"Time since update: {time_since_update}")
            odds_logger.info(f"Odds health status: {is_healthy}")
            
            return is_healthy
                
        except Exception as e:
            odds_logger.error(f"Error checking odds health: {str(e)}")
            return False

    def get_odds_status(self) -> dict:
        """Get current odds status and timing information"""
        try:
            now = pd.Timestamp.now()
            last_refresh = self._last_update_attempt 
            
            if last_refresh is None:
                return {
                    'is_healthy': False,
                    'last_refresh': None,
                    'minutes_ago': None,
                    'seconds_ago': None,
                    'needs_refresh': True
                }
            
            time_since_update = now - last_refresh
            seconds_ago = time_since_update.total_seconds()
            minutes_ago = int(seconds_ago / 60)
            is_healthy = seconds_ago < self.HEALTH_THRESHOLD
            needs_refresh = seconds_ago >= self.UPDATE_INTERVAL
            
            return {
                'is_healthy': is_healthy,
                'last_refresh': last_refresh,
                'minutes_ago': minutes_ago,
                'seconds_ago': seconds_ago,
                'needs_refresh': needs_refresh
            }
            
        except Exception as e:
            odds_logger.error(f"Error getting odds status: {str(e)}")
            return {
                'is_healthy': False,
                'last_refresh': None,
                'minutes_ago': None,
                'seconds_ago': None,
                'needs_refresh': True
            }

    def update_odds(self) -> bool:
        """Update odds data and return success status"""
        try:
            if not self.api:
                odds_logger.error("Betfair API not initialized")
                return False

            success = False  # Track if we successfully got any odds

            # Get goals odds
            goals_odds = self.api.get_odds(market_type='goals')
            if not goals_odds.empty:
                self.api.save_odds(goals_odds, 'goals')
                save_historical_odds(goals_odds, 'goals')
                success = True
            
            # Get cards odds
            cards_odds = self.api.get_odds(market_type='cards')
            if not cards_odds.empty:
                self.api.save_odds(cards_odds, 'cards')
                save_historical_odds(cards_odds, 'cards')
                success = True
                
            # Only update last refresh time if we actually got odds
            if success:
                self._last_update_attempt  = pd.Timestamp.now()
                odds_logger.info(f"Updated last_odds_refresh to {self._last_update_attempt }")
            
            return success
            
        except Exception as e:
            odds_logger.error(f"Failed to update odds: {str(e)}")
            return False

    def monitor_scheduler(self):
        """Check if scheduler is running and restart if needed"""
        if self.scheduler and not self.scheduler.running:
            odds_logger.warning("Scheduler stopped running, attempting restart")
            self.scheduler = None
            self.start_scheduler()

    def load_odds(self, market_type: str) -> pd.DataFrame:
        """Load current odds data from file"""
        try:
            filename = f'upcoming_odds_{market_type}.csv'
            filepath = os.path.join(self.DATA_DIR, filename)
            
            if not os.path.exists(filepath):
                return pd.DataFrame()
                
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
            df['over'] = pd.to_numeric(df['over'], errors='coerce')
            df['under'] = pd.to_numeric(df['under'], errors='coerce')

            df = df.dropna(subset=['timestamp'])
            
            return df
            
        except Exception as e:
            odds_logger.error(f"Error loading {market_type} odds: {str(e)}")
            return pd.DataFrame()

    def load_historical_odds(self, market_type: str) -> pd.DataFrame:
        """Load historical odds data with robust error handling"""
        try:
            filename = f'historical_odds_{market_type}.csv'
            filepath = os.path.join(self.DATA_DIR, filename)
            
            if not os.path.exists(filepath):
                odds_logger.warning(f"No historical odds file found at {filepath}")
                return pd.DataFrame()

            # Read CSV with error handling for malformed rows
            df = pd.read_csv(
                filepath,
                on_bad_lines='skip',  # Skip problematic lines instead of failing
                dtype={
                    'eventName': str,
                    'marketType': str,
                    'date': str,
                    'over': float,
                    'under': float
                }
            )
            
            # Convert timestamps with error handling
            df['timestamp'] = pd.to_datetime(
                df['timestamp'],
                format='mixed',  # Try multiple formats
                errors='coerce'  # Invalid timestamps become NaT
            )
            
            # Remove any rows with invalid timestamps
            df = df.dropna(subset=['timestamp'])
            
            # Ensure we have all required columns
            required_cols = ['timestamp', 'eventName', 'marketType', 'date', 'over', 'under']
            if not all(col in df.columns for col in required_cols):
                odds_logger.error(f"Missing required columns. Found: {df.columns.tolist()}")
                return pd.DataFrame()
                
            # Keep only the required columns
            df = df[required_cols]
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # odds_logger.info(f"Successfully loaded {len(df)} historical odds entries for {market_type}")
            return df
            
        except Exception as e:
            odds_logger.error(f"Error loading historical {market_type} odds: {str(e)}")
            return pd.DataFrame()

    def get_available_markets(self, odds_df: pd.DataFrame, fixture_name: str, 
                            market_type: str) -> list:
        """Get available markets for a specific fixture and market type"""
        fixture_odds = odds_df[odds_df['eventName'] == fixture_name]
        return sorted(fixture_odds['marketType'].unique())

    def display_odds_card(self, fixture: pd.Series, market_type: str) -> None:
        """Display odds information for a fixture"""
        try:
            odds_df = self.load_odds(market_type)
            historical_df = self.load_historical_odds(market_type)
            
            if odds_df.empty:
                st.write(f"No {market_type} odds available")
                return
                
            fixture_name = fixture['Home'] + ' vs ' + fixture['Away']
            available_markets = self.get_available_markets(odds_df, fixture_name, market_type)
            
            if not available_markets:
                st.write(f"No markets available for this fixture")
                return
                
            selected_market = st.selectbox(
                "Select Market",
                available_markets,
                format_func=lambda x: x.replace('OVER_UNDER_', 'Over/Under ').replace('_', '.'),
                key=f"{market_type}_{fixture_name}"
            )
            
            # Get predictions from session state
            prediction_manager = None
            if 'prediction_manager' in st.session_state and st.session_state.predictions_generated:
                prediction_manager = st.session_state.prediction_manager
            
            self._display_market_odds(
                odds_df, 
                historical_df, 
                fixture_name, 
                selected_market,
                prediction_manager
            )
            
        except Exception as e:
            odds_logger.error(f"Error displaying odds card: {str(e)}")
            st.error("Error displaying odds")

    def _display_market_odds(self, odds_df: pd.DataFrame, historical_df: pd.DataFrame, 
                            fixture_name: str, selected_market: str,
                            prediction_manager=None) -> None:
        """Display odds for a specific market with model comparison"""
        # Get threshold from market type and convert properly
        raw_threshold = selected_market.replace('OVER_UNDER_', '').replace('_', '.')
        threshold = float(raw_threshold) / 10 if len(raw_threshold) == 2 else float(raw_threshold)
        
        # odds_logger.info(f"Market: {selected_market}, Raw threshold: {raw_threshold}, Converted threshold: {threshold}")
        
        # Get model predictions and expected goals
        model_probabilities = None
        expected_goals = None
        
        if prediction_manager and hasattr(prediction_manager, 'predictions_df'):
            predictions = prediction_manager.predictions_df
            if predictions is not None and not predictions.empty:
                match_predictions = predictions[predictions['Match'] == fixture_name]
                if not match_predictions.empty:
                    # Get expected goals
                    if 'Team1' in match_predictions.columns and 'Team2' in match_predictions.columns:
                        expected_goals = {
                            'team1': match_predictions.iloc[0]['Team1'],
                            'team2': match_predictions.iloc[0]['Team2'],
                            'xg1': match_predictions.iloc[0]['Team1_Expected'],
                            'xg2': match_predictions.iloc[0]['Team2_Expected']
                        }
                    
                    # Get probabilities for current market - format to exactly match the data
                    over_col = f'Over_{threshold}_Prob'
                    under_col = f'Under_{threshold}_Prob'
                    
                    # odds_logger.info(f"Looking for columns: {over_col}, {under_col}")
                    # odds_logger.info(f"Available columns: {match_predictions.columns.tolist()}")
                    
                    if over_col in match_predictions.columns and under_col in match_predictions.columns:
                        model_probabilities = {
                            'over': match_predictions.iloc[0][over_col],
                            'under': match_predictions.iloc[0][under_col]
                        }
                        # odds_logger.info(f"Found probabilities: {model_probabilities}")
                    else:
                        odds_logger.warning(f"Probability columns not found for threshold {threshold}")

        
        # Display expected goals if available
        if expected_goals:
            st.markdown("#### Expected Goals")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"**{expected_goals['team1']}:** <span style='font-size: 20px; color: #1F77B4'>{expected_goals['xg1']:.2f}</span>", 
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    f"**{expected_goals['team2']}:** <span style='font-size: 20px; color: #1F77B4'>{expected_goals['xg2']:.2f}</span>", 
                    unsafe_allow_html=True
                )
            st.markdown("---")

        # Rest of the odds display...
        fixture_odds = odds_df[
            (odds_df['eventName'] == fixture_name) & 
            (odds_df['marketType'] == selected_market)
        ]
        
        if fixture_odds.empty:
            st.write("No odds available for selected market")
            return
            
        odds = fixture_odds.iloc[0]
        market_history = historical_df[
            (historical_df['eventName'] == fixture_name) &
            (historical_df['marketType'] == selected_market)
        ].copy()
        
        # Display odds metrics with comparison
        # Define consistent colors
        OVER_COLOR = "#1F77B4"  # Streamlit's default blue
        UNDER_COLOR = "#FF7F0E"  # Streamlit's default orange
        VALUE_COLOR = "#00cc66"  # Keep green for value bets
        NEUTRAL_COLOR = "#808495"  # Keep original gray for non-value

        # Display odds metrics with comparison
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Market Odds**")
            if pd.notna(odds['over']) and isinstance(odds['over'], (int, float)):
                implied_prob_over = (1 / float(odds['over'])) * 100
                st.markdown(f"**Over {threshold}:** <span style='font-size: 18px; color: {OVER_COLOR}'>{float(odds['over']):.2f}</span> <br><span style='font-size: 12px; color: #808495'>*Implied Prob: {implied_prob_over:.1f}%*</span>", unsafe_allow_html=True)
                
            if pd.notna(odds['under']) and isinstance(odds['under'], (int, float)):
                implied_prob_under = (1 / float(odds['under'])) * 100
                st.markdown(f"**Under {threshold}:** <span style='font-size: 18px; color: {UNDER_COLOR}'>{float(odds['under']):.2f}</span> <br><span style='font-size: 12px; color: #808495'>*Implied Prob: {implied_prob_under:.1f}%*</span>", unsafe_allow_html=True)

        with col2:
            st.markdown("**Model Probs**")
            if model_probabilities:
                model_over = model_probabilities['over'] * 100
                model_under = model_probabilities['under'] * 100
                st.markdown(f"**Over {threshold}:** <span style='font-size: 18px; color: {OVER_COLOR}'>{model_over:.1f}%</span> <br><span style='font-size: 12px; color: #808495'>*Model Confidence*</span>", unsafe_allow_html=True)
                st.markdown(f"**Under {threshold}:** <span style='font-size: 18px; color: {UNDER_COLOR}'>{model_under:.1f}%</span> <br><span style='font-size: 12px; color: #808495'>*Model Confidence*</span>", unsafe_allow_html=True)
            else:
                st.markdown("*No model predictions available*")

        with col3:
            st.markdown("**Edge Analysis**")
            if model_probabilities:
                if pd.notna(odds['over']):
                    edge_over = (model_probabilities['over'] - (1/odds['over'])) * 100
                    edge_color = VALUE_COLOR if edge_over > 3 else NEUTRAL_COLOR
                    value_text = "Value" if edge_over > 3 else "No Value"
                    st.markdown(f"**Over {threshold}:** <span style='font-size: 18px; color: {edge_color}'>{edge_over:.1f}%</span> <br><span style='font-size: 12px; color: #808495'>*{value_text}*</span>", unsafe_allow_html=True)
                
                if pd.notna(odds['under']):
                    edge_under = (model_probabilities['under'] - (1/odds['under'])) * 100
                    edge_color = VALUE_COLOR if edge_under > 3 else NEUTRAL_COLOR
                    value_text = "Value" if edge_under > 3 else "No Value"
                    st.markdown(f"**Under {threshold}:** <span style='font-size: 18px; color: {edge_color}'>{edge_under:.1f}%</span> <br><span style='font-size: 12px; color: #808495'>*{value_text}*</span>", unsafe_allow_html=True)


        # Display historical trend chart
        st.markdown("#### Price History")
        self._display_trend_chart(market_history, fixture_name, selected_market)
        
        # Display trend indicators
        self._display_trend_indicators(market_history)
        
        # Display the timestamp from the odds
        st.caption(f"Last updated: {odds['timestamp'].strftime('%H:%M:%S')}")

    def _display_trend_chart(self, market_history: pd.DataFrame, fixture_name: str, selected_market: str) -> None:
        """Display historical trend chart"""
        if not market_history.empty:
            # Ensure timestamps are datetime
            market_history['timestamp'] = pd.to_datetime(market_history['timestamp'])
            market_history = market_history.sort_values('timestamp')
            
            # Log the time range in the data
            # odds_logger.info(f"Full history time range: {market_history['timestamp'].min()} to {market_history['timestamp'].max()}")
            # odds_logger.info(f"Total records: {len(market_history)}")
            
            # Create the melted data for charting
            chart_data = market_history.melt(
                id_vars=['timestamp'],
                value_vars=['over', 'under'],
                var_name='type',
                value_name='price'
            )
            
            # Create a unique key using timestamp and hash of fixture and market
            timestamp_str = pd.Timestamp.now().strftime('%Y%m%d%H%M%S%f')
            unique_id = hash(f"{fixture_name}{selected_market}")
            slider_key = f"history_hours_{timestamp_str}_{unique_id}"
            
            # Get the time range in the data
            latest_time = market_history['timestamp'].max()
            earliest_time = market_history['timestamp'].min()
            total_hours = (latest_time - earliest_time).total_seconds() / 3600
            
            # Set max hours for slider based on available data
            max_hours = max(2, min(24, int(total_hours)))
            
            hours_to_show = st.slider(
                "Hours of history to show", 
                1, max_hours, min(3, max_hours),
                key=slider_key
            )
            
            # Calculate cutoff time
            cutoff_time = latest_time - pd.Timedelta(hours=hours_to_show)
            
            # Filter data
            filtered_data = chart_data[chart_data['timestamp'] >= cutoff_time].copy()
            
            if not filtered_data.empty:
                # Create pivot table for charting
                chart_df = filtered_data.pivot(index='timestamp', columns='type', values='price')
                
                # Display the chart
                st.line_chart(
                    data=chart_df,
                    use_container_width=True
                )
            else:
                st.warning("No data available for selected time period")

    def _analyze_trend(self, series: pd.Series, timestamps: pd.Series) -> int:
        """
        Analyze trend direction based on data from last 3 hours
        
        Args:
            series: Series of odds values
            timestamps: Series of corresponding timestamps
        """
        if len(series) < 2:
            return 0
            
        # Convert timestamps to datetime if they aren't already
        timestamps = pd.to_datetime(timestamps)
        
        # Get current time from latest timestamp
        current_time = timestamps.iloc[-1]
        
        # Filter for last 3 hours of data
        three_hours_ago = current_time - pd.Timedelta(hours=3)
        mask = timestamps >= three_hours_ago
        recent_data = series[mask]
        
        if len(recent_data) < 2:
            return 0
            
        # Calculate linear regression on the recent data
        x = np.arange(len(recent_data))
        y = recent_data.values
        slope, _ = np.polyfit(x, y, 1)
        
        # Calculate total percentage change over period
        total_change = (recent_data.iloc[-1] - recent_data.iloc[0]) / recent_data.iloc[0]
        
        # Consider both slope and total change
        threshold = 0.005  # 0.5% change
        if abs(total_change) < threshold:
            return 0
        return 1 if total_change > 0 else -1

    def _display_trend_indicators(self, market_history: pd.DataFrame) -> None:
        """Display trend indicators with color coding"""
        trend_cols = st.columns(2)
        
        def get_trend_display(trend: int) -> tuple[str, str]:
            """Returns (trend_text, color) based on trend direction"""
            if trend > 0:
                return "↑ Rising", "#00cc66"  # Red for rising (odds getting worse)
            elif trend < 0:
                return "↓ Falling", "#ff4b4b"  # Green for falling (odds improving)
            return "→ Stable", "#808495"  # Gray for stable
        
        with trend_cols[0]:
            over_trend = self._analyze_trend(market_history['over'], market_history['timestamp'])
            trend_text, color = get_trend_display(over_trend)
            st.markdown(f"Over Trend: <span style='color: {color}'>{trend_text}</span>", unsafe_allow_html=True)
        
        with trend_cols[1]:
            under_trend = self._analyze_trend(market_history['under'], market_history['timestamp'])
            trend_text, color = get_trend_display(under_trend)
            st.markdown(f"Under Trend: <span style='color: {color}'>{trend_text}</span>", unsafe_allow_html=True)

    def check_health(self) -> bool:
        """Check if odds data is being updated regularly"""
        try:
            if self._last_update_attempt  is None:
                odds_logger.info("No last_odds_refresh in session state")
                return False
                
            now = pd.Timestamp.now()
            time_since_update = now - self._last_update_attempt 
            
            odds_logger.info(f"Last odds refresh: {self._last_update_attempt }")
            odds_logger.info(f"Time since update: {time_since_update}")
            
            max_age = pd.Timedelta(minutes=10)
            is_healthy = time_since_update < max_age
            
            odds_logger.info(f"Odds health status: {is_healthy}")
            return is_healthy
                
        except Exception as e:
            odds_logger.error(f"Error checking odds health: {str(e)}")
            return False
