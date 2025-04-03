import os
import sys

# Constants and path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_AND_SCRIPTS_DIR = os.path.join(SCRIPT_DIR, "Data And Scripts")
DATA_DIR = os.path.join(DATA_AND_SCRIPTS_DIR, "Data")
LOGPATH = '/Users/danielcrake/Desktop/Football Betting 2025/logs'

sys.path.append(DATA_AND_SCRIPTS_DIR)

# Imports
import streamlit as st
import pandas as pd
import time
import subprocess
from datetime import datetime, timedelta
import logging
import numpy as np
from typing import Optional
from appUtils import (
    FixtureManager,
    OddsManager,
    DataLoader,
    PredictionManager,
    setup_logging
)

data_loader = DataLoader(DATA_DIR, DATA_AND_SCRIPTS_DIR)

dashboard_logger = setup_logging('dashboard')

def initialize_session_state():
    """Initialize session state variables"""
    # Basic navigation variables
    if 'page' not in st.session_state:
        st.session_state.page = 'intro'
    if 'last_scrape_time' not in st.session_state:
        st.session_state.last_scrape_time = None
    if 'last_odds_refresh' not in st.session_state:
        st.session_state.last_odds_refresh = None
    if 'scheduler' not in st.session_state:
        st.session_state.scheduler = None
    if 'odds_initialized' not in st.session_state:
        st.session_state.odds_initialized = False
    if 'fixtures_data' not in st.session_state:
        st.session_state.fixtures_data = None
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = True
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = DataLoader(DATA_DIR, DATA_AND_SCRIPTS_DIR)
    if 'odds_manager' not in st.session_state:
        odds_manager = OddsManager()
        if odds_manager.update_odds():
            st.session_state.last_odds_refresh = pd.Timestamp.now()
        st.session_state.odds_manager = odds_manager
    if 'cached_lineups' not in st.session_state:
        st.session_state.cached_lineups = {}
    # New prediction-related states
    if 'prediction_manager' not in st.session_state:
        st.session_state.prediction_manager = None
    if 'predictions_generated' not in st.session_state:
        st.session_state.predictions_generated = False
    if 'last_prediction_time' not in st.session_state:
        st.session_state.last_prediction_time = None
    if 'bankroll' not in st.session_state:
        st.session_state.bankroll = 200.0
    if 'max_daily_exposure' not in st.session_state:
        st.session_state.max_daily_exposure = 0.25
    if 'excluded_markets' not in st.session_state:
        st.session_state.excluded_markets = ['35', '55', '65']
    
    # Make sure these are set to prevent errors
    if 'lineup_cache' not in st.session_state:
        st.session_state.lineup_cache = {}
    if 'prev_unplayed_only' not in st.session_state:
        st.session_state.prev_unplayed_only = True
    if 'prev_completed' not in st.session_state:
        st.session_state.prev_completed = False
    if 'prev_selected_gw' not in st.session_state:
        st.session_state.prev_selected_gw = []
    if 'selected_gameweeks' not in st.session_state:
        st.session_state.selected_gameweeks = []

def reset_session_state():
    """Reset essential session state variables while preserving user settings"""
    # Store important user settings that we want to preserve
    preserved_settings = {}
    settings_to_preserve = [
        'bankroll', 'max_daily_exposure', 'excluded_markets', 
        'auto_refresh', 'last_scrape_time'
    ]
    
    for key in settings_to_preserve:
        if key in st.session_state:
            preserved_settings[key] = st.session_state[key]
    
    # Clear all button states (they usually end with '_clicked')
    for key in list(st.session_state.keys()):
        if key.endswith('clicked') or key.endswith('button'):
            if key in st.session_state:
                del st.session_state[key]
    
    # Reset navigation and data state
    if 'page' in st.session_state:
        st.session_state.page = 'intro'  # Reset to intro page
    
    # Clear fixture and prediction data
    keys_to_clear = [
        'fixtures_data', 'fixture_cards', 'cached_lineups',
        'lineup_cache', 'predictions_generated', 'prediction_manager'
    ]
    
    for key in keys_to_clear:
        if key in st.session_state:
            st.session_state[key] = None if key != 'cached_lineups' and key != 'lineup_cache' else {}
    
    # Restore preserved settings
    for key, value in preserved_settings.items():
        st.session_state[key] = value
        
    # Set a flag to indicate reset was performed
    st.session_state.reset_performed = True
    
    # Reset initialization flag to force re-initialization
    st.session_state.initialized = False

def initialize_predictions(
    full_fixtures_df: pd.DataFrame, 
    upcoming_fixtures_df: pd.DataFrame, 
    data_loader: DataLoader) -> bool:
    """Initialize prediction manager and generate predictions"""
    try:
        logging.info("Starting prediction initialization...")
        
        # Load required data
        player_stats_path = os.path.join(DATA_DIR, 'all_players.csv')
        player_stats = pd.read_csv(player_stats_path)
        
        # Prepare upcoming games data
        upcoming_games = upcoming_fixtures_df.copy()
        upcoming_games['Date'] = pd.to_datetime(upcoming_games['Date'])
        
        # Initialize prediction manager with current bankroll and exposure settings
        prediction_manager = PredictionManager(
            bankroll=st.session_state.bankroll,
            max_daily_exposure=st.session_state.max_daily_exposure,
            edge_threshold=0.03,
            min_probability=0.5,
            form_window=10
        )
        
        # Generate predictions using full fixtures data for history
        with st.spinner('Generating predictions... This may take a few minutes.'):
            logging.info("Generating predictions...")
            prediction_manager.generate_predictions(
                fixtures_df=full_fixtures_df,
                player_stats=player_stats,
                upcoming_games=upcoming_games
            )
            
            # Get odds and analyze value bets
            logging.info("Loading odds data...")
            odds_df = st.session_state.odds_manager.load_odds('goals')
            
            if odds_df.empty:
                logging.warning("No odds data available")
                st.warning("No odds data available. Predictions generated but betting analysis skipped.")
            else:
                # Filter out excluded markets
                if st.session_state.excluded_markets:
                    # Extract market values from marketType (OVER_UNDER_X.X)
                    logging.info(f"Filtering out excluded markets: {st.session_state.excluded_markets}")
                    filtered_odds = odds_df.copy()
                    for market in st.session_state.excluded_markets:
                        # Filter out markets like OVER_UNDER_3.5
                        filtered_odds = filtered_odds[~filtered_odds['marketType'].str.endswith(f'_{market}')]
                    
                    logging.info(f"Filtered {len(odds_df) - len(filtered_odds)} odds entries")
                    odds_df = filtered_odds
                
                logging.info("Analyzing value bets...")
                prediction_manager.analyze_value_bets(odds_df)
        
        # Update session state
        st.session_state.prediction_manager = prediction_manager
        st.session_state.predictions_generated = True
        st.session_state.last_prediction_time = pd.Timestamp.now()
        
        logging.info("Prediction initialization completed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Error initializing predictions: {str(e)}")
        st.error("Failed to generate predictions. Please check the logs for details.")
        return False

def display_market_filters():
    """Display controls for excluding specific betting markets"""
    # Get all available markets (format matches the data structure)
    available_markets = ['05', '15', '25', '35', '45', '55', '65']
    
    # Create multiselect for excluded markets with better display
    excluded_markets = st.sidebar.multiselect(
        "Exclude Goal Markets",
        options=available_markets,
        default=st.session_state.excluded_markets,
        format_func=lambda x: f"{x[0]}.{x[1]} Goals",  # Display as "0.5 Goals", "1.5 Goals", etc.
        help="Markets excluded from betting analysis. 3.5, 5.5 and 6.5 are excluded by default as the model is not good at prediction for these markets yet."
    )
    
    # Update session state if changed
    if excluded_markets != st.session_state.excluded_markets:
        st.session_state.excluded_markets = excluded_markets
        # Reset predictions if markets changed
        if st.session_state.predictions_generated:
            st.session_state.predictions_generated = False
            st.sidebar.info("Market filters changed. Please regenerate predictions.")

def run_sky_scraper_only():
    """Run Sky Sports scraper for selected gameweeks"""
    try:
        import subprocess
        
        # Get current fixtures
        data_loader = st.session_state.data_loader
        fixtures_df = data_loader.load_fixtures()
        
        if fixtures_df.empty:
            st.warning("No fixtures found. Please update FBRef data first.")
            return False
            
        # Get current season
        current_season = fixtures_df['Season'].max()
        
        # Get the selected gameweeks from session state
        if 'selected_gameweeks' not in st.session_state or not st.session_state.selected_gameweeks:
            st.warning("No gameweeks selected. Please select at least one gameweek.")
            return False
            
        selected_gameweeks = st.session_state.selected_gameweeks
        
        success = True
        with st.spinner(f"Updating lineups for {len(selected_gameweeks)} gameweeks..."):
            for gameweek in selected_gameweeks:
                # Find fixtures for this gameweek
                gw_fixtures = fixtures_df[(fixtures_df['Season'] == current_season) & 
                                          (fixtures_df['Wk'] == gameweek)]
                
                if gw_fixtures.empty:
                    st.warning(f"No fixtures found for Gameweek {gameweek}")
                    continue
                
                # Get the date for the first match of the gameweek
                gw_fixtures['Date'] = pd.to_datetime(gw_fixtures['Date'])
                matchday = gw_fixtures['Date'].min().strftime("%d-%B-%Y")
                
                dashboard_logger.info(f"Running Sky Sports scraper for GW{gameweek}")
                
                try:
                    # Run the Sky Sports scraper for this gameweek
                    result = subprocess.run([
                        sys.executable, 
                        os.path.join(DATA_AND_SCRIPTS_DIR, "skysports-scraper.py"),
                        "--date", matchday,
                        "--season", str(current_season),
                        "--gameweek", str(int(gameweek))  # Convert to integer
                    ], check=True, capture_output=True, text=True)
                    
                    dashboard_logger.info(f"Sky Sports scraper for GW{gameweek} completed")
                    
                except subprocess.CalledProcessError as e:
                    dashboard_logger.error(f"Error scraping lineups for GW{gameweek}: {str(e)}")
                    st.error(f"Failed to update lineups for Gameweek {gameweek}")
                    success = False
            
            # Reset cached lineups to force reload but preserve predictions
            st.session_state.cached_lineups = {}

            if 'fixtures_data' in st.session_state:
                st.session_state.fixtures_data = None
            
            st.success("Lineups updated successfully!")
            time.sleep(1)  # Give user time to see the message
            st.rerun()  # Refresh the page to show updated lineups
            
        return success
        
    except Exception as e:
        dashboard_logger.error(f"Error running Sky Sports scraper: {str(e)}")
        st.error(f"Error updating lineups: {str(e)}")
        return False

def display_betting_controls():
    """Display bankroll and betting controls"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Betting Settings")
    
    # Bankroll input with more precise control
    new_bankroll = st.sidebar.number_input(
        "Bankroll (£)",
        min_value=10.0,
        max_value=10000.0,
        value=st.session_state.bankroll,
        step=10.0,
        format="%.2f",
        help="Your current betting bankroll"
    )
    
    # Daily exposure input as percentage
    new_exposure = st.sidebar.slider(
        "Max Daily Exposure (%)",
        min_value=5,
        max_value=50,
        value=int(st.session_state.max_daily_exposure * 100),
        step=5,
        help="Maximum percentage of bankroll to risk per day"
    ) / 100.0
    
    # Update values if changed
    if new_bankroll != st.session_state.bankroll or new_exposure != st.session_state.max_daily_exposure:
        st.session_state.bankroll = new_bankroll
        st.session_state.max_daily_exposure = new_exposure
        
        # Update prediction manager if it exists
        if st.session_state.prediction_manager is not None:
            st.session_state.prediction_manager.update_bankroll(new_bankroll)
            st.session_state.prediction_manager.update_daily_exposure(new_exposure)
            st.session_state.predictions_generated = False
            st.info("Betting settings updated. Please regenerate predictions.")
            
    # Display current exposure amount
    daily_limit = new_bankroll * new_exposure
    st.sidebar.caption(f"Daily betting limit: £{daily_limit:.2f}")

def display_kelly_calculator():
    """Display a Kelly criterion calculator in the sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Kelly Calculator")
    
    # Input for odds (in decimal format)
    odds = st.sidebar.number_input(
        "Decimal Odds",
        min_value=1.01,
        max_value=100.0,
        value=2.0,
        step=0.01,
        format="%.2f",
        help="Enter decimal odds (e.g., 2.00 for even money)"
    )
    
    # Input for probability estimate
    prob = st.sidebar.slider(
        "Estimated Probability (%)",
        min_value=1,
        max_value=99,
        value=50,
        step=1,
        help="Your estimated probability of winning"
    ) / 100.0
    
    # Input for stake amount
    stake = st.sidebar.number_input(
        "Available Stake (£)",
        min_value=1.0,
        max_value=10000.0,
        value=100.0,
        step=10.0,
        format="%.2f",
        help="Amount available to bet"
    )
    
    # Calculate Kelly stake
    b = odds - 1  # Decimal odds to fractional odds
    q = 1 - prob  # Probability of losing
    kelly = (b * prob - q) / b  # Kelly formula
    
    if kelly > 0:
        recommended_stake = kelly * stake
        st.sidebar.markdown("#### Recommended Stake")
        
        # Full Kelly
        st.sidebar.markdown(f"**Full Kelly:** £{recommended_stake:.2f} ({kelly*100:.1f}%)")
        
        # Half Kelly
        st.sidebar.markdown(f"**Half Kelly:** £{(recommended_stake/2):.2f} ({kelly*50:.1f}%)")
        
        # Quarter Kelly
        st.sidebar.markdown(f"**Quarter Kelly:** £{(recommended_stake/4):.2f} ({kelly*25:.1f}%)")
        
        # Expected value
        ev = (prob * (stake * (odds - 1)) - (1 - prob) * stake)
        st.sidebar.markdown(f"**Expected Value:** £{ev:.2f}")
    else:
        st.sidebar.warning("No value found at these odds/probability")

def display_betting_report():
    """Display betting report and value bets"""
    if not st.session_state.predictions_generated:
        st.warning("No predictions available. Generate predictions first.")
        return
        
    prediction_manager = st.session_state.prediction_manager
    
    # Display value bets table
    if prediction_manager.potential_bets is not None and not prediction_manager.potential_bets.empty:
        st.subheader("🎯 Recommended Value Bets")
        # Format the bets dataframe for display
        display_bets = prediction_manager.potential_bets.copy()
        
        # Format numeric columns
        display_bets['Probability'] = (display_bets['Probability'] * 100).round(1).astype(str) + '%'
        display_bets['Edge'] = (display_bets['Edge'] * 100).round(1).astype(str) + '%'
        display_bets['Stake'] = display_bets['Stake'].round(2).astype(str) + ' units'
        
        # Format date column
        display_bets['Date'] = pd.to_datetime(display_bets['Date']).dt.strftime('%Y-%m-%d')
        
        # Reorder and rename columns
        columns = {
            'Date': 'Date',
            'Match': 'Match',
            'Market': 'Market',
            'Bet_Type': 'Selection',
            'Odds': 'Odds',
            'Probability': 'Model Prob',
            'Edge': 'Edge',
            'Stake': 'Stake'
        }
        
        st.dataframe(
            display_bets[columns.keys()].rename(columns=columns),
            hide_index=True,
            use_container_width=True
        )
        
        # Show full report in expander
        with st.expander("📊 View Detailed Betting Report"):
            st.markdown(prediction_manager.get_report())
    else:
        st.info("No value bets found meeting the criteria.")
    
def display_status_sidebar(odds_manager):
    """Display system status in sidebar with scheduler controls"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Status")
    
    # Scheduler Controls
    st.sidebar.subheader("Scheduler Settings")
    
    # Toggle scheduler
    scheduler_active = st.sidebar.checkbox(
        "Enable Odds Scheduler",
        value=odds_manager.scheduler.running if odds_manager.scheduler else False,
        key="scheduler_toggle"
    )
    
    # Update interval slider (in minutes)
    update_interval = st.sidebar.slider(
        "Update Interval (minutes)",
        min_value=1,
        max_value=30,
        value=int(odds_manager.UPDATE_INTERVAL / 60),
        step=1,
        key="update_interval"
    )
    
    # Apply scheduler changes if needed
    if scheduler_active != (odds_manager.scheduler.running if odds_manager.scheduler else False):
        if scheduler_active:
            odds_manager.UPDATE_INTERVAL = update_interval * 60
            odds_manager.HEALTH_THRESHOLD = odds_manager.UPDATE_INTERVAL * 1.5
            odds_manager.start_scheduler()
        else:
            if odds_manager.scheduler:
                odds_manager.scheduler.shutdown()
                odds_manager.scheduler = None
    
    # Update interval if changed
    elif scheduler_active and odds_manager.UPDATE_INTERVAL != update_interval * 60:
        odds_manager.UPDATE_INTERVAL = update_interval * 60
        odds_manager.HEALTH_THRESHOLD = odds_manager.UPDATE_INTERVAL * 1.5
        if odds_manager.scheduler:
            odds_manager.scheduler.shutdown()
            odds_manager.scheduler = None
            odds_manager.start_scheduler()
    
    # Get odds status
    status = odds_manager.get_odds_status()
    
    st.sidebar.markdown("---")
    
    # Display status
    status_cols = st.sidebar.columns(2)
    with status_cols[0]:
        status_color = "💚" if status['is_healthy'] else "🔴"
        st.write("Odds Status:")
    with status_cols[1]:
        st.write(f"{status_color} {'Active' if status['is_healthy'] else 'Inactive'}")
    
    # Show auto-refresh status
    refresh_cols = st.sidebar.columns(2)
    with refresh_cols[0]:
        st.write("Auto-refresh:")
    with refresh_cols[1]:
        st.write("💚 On" if st.session_state.auto_refresh else "🔴 Off")
    
    # Show scheduler status
    scheduler_cols = st.sidebar.columns(2)
    with scheduler_cols[0]:
        st.write("Scheduler:")
    with scheduler_cols[1]:
        st.write("💚 Running" if scheduler_active else "🔴 Stopped")
    
    # Show current update interval
    interval_cols = st.sidebar.columns(2)
    with interval_cols[0]:
        st.write("Update Interval:")
    with interval_cols[1]:
        st.write(f"{update_interval} minutes")
    
    # Show last update time with real-time counting
    if status['last_refresh']:
        time_cols = st.sidebar.columns(2)
        with time_cols[0]:
            st.write("Last Odds Update:")
        with time_cols[1]:
            seconds_ago = status.get('seconds_ago', 0)
            if seconds_ago < 10:
                st.write("Just now")
            elif seconds_ago < 60:
                st.write(f"{int(seconds_ago)}s ago")
            else:
                st.write(f"{int(seconds_ago / 60)}m {int(seconds_ago % 60)}s ago")
    
    # Only show warning if truly unhealthy and scheduler is active
    if not status['is_healthy'] and scheduler_active:
        st.sidebar.warning("Odds updates are delayed. Check API connection.")
    
    return status['is_healthy']

def update_odds_only():
    """Update only the odds data without refreshing lineups"""
    try:
        odds_manager = st.session_state.odds_manager
        odds_manager.update_odds()
        st.session_state.last_odds_refresh = datetime.now()
    except Exception as e:
        logging.error(f"Error updating odds: {str(e)}")

def handle_auto_refresh():
    """Handle auto-refresh logic"""
    auto_refresh = st.sidebar.checkbox(
        "Auto-refresh Odds",
        value=st.session_state.auto_refresh,
        key="auto_refresh_checkbox",
        help="Automatically refresh dashboard every second"
    )
    
    if auto_refresh != st.session_state.auto_refresh:
        st.session_state.auto_refresh = auto_refresh
        st.rerun()

def show_intro_page():
    st.title("Premier League Fixtures Dashboard ⚽")
    
    if st.session_state.last_scrape_time:
        st.info(f"Last FBRef data update: {st.session_state.last_scrape_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Use simple static key instead of timestamp-based key
        if st.button("🔄 Update FBRef Data", key="update_fbref_data_btn", help="Run the FBRef scraper to update fixture data"):
            data_loader = st.session_state.data_loader
            if data_loader.initialize_data():
                st.session_state.last_scrape_time = datetime.now()
                if 'fixtures_data' in st.session_state:
                    st.session_state.fixtures_data = None
                st.success("Data updated successfully!")
            else:
                st.error("Failed to update data")
    
    with col2:
        # Use simple static key instead of timestamp-based key
        if st.button("📊 View Dashboard", key="view_dashboard_btn", help="Continue to the main dashboard"):
            st.session_state.page = 'dashboard'
            st.rerun()
    
    st.markdown("---")
    st.markdown("""
    ### ℹ️ Quick Guide
    - **Update FBRef Data**: Use this if you need to refresh the fixture data. This might take a few minutes.
    - **View Dashboard**: Continue to the main dashboard using existing data.
    
    Note: You don't need to update the FBRef data every time - once per day is usually sufficient.
    """)


def show_dashboard():
    """Show the main dashboard"""
    # Make sure we're on the dashboard page
    st.session_state.page = 'dashboard'
    
    # Get data loader and odds manager from session state
    data_loader = st.session_state.data_loader
    odds_manager = st.session_state.odds_manager

    # Only start the scheduler if it's not already running
    if odds_manager.scheduler is None or not odds_manager.scheduler.running:
        odds_manager.start_scheduler()
    
    odds_manager.monitor_scheduler()

    st.title("Premier League Fixtures Dashboard ⚽")
    
    if not data_loader.validate_data_directory():
        return

    # ===== SIDEBAR ORGANIZATION =====
    
    # 1. BASIC CONTROLS
    st.sidebar.title("Controls")
    
    # Simple static key for return button
    if st.sidebar.button("← Return to Intro", key="return_to_intro_btn"):
        st.session_state.page = 'intro'
        st.rerun()
    
    # Simple static key for update lineups button
    if st.sidebar.button("🔄 Update Lineups Only", key="update_lineups_btn", help="Run the Sky Sports scraper to update lineups without affecting predictions"):
        if run_sky_scraper_only():
            st.success("Lineups updated successfully!")
            time.sleep(1)  # Give user time to see the message
            st.rerun()  # Refresh the page to show updated lineups
    
    st.sidebar.markdown("---")
    handle_auto_refresh()

    # 2. SYSTEM STATUS
    # Display status and handle refresh
    odds_healthy = display_status_sidebar(odds_manager)
    
    # Load fixtures
    fixtures_df = data_loader.load_fixtures()
    if fixtures_df.empty:
        st.warning("No fixtures found. Please return to the intro page and run the scraper.")
        return

    # Initialize fixture manager
    fixture_manager = FixtureManager(fixtures_df, data_loader=data_loader)

    try:
        # 3. GAME FILTERS
        st.sidebar.markdown("---")
        st.sidebar.subheader("Game Filters")

        # Store previous filter states to detect changes
        prev_unplayed_only = st.session_state.prev_unplayed_only
        prev_completed = st.session_state.prev_completed
        prev_selected_gw = st.session_state.prev_selected_gw

        # Use simple static keys
        show_unplayed_only = st.sidebar.checkbox("Show Unplayed Games Only", value=True, key="unplayed_only_cb")
        show_completed = False if show_unplayed_only else st.sidebar.checkbox("Show Completed Games", value=False, key="completed_games_cb")

        # Get gameweek options
        available_gw = fixture_manager.get_available_gameweeks(show_unplayed_only)
        default_gw = fixture_manager.get_default_gameweeks(available_gw)

        # Simple static key for gameweek select
        selected_gw = st.sidebar.multiselect(
            "Gameweek",
            available_gw,
            default=default_gw,
            format_func=fixture_manager.format_gameweek,
            key="gameweek_select"
        )

        # Check if any filter has changed
        filters_changed = (prev_unplayed_only != show_unplayed_only or 
                        prev_completed != show_completed or 
                        set(prev_selected_gw) != set(selected_gw))

        # Update session state with current values for next comparison
        st.session_state.prev_unplayed_only = show_unplayed_only
        st.session_state.prev_completed = show_completed
        st.session_state.prev_selected_gw = selected_gw

        # If filters changed, reset everything and force a rerun
        if filters_changed:
            # Show a message
            with st.spinner("Updating fixtures..."):
                # Reset fixture-related data only, not all session state
                if 'fixture_cards' in st.session_state:
                    del st.session_state.fixture_cards
                
                if 'fixtures_data' in st.session_state:
                    st.session_state.fixtures_data = None
                
                if 'lineup_cache' in st.session_state:
                    st.session_state.lineup_cache = {}
                
                if 'cached_lineups' in st.session_state:
                    st.session_state.cached_lineups = {}
                
                # Wait a moment for the message to be visible
                time.sleep(0.5)
            
            st.rerun()

        if selected_gw:
            # Get filtered fixtures
            filtered_fixtures_df = fixture_manager.filter_fixtures(
                selected_gw, 
                show_unplayed_only, 
                show_completed
            )

            st.session_state.selected_gameweeks = selected_gw
            
            # 4. PREDICTION SETTINGS
            st.sidebar.markdown("---")
            st.sidebar.subheader("Prediction Settings")
            
            # Add market filters first (within Prediction Settings)
            display_market_filters()
            
            # Add betting controls to sidebar
            display_betting_controls()
            
            # Simple static key for predictions button
            if st.sidebar.button("🎯 Generate Predictions", key="generate_predictions_btn"):
                initialize_predictions(fixtures_df, filtered_fixtures_df, data_loader)
            
            # Show when predictions were last generated
            if st.session_state.last_prediction_time:
                st.sidebar.info(f"Last predictions: {st.session_state.last_prediction_time.strftime('%H:%M:%S')}")
            
            # 5. BETTING TOOLS
            st.sidebar.markdown("---")
            st.sidebar.subheader("Betting Tools")
            display_kelly_calculator()
            
            # Display betting report if predictions are available
            if st.session_state.predictions_generated:
                display_betting_report()
            
            # Display fixtures by gameweek
            for gw in sorted(filtered_fixtures_df['Wk'].unique()):
                st.markdown(f"## Gameweek {gw}")
                
                if gw in fixture_manager.active_gameweeks:
                    st.markdown("**🟢 Active Gameweek**")
                
                gw_fixtures = filtered_fixtures_df[filtered_fixtures_df['Wk'] == gw]
                
                # Create fixture cards
                for i in range(0, len(gw_fixtures), 2):
                    cols = st.columns(2)
                    with cols[0]:
                        fixture_manager.create_fixture_card(gw_fixtures.iloc[i], odds_manager)
                    if i + 1 < len(gw_fixtures):
                        with cols[1]:
                            fixture_manager.create_fixture_card(gw_fixtures.iloc[i + 1], odds_manager)

                st.markdown("---")
                
    except Exception as e:
        logging.error(f"Error in dashboard: {str(e)}")
        st.error(f"An error occurred while displaying the fixtures: {str(e)}")
    
    if st.session_state.auto_refresh:
        time.sleep(30)
        st.rerun()

def main():
    st.set_page_config(
        page_title="Premier League Fixtures Dashboard",
        page_icon="⚽",
        layout="wide"
    )
    
    # Simple reset button in the sidebar
    if st.sidebar.button("🔄 Reset App", key="reset_app_btn", help="Reset application state if buttons stop working"):
        # Clear session state completely
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # Always initialize session state
    initialize_session_state()
    
    # Navigate based on the page state
    if st.session_state.page == 'intro':
        show_intro_page()
    else:
        show_dashboard()


if __name__ == "__main__":
    main()