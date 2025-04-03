import pandas as pd
import numpy as np
import datetime
import logging
from typing import Tuple, Dict, List, Optional

from .modellingUtils.featureengineering import TeamFeatureEngineering
from .modellingUtils.goal_model_evaluator import GoalModelEvaluator, PoissonProbabilityCalculator
from .modellingUtils.upcoming_betting import analyze_upcoming_value_bets, generate_upcoming_bets_report

# At the top of prediction_system.py
from .logging_setup import setup_logging

# Set up logging
prediction_logger = setup_logging('prediction_system')

class PredictionManager:
    def __init__(self, 
                 bankroll: float = 200,
                 max_daily_exposure: float = 0.25,
                 edge_threshold: float = 0.03,
                 min_probability: float = 0.5,
                 form_window: int = 10,
                 best_train_date: datetime = datetime.date(2024, 8, 10)):
        """
        Initialize the PredictionManager with betting and model parameters.
        Removed kelly_fraction and max_single_bet as they're no longer needed.
        """
        self.predictions_df = None
        self.potential_bets = None
        self.market_analysis = None
        self.future_probabilities = None
        self.bankroll = bankroll
        self.form_window = form_window
        self.best_train_date = best_train_date
        self.betting_params = {
            'max_daily_exposure': max_daily_exposure,
            'edge_threshold': edge_threshold,
            'min_probability': min_probability
        }
        
        # Initialize model parameters
        self.market_selection = 'Correct_1.5'
        self.reduced_features = None
        self.xgb_params = None
        self.weight_config = None
        
        # Load model configuration
        self._load_model_config()
        
    def _load_model_config(self):
        """Load model configuration from files, matching notebook implementation"""
        try:
            prediction_logger.info("Loading model configuration...")
            
            # Load best features exactly as notebook does
            results = pd.read_csv("/Users/danielcrake/Desktop/Football Betting 2025/src/modeling/Spark Feature Testing/results.csv")
            self.reduced_features = results[
                results['accuracy_1.5'] == results['accuracy_1.5'].max()].loc[376,'features'].split(',')
            
            # Load model parameters matching notebook approach
            results_df = pd.read_csv('/Users/danielcrake/Desktop/Football Betting 2025/src/Data/parameter_search_results.csv')
            best_results = results_df.nlargest(10, self.market_selection)
            best_config = best_results.iloc[0]
            
            # Extract XGBoost parameters
            xgb_cols = [x for x in results_df if 'xgb' in x]
            self.xgb_params = best_config[xgb_cols].to_dict()
            self.xgb_params = {x.strip('xgb_'): y for x,y in self.xgb_params.items()}
            self.weight_config = eval(best_config['weight_config'])
            
            prediction_logger.info("Model configuration loaded successfully")
            
        except Exception as e:
            prediction_logger.error(f"Error loading model configuration: {str(e)}")
            raise

    def _prepare_game_goals(self, fixtures_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare game goals data matching notebook implementation"""
        try:
            fixtures_df = fixtures_df.copy()
            fixtures_df[['homeGoals', 'awayGoals']] = fixtures_df['Score'].str.split('–', expand=True)
            
            # Create home and away dataframes
            home = fixtures_df[['Wk', 'Date', 'Home', 'Game', 'Season', 'homeGoals']].rename(
                columns={'Home': 'Team', 'homeGoals': 'teamGoals'}
            )
            away = fixtures_df[['Wk', 'Date', 'Away', 'Game', 'Season', 'awayGoals']].rename(
                columns={'Away': 'Team', 'awayGoals': 'teamGoals'}
            )

            # Combine and process
            game_goals = pd.concat([home, away], ignore_index=True)
            game_goals = game_goals[game_goals['teamGoals'].notna()]
            game_goals['teamGoals'] = game_goals['teamGoals'].astype(int)
            game_goals['Season'] = game_goals['Season'].astype(int)
            game_goals['Date'] = pd.to_datetime(game_goals['Date'])

            prediction_logger.info(f'Game Goals Size: {game_goals.shape}')
            prediction_logger.info(f'Game Goals Example: {game_goals.head()}')
            return game_goals
            
        except Exception as e:
            prediction_logger.error(f"Error preparing game goals: {str(e)}")
            raise

    def generate_predictions(self, 
                           fixtures_df: pd.DataFrame, 
                           player_stats: pd.DataFrame, 
                           upcoming_games: pd.DataFrame) -> None:
        """Generate predictions for upcoming fixtures, aligned with notebook implementation"""
        try:
            prediction_logger.info("Starting prediction generation process...")
            
            # Prepare data
            game_goals = self._prepare_game_goals(fixtures_df)
            upcoming_games = upcoming_games.copy()
            upcoming_games['Date'] = pd.to_datetime(upcoming_games['Date'])
            
            # Create features exactly as notebook does
            feature_eng = TeamFeatureEngineering(
                player_stats, 
                game_goals, 
                form_window=self.form_window
            )
            prediction_logger.info(f'Player Data Size: {player_stats.shape}')
            prediction_logger.info(f"Player Data Checking Stats: {player_stats.loc[:,'Performance - Gls': 'Performance - PKatt'].mean()}")
            # prediction_logger.info(f'Player Data Example: {player_stats.head(10)}')
            
            # Generate features matching notebook
            all_features = feature_eng.create_all_features(use_shift=True, alpha=0.1, scale=True)
            prediction_logger.info(f'All Features Size: {all_features.shape}')
            prediction_logger.info(f"All Features Checking Stats: {all_features.loc[:,'Performance - Gls_ewm_mean': 'Performance - PKatt_ewm_mean'].mean()}")
            # prediction_logger.info(f'All Features Example: {all_features.head(10)}')

            # Get full feature list
            info_cols = ['team', 'opponent', 'Season', 'game', 'teamGoals', 'Date']
            stat_cols = [x for x in all_features.columns if x not in info_cols]
            all_features = all_features[info_cols + stat_cols]
            # prediction_logger.info(f"All Features Checking Stats Sorted: {all_features[['team', 'opponent', 'Season', 'game', 'teamGoals', 'Date']].head(10)}")
            
            # Generate future features
            future_features = feature_eng.create_features_for_future_matches(
                upcoming_games,
                use_shift=True,
                alpha=0.1,
                scale=True
            )
            prediction_logger.info(f"Future features Checking Stats: {future_features.loc[:,'Performance - Gls_ewm_mean': 'Performance - PKatt_ewm_mean'].mean()}")
            # Combine features
            all_features_with_future = pd.concat([all_features.sort_values(['Date', 'team'], ascending=False), # Sorting features is important for how the XGBoost model creates predictions
                                                   future_features])
            
            # Initialize model
            full_features = [col for col in all_features.columns 
                           if col not in ['team', 'opponent', 'Season', 'game', 'teamGoals', 'Date']]
            
            evaluator = GoalModelEvaluator(
                data=all_features_with_future,
                full_features=full_features,
                reduced_features=self.reduced_features
            )
            
            # Generate predictions
            predictions, _ = evaluator.predict_with_configuration(
                split_date=str(self.best_train_date),
                use_full_features=False,
                weight_config=self.weight_config,
                xgb_params=self.xgb_params,
                use_weights=True
            )
            
            # Calculate probabilities
            calculator = PoissonProbabilityCalculator(gamma=0.15)
            future_predictions = predictions[predictions['Date'].isin(upcoming_games['Date'])]
            future_match_data = calculator.prepare_match_data(future_predictions)
            self.future_probabilities = calculator.calculate_match_probabilities(future_match_data)
            
            # Create predictions DataFrame
            self.predictions_df = calculator.analyze_multi_threshold_predictions(self.future_probabilities)
            prediction_logger.info(f"Generated predictions for {len(self.predictions_df)} matches")
            
        except Exception as e:
            prediction_logger.error(f"Error generating predictions: {str(e)}")
            raise

    def analyze_value_bets(self, upcoming_odds: pd.DataFrame) -> None:
        """Analyze value betting opportunities with new Kelly approach"""
        try:
            prediction_logger.info("Starting value bet analysis...")
            self.odds_df = upcoming_odds.copy()
            
            # Filter odds
            # filtered_odds = upcoming_odds[upcoming_odds['marketType'] != 'OVER_UNDER_35'] now added in dashboard filtering
            filtered_odds = upcoming_odds
            
            # Analyze betting opportunities
            self.potential_bets, self.market_analysis = analyze_upcoming_value_bets(
                predictions_df=self.predictions_df,
                odds_df=filtered_odds,
                initial_bankroll=self.bankroll,
                max_daily_exposure=self.betting_params['max_daily_exposure'],
                edge_threshold=self.betting_params['edge_threshold'],
                min_probability=self.betting_params['min_probability']
            )
            
            if self.potential_bets is not None:
                prediction_logger.info(f"Found {len(self.potential_bets)} potential value bets")
                for _, bet in self.potential_bets.iterrows():
                    prediction_logger.info(f"Value bet found: {bet['Match']} - {bet['Market']} {bet['Bet_Type']} @ {bet['Odds']:.2f}")
            else:
                prediction_logger.info("No value bets found")
            
        except Exception as e:
            prediction_logger.error(f"Error analyzing value bets: {str(e)}")
            raise

    def get_match_prediction(self, match_name: str) -> Optional[Dict]:
        """Get prediction details for a specific match"""
        try:
            for prediction in self.future_probabilities:
                if f"{prediction['Team1']} vs {prediction['Team2']}" == match_name:
                    return prediction
            return None
        except Exception as e:
            prediction_logger.error(f"Error getting match prediction: {str(e)}")
            return None

    def get_match_bets(self, match_name: str) -> pd.DataFrame:
        """Get recommended bets for a specific match"""
        try:
            if self.potential_bets is not None:
                return self.potential_bets[self.potential_bets['Match'] == match_name]
            return pd.DataFrame()
        except Exception as e:
            prediction_logger.error(f"Error getting match bets: {str(e)}")
            return pd.DataFrame()
        
    def get_report(self) -> str:
        """Generate a betting report"""
        try:
            return generate_upcoming_bets_report(
                self.potential_bets,
                self.market_analysis,
                initial_bankroll=self.bankroll,
                max_daily_exposure=self.betting_params['max_daily_exposure'],
                edge_threshold=self.betting_params['edge_threshold']
            )
        except Exception as e:
            prediction_logger.error(f"Error generating report: {str(e)}")
            raise
    
    def update_bankroll(self, new_bankroll: float) -> None:
        """Update bankroll and recalculate bets if needed"""
        self.bankroll = new_bankroll
        if self.predictions_df is not None and not self.predictions_df.empty:
            self.analyze_value_bets(self.odds_df)  # Re-analyze with new bankroll

    def update_daily_exposure(self, new_exposure: float) -> None:
        """Update daily exposure limit and recalculate bets if needed"""
        self.betting_params['max_daily_exposure'] = new_exposure
        if self.predictions_df is not None and not self.predictions_df.empty:
            self.analyze_value_bets(self.odds_df)  # Re-analyze with new exposure