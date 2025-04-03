import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from .estimators import StyleEstimator, MatchupEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class TeamFeatureEngineering:
    """
    Class to handle team-level feature engineering using all available features,
    integrating with estimators for future predictions
    """
    def __init__(self, 
                 player_stats_df: pd.DataFrame, 
                 game_goals_df: pd.DataFrame,
                 form_window: int = 5):
        
        # Filter out players with less than 45 minutes
        filtered_stats = player_stats_df[player_stats_df['Min'] >= 10].copy()
        print(filtered_stats.shape)

        self.player_stats = filtered_stats
        self.game_goals = game_goals_df.copy()
        self.form_window = form_window
        
        # Initialize estimators for future predictions
        self.style_estimator = StyleEstimator(window_size=form_window)
        self.matchup_estimator = MatchupEstimator(window_size=form_window)
        self.scaler = None 

        
        # Create feature groupings
        self.feature_sets = {
            'performance': [
                'Performance - Gls', 'Performance - Ast', 'Performance - PK', 
                'Performance - PKatt', 'Performance - Sh', 'Performance - SoT',
                'Performance - CrdY', 'Performance - CrdR', 'Performance - Touches',
                'Performance - Tkl', 'Performance - Int', 'Performance - Blocks',
                'Performance - 2CrdY', 'Performance - Fls', 'Performance - Fld',
                'Performance - Off', 'Performance - Crs', 'Performance - TklW',
                'Performance - PKwon', 'Performance - PKcon', 'Performance - OG',
                'Performance - Recov'
            ],
            'expected': [
                'Expected - xG', 'Expected - npxG', 'Expected - xAG'
            ],
            'sca': [
                'SCA - SCA', 'SCA - GCA'
            ],
            'passes': [
                'Passes - Cmp', 'Passes - Att', 'Passes - Cmp%', 'Passes - PrgP',
                'Total - Cmp', 'Total - Att', 'Total - Cmp%', 'Total - TotDist',
                'Total - PrgDist', 'Short - Cmp', 'Short - Att', 'Short - Cmp%',
                'Medium - Cmp', 'Medium - Att', 'Medium - Cmp%', 'Long - Cmp',
                'Long - Att', 'Long - Cmp%'
            ],
            'pass_types': [
                'Pass Types - Live', 'Pass Types - Dead', 'Pass Types - FK',
                'Pass Types - TB', 'Pass Types - Sw', 'Pass Types - Crs',
                'Pass Types - TI', 'Pass Types - CK'
            ],
            'corner_kicks': [
                'Corner Kicks - In', 'Corner Kicks - Out', 'Corner Kicks - Str'
            ],
            'defensive': [
                'Tackles - Tkl', 'Tackles - TklW', 'Tackles - Def 3rd',
                'Tackles - Mid 3rd', 'Tackles - Att 3rd', 'Challenges - Tkl',
                'Challenges - Att', 'Challenges - Tkl%', 'Challenges - Lost',
                'Blocks - Blocks', 'Blocks - Sh', 'Blocks - Pass', 'Int',
                'Tkl+Int', 'Clr', 'Err'
            ],
            'touches': [
                'Touches - Touches', 'Touches - Def Pen', 'Touches - Def 3rd',
                'Touches - Mid 3rd', 'Touches - Att 3rd', 'Touches - Att Pen',
                'Touches - Live'
            ],
            'take_ons': [
                'Take-Ons - Att', 'Take-Ons - Succ', 'Take-Ons - Succ%',
                'Take-Ons - Tkld', 'Take-Ons - Tkld%'
            ],
            'carries': [
                'Carries - Carries', 'Carries - TotDist', 'Carries - PrgDist',
                'Carries - PrgC', 'Carries - 1/3', 'Carries - CPA', 'Carries - Mis',
                'Carries - Dis'
            ],
            'receiving': [
                'Receiving - Rec', 'Receiving - PrgR'
            ],
            'aerials': [
                'Aerial Duels - Won', 'Aerial Duels - Lost', 'Aerial Duels - Won%'
            ],
            'goalkeeper': [
                'Shot Stopping - SoTA', 'Shot Stopping - GA', 'Shot Stopping - Saves',
                'Shot Stopping - Save%', 'Shot Stopping - PSxG', 'Launched - Cmp',
                'Launched - Att', 'Launched - Cmp%', 'Passes - Thr', 'Passes - Launch%',
                'Passes - AvgLen', 'Goal Kicks - Att', 'Goal Kicks - Launch%',
                'Goal Kicks - AvgLen', 'Crosses - Opp', 'Crosses - Stp', 'Crosses - Stp%',
                'Sweeper - #OPA', 'Sweeper - AvgDist'
            ]
        }
    
    def create_team_game_stats(self) -> pd.DataFrame:
        """
        Create team-level statistics, handling both historical and future games
        """
        # First aggregate player stats where we have them
        grouped = self.player_stats.groupby(['team','game','Season'])
        team_stats = grouped.apply(lambda x: pd.Series(self._aggregate_team_stats(x))).reset_index()
        
        # Get all games (including future ones) from game_goals
        all_games = self.game_goals[['Team', 'Game', 'Date', 'Season', 'teamGoals']].copy()
        
        # Merge with team stats, but use outer merge to keep future games
        team_stats = pd.merge(
            all_games,
            team_stats,
            left_on=['Team', 'Game', 'Season'],
            right_on=['team', 'game', 'Season'],
            how='outer'  # Changed from 'left' to 'outer'
        )
        
        # Fill missing team/game info from all_games
        team_stats['team'] = team_stats['team'].fillna(team_stats['Team'])
        team_stats['game'] = team_stats['game'].fillna(team_stats['Game'])
          
        # Fill NaN values in feature columns with 0
        feature_columns = [col for col in team_stats.columns 
                        if any(col.startswith(prefix) for prefix in self.feature_sets.keys())]
        team_stats[feature_columns] = team_stats[feature_columns].fillna(0)
        
        self.team_stats = team_stats
        return team_stats
    
    def _aggregate_team_stats(self, group_df: pd.DataFrame) -> Dict:
        """
        Aggregate GK features only from GK rows (by averaging them if multiple GK),
        outfield features (by sum or mean depending on if it's a % or raw count).
        """
        gk_df = group_df[group_df['Pos'] == 'GK']
        outfield_df = group_df[group_df['Pos'] != 'GK']
        
        results = {}
        for feature_set, features in self.feature_sets.items():
            if feature_set == 'goalkeeper':
                # Always take the mean of GK features, in case multiple GKs appear
                # (rare, but possible if keeper was substituted).
                df_selected = gk_df
                for feature in features:
                    if feature in df_selected.columns:
                        val = df_selected[feature].mean() if not df_selected.empty else np.nan
                        results[feature] = val
                    else:
                        results[feature] = np.nan
            else:
                # Outfield aggregator: sum or mean based on the feature name
                df_selected = outfield_df
                for feature in features:
                    if feature in df_selected.columns:
                        if any(x in feature.lower() for x in ['%','ratio','avg','mean']):
                            agg_method = 'mean'
                        else:
                            agg_method = 'sum'
                        val = df_selected[feature].agg(agg_method) if not df_selected.empty else np.nan
                        results[feature] = val
                    else:
                        results[feature] = np.nan
        return results
        
    
    def create_form_features(self,
                                team_stats_df: pd.DataFrame,
                                use_shift: bool = False,
                                scale: bool = True,
                                alpha: float = 0.3) -> pd.DataFrame:
        """
        Creates exponentially weighted form features from 'team_stats_df'.
        
        Args:
            team_stats_df: DataFrame containing team statistics
            use_shift: If True, shift(1) before EWM to avoid lookahead
            scale: If True, apply StandardScaler to the resulting numeric columns
            alpha: EWM alpha parameter (higher = more weight to recent games)
        """
        form_features_list = []
        
        # Keep track of non-feature columns
        id_columns = ['team', 'Season', 'game', 'teamGoals', 'Date']
        id_columns = [col for col in id_columns if col in team_stats_df.columns]
        
        for team in team_stats_df['team'].unique():
            # Get team data and sort by date
            team_data = team_stats_df[team_stats_df['team'] == team].sort_values('Date')
            
            # Get numeric columns, excluding identifiers
            numeric_cols = team_data.select_dtypes(include=[np.number]).columns
            numeric_cols = numeric_cols.drop(id_columns, errors='ignore')
            
            # Prepare data for EWM
            data_for_ewm = (team_data[numeric_cols].shift(1) 
                        if use_shift else team_data[numeric_cols])
            
            # Calculate EWM statistics
            ewm_mean = data_for_ewm.ewm(
                alpha=alpha,
                min_periods=1,
                adjust=False
            ).mean()
            
            ewm_std = data_for_ewm.ewm(
                alpha=alpha,
                min_periods=1,
                adjust=False
            ).std()
            
            # Rename columns
            ewm_mean.columns = [f"{col}_ewm_mean" for col in ewm_mean.columns]
            ewm_std.columns = [f"{col}_ewm_std" for col in ewm_std.columns]
            
            # Create list of DataFrames to concatenate
            dfs_to_concat = [
                ewm_mean,
                ewm_std,
                team_data[id_columns]
            ]
            
            # Concatenate all columns at once
            team_features = pd.concat(dfs_to_concat, axis=1)
            team_features = team_features.fillna(0)
            
            form_features_list.append(team_features)
        
        # Concatenate all teams' data
        form_df = pd.concat(form_features_list, axis=0).sort_index()
      
        if scale:
            # Separate features from identifiers
            feature_cols = [col for col in form_df.columns 
                        if col not in id_columns and 
                        col.endswith(('_ewm_mean', '_ewm_std'))]
            
            # Scale only the feature columns
            scaler = StandardScaler()
            form_df[feature_cols] = scaler.fit_transform(
                form_df[feature_cols].fillna(0)
            )
        
        return form_df

    def get_optimal_alpha(
        self,
        team_stats_df: pd.DataFrame,
        target_col: str = 'teamGoals',
        alpha_range: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5]) -> float:
        """
        Find optimal alpha parameter by testing prediction accuracy
        on the target column (usually goals).
        """
        errors = []
        
        for alpha in alpha_range:
            predictions = []
            actuals = []
            
            for team in team_stats_df['team'].unique():
                team_data = team_stats_df[team_stats_df['team'] == team].sort_values('Date')
                
                if target_col not in team_data.columns:
                    continue
                    
                # Calculate EWM with current alpha
                ewm_pred = team_data[target_col].shift(1).ewm(
                    alpha=alpha,
                    adjust=False
                ).mean()
                
                # Store predictions and actuals
                predictions.extend(ewm_pred.iloc[1:])  # Skip first row
                actuals.extend(team_data[target_col].iloc[1:])
            
            # Calculate error for this alpha
            mse = mean_squared_error(actuals, predictions)
            errors.append({'alpha': alpha, 'mse': mse})
        
        # Find best alpha
        best_alpha = min(errors, key=lambda x: x['mse'])['alpha']
        return errors

    def _extract_opponent(self, row):
        """Extract opponent from game field"""
        if pd.isna(row['game']):
            teams = row['Game'].split(' vs ')
            if row['Team'] == teams[0]:
                return teams[1]
            else:
                return teams[0]
        else:
            teams = row['game'].split(' vs ')
            if row['team'] == teams[0]:
                return teams[1]
            else:
                return teams[0]
              

    def create_all_features(self, 
                        use_shift: bool = True,
                        scale: bool = True,
                        alpha: float = 0.3,
                        fit_scaler: bool = True) -> pd.DataFrame:  # Add fit_scaler parameter
        """
        Creates all features including form, style, and matchup features
        
        Args:
            fit_scaler: If True, fit a new scaler. If False, use existing scaler.
                    Only fit scaler on historical data.
        """
        # Create base team stats
        team_stats = self.create_team_game_stats()
        
        # Create form features without scaling first
        unscaled_form = self.create_form_features(
            team_stats,
            use_shift=use_shift,
            scale=False,
            alpha=alpha
        )
        
        # Extract opponent from game field
        unscaled_form['opponent'] = unscaled_form.apply(self._extract_opponent, axis=1)
        self.team_stats['opponent'] = self.team_stats.apply(self._extract_opponent, axis=1)
        
        # Create opponent features by merging
        opponent_features = unscaled_form.copy()
        feature_cols = [col for col in opponent_features.columns 
                       if col.endswith(('_ewm_mean', '_ewm_std'))]
        opponent_cols = {col: f'opponent_{col}' for col in feature_cols}
        opponent_features = opponent_features.rename(columns=opponent_cols)
        
        # Merge main features with opponent features
        complete_features = pd.merge(
            unscaled_form,
            opponent_features[['team', 'game', 'Season'] + list(opponent_cols.values())],
            left_on=['opponent', 'game', 'Season'],
            right_on=['team', 'game', 'Season'],
            how='left',
            suffixes=('', '_opp')
        ).drop(columns=['team_opp'])
        
        # Calculate style and matchup features
        style_features = []
        matchup_features = []
        
        # Get unique games for processing
        unique_games = complete_features[['team', 'opponent', 'game', 'Date', 'Season']].drop_duplicates()
        
        for _, game in unique_games.iterrows():
            # Get the row with form features for this game
            form_row = complete_features[
                (complete_features['team'] == game['team']) &
                (complete_features['game'] == game['game']) &
                (complete_features['Season'] == game['Season'])
            ].iloc[0]
            
            # Style features using form features
            style_feat = self.style_estimator.estimate_features(
                form_row,
                game['Date']
            )
            style_feat.update({
                'team': game['team'],
                'game': game['game'],
                'Season': game['Season']
            })
            style_features.append(style_feat)
            
            # Get historical matchup data only once per game
            historical_stats = self.team_stats[
                (self.team_stats['Date'] < game['Date'])
            ]
            
            # Matchup features using historical stats
            matchup_feat = self.matchup_estimator.estimate_features(
                historical_stats,
                game['team'],
                game['opponent'],
                game['Date']
            )
            matchup_feat.update({
                'team': game['team'],
                'game': game['game'],
                'Season': game['Season']
            })
            matchup_features.append(matchup_feat)
        
        # Convert to DataFrames and merge
        style_df = pd.DataFrame(style_features)
        matchup_df = pd.DataFrame(matchup_features)
        
        all_features = pd.merge(
            complete_features,
            style_df,
            on=['team', 'game', 'Season'],
            how='left'
        )
        
        all_features = pd.merge(
            all_features,
            matchup_df,
            on=['team', 'game', 'Season'],
            how='left'
        )
        
        if scale:
            feature_cols = [col for col in all_features.columns 
                        if col not in ['team', 'opponent', 'game', 'Season', 'Date', 'teamGoals'] 
                        and all_features[col].dtype in ['float64', 'int64']]
            
            if fit_scaler:
                # Fit new scaler on historical data
                self.scaler = StandardScaler()
                all_features[feature_cols] = self.scaler.fit_transform(
                    all_features[feature_cols].fillna(0)
                )
            elif self.scaler is not None:
                # Use existing scaler
                all_features[feature_cols] = self.scaler.transform(
                    all_features[feature_cols].fillna(0)
                )
        
        return all_features
    
    # Add these methods to your existing TeamFeatureEngineering class:

    def prepare_future_matches(self, future_fixtures_df: pd.DataFrame) -> pd.DataFrame:
        """Create placeholder rows for upcoming matches"""
        match_rows = []
        for _, match in future_fixtures_df.iterrows():
            # Home team row
            home_row = {
                'Wk': match['Wk'],
                'Team': match['Home'],
                'Season': match['Season'],
                'Game': match['Game'],
                'Date': pd.to_datetime(match['Date']),
                'teamGoals': np.nan  # Unknown for future matches
            }
            
            # Away team row
            away_row = {
                'Wk': match['Wk'],
                'Team': match['Away'],
                'Season': match['Season'],
                'Game': match['Game'],
                'Date': pd.to_datetime(match['Date']),
                'teamGoals': np.nan  # Unknown for future matches
            }
            
            match_rows.extend([home_row, away_row])
            
        return pd.DataFrame(match_rows)

    def create_features_for_future_matches(self, 
                                    future_fixtures_df: pd.DataFrame,
                                    use_shift: bool = True,
                                    alpha: float = 0.1,
                                    scale: bool = True) -> pd.DataFrame:
        """
        Generate features for upcoming matches using historical data.
        Uses same scaler as historical data for consistency.
        """
        future_matches_df = self.prepare_future_matches(future_fixtures_df)
        
        # Store original game_goals
        original_game_goals = self.game_goals
        
        try:
            # Temporarily add future matches to game_goals
            self.game_goals = pd.concat([
                self.game_goals,
                future_matches_df
            ]).sort_values('Date')
            
            # Create features using existing scaler
            future_features = self.create_all_features(
                use_shift=use_shift,
                alpha=alpha,
                scale=scale,
                fit_scaler=False  # Use existing scaler
            )
            
            # Extract only the future matches
            future_features = future_features[
                future_features['Date'].isin(future_matches_df['Date'])
            ].copy()
            
            return future_features
            
        finally:
            # Restore original game_goals
            self.game_goals = original_game_goals
