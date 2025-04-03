from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

class StyleEstimator:
    """Enhanced estimator for team playing styles using comprehensive metrics"""
    
    STYLE_FEATURES = {
        'possession_control': [
            'Touches - Mid 3rd',        # Control in midfield
            'Passes - Cmp%',            # Passing accuracy
            'Short - Cmp%',             # Short passing success
            'Medium - Cmp%',            # Medium passing success
            'Touches - Live',           # Live ball touches
            'Carries - Carries',        # Ball carrying
            'Performance - Recov'       # Ball recoveries
        ],
        
        'attacking_style': [
            'Passes - PrgP',            # Progressive passes
            'Carries - PrgC',           # Progressive carries
            'Carries - 1/3',            # Carries into final third
            'Carries - CPA',            # Carries into penalty area
            'Pass Types - TB',          # Through balls
            'Pass Types - Crs',         # Crosses
            'Take-Ons - Succ%',         # Dribbling success rate
            'Touches - Att 3rd',        # Touches in attacking third
            'Touches - Att Pen'         # Touches in penalty area
        ],
        
        'direct_play': [
            'Long - Cmp',               # Completed long passes
            'Long - Att',               # Attempted long passes
            'Pass Types - Sw',          # Switches of play
            'Aerial Duels - Won',       # Aerial duels won
            'Aerial Duels - Won%',      # Aerial success rate
            'Total - PrgDist',          # Progressive passing distance
            'Carries - PrgDist'         # Progressive carrying distance
        ],
        
        'pressing_intensity': [
            'Tackles - Att 3rd',        # High pressing (tackles in attacking third)
            'Tackles - Mid 3rd',        # Mid pressing (tackles in middle third)
            'Challenges - Tkl',         # Successful tackles
            'Performance - Int',        # Interceptions
            'Challenges - Tkl%',        # Tackling success rate
            'Blocks - Pass',            # Blocked passes
            'Performance - PrgP'        # Pressing success leading to progressive passes
        ],
        
        'set_piece_focus': [
            'Pass Types - Dead',        # Dead ball passes
            'Pass Types - FK',          # Free kicks
            'Pass Types - CK',          # Corner kicks
            'Corner Kicks - In',        # Inswinging corners
            'Corner Kicks - Out',       # Outswinging corners
            'Corner Kicks - Str'        # Straight corners
        ],
        
        'counter_attacking': [
            'Take-Ons - Succ',          # Successful take-ons
            'Carries - 1/3',            # Quick carries into final third
            'SCA - SCA',                # Shot-creating actions
            'SCA - GCA',                # Goal-creating actions
            'Expected - xG',            # Expected goals
            'Expected - xAG'            # Expected assisted goals
        ],
        
        'defensive_organization': [
            'Touches - Def 3rd',        # Touches in defensive third
            'Touches - Def Pen',        # Touches in defensive penalty area
            'Blocks - Blocks',          # Total blocks
            'Blocks - Sh',              # Blocked shots
            'Performance - Tkl',        # Tackles
            'Performance - Int',        # Interceptions
            'Clr'                       # Clearances
        ],
        
        'build_up_style': [
            'Short - Cmp%',             # Short passing accuracy
            'Touches - Def 3rd',        # Build-up touches
            'Passes - PrgP',            # Progressive passes
            'Carries - PrgC',           # Progressive carries
            'Pass Types - Live',        # Live ball passes
            'Receiving - PrgR'          # Progressive passes received
        ]
    }
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
    
    def estimate_features(self, row: pd.Series, date: datetime) -> Dict:
        """
        Estimate style features from pre-calculated form features
        
        Args:
            row: Series containing team and opponent form features
            date: Date of the match
        """
        features = {}
        
        # Calculate style metrics for both teams using form features
        for style, feature_list in self.STYLE_FEATURES.items():
            # Get relevant form features for this style
            team_features = []
            opp_features = []
            
            for feature in feature_list:
                # Get team feature from form
                team_feat = f"{feature}_ewm_mean"
                if team_feat in row:
                    team_features.append(row[team_feat])
                
                # Get opponent feature from form
                opp_feat = f"opponent_{feature}_ewm_mean"
                if opp_feat in row:
                    opp_features.append(row[opp_feat])
            
            if team_features and opp_features:
                # Calculate mean style scores
                team_score = np.mean(team_features)
                opp_score = np.mean(opp_features)
                
                # Calculate style metrics
                features[f'{style}_diff'] = team_score - opp_score
                features[f'{style}_interaction'] = self._calculate_interaction_score(
                    team_score, opp_score
                )
                features[f'{style}_dominance'] = self._calculate_dominance_score(
                    team_score, opp_score
                )
        
        return features
    
    def _calculate_dominance_score(self, 
                                 team_score: float, 
                                 opponent_score: float) -> float:
        """Calculate relative dominance in a particular style aspect"""
        if team_score == 0 and opponent_score == 0:
            return 0
        return (team_score - opponent_score) / (abs(team_score) + abs(opponent_score) + 1e-6)
    
    def _calculate_interaction_score(self, 
                                   team_score: float, 
                                   opponent_score: float) -> float:
        """
        Calculate interaction score between team styles
        Positive values indicate complementary styles
        Negative values indicate contrasting styles
        """
        return team_score * opponent_score

class MatchupEstimator:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
    
    def estimate_features(self, 
                         team_stats_df: pd.DataFrame,
                         team: str,
                         opponent: str,
                         date: datetime) -> Dict:
        """
        Estimate features based on historical head-to-head matchups only
        """
        # Get strictly historical matchups
        h2h_matches = team_stats_df[
            (
                ((team_stats_df['team'] == team) & (team_stats_df['opponent'] == opponent)) |
                ((team_stats_df['team'] == opponent) & (team_stats_df['opponent'] == team))
            ) &
            (team_stats_df['Date'] < date)
        ].copy()
        
        if len(h2h_matches) == 0:
            return self._get_default_features()
            
        # Sort for calculations
        h2h_matches = h2h_matches.sort_values('Date')
        
        # Get unique games count (using game field)
        unique_games = h2h_matches.drop_duplicates(subset=['game', 'Date'])
        game_count = len(unique_games)
        
        # Calculate team-specific matches using full dataset
        team_matches = h2h_matches[h2h_matches['team'] == team]
        opp_matches = h2h_matches[h2h_matches['team'] == opponent]
        
        # Calculate H2H stats
        features = {
            'h2h_matches_count': game_count,  # Using deduplicated count
            'h2h_team_goals_mean': team_matches['teamGoals'].mean() if len(team_matches) > 0 else 0,
            'h2h_opp_goals_mean': opp_matches['teamGoals'].mean() if len(opp_matches) > 0 else 0,
            'h2h_total_goals_mean': h2h_matches['teamGoals'].mean(),
            'h2h_goal_diff_mean': (
                (team_matches['teamGoals'].mean() if len(team_matches) > 0 else 0) - 
                (opp_matches['teamGoals'].mean() if len(opp_matches) > 0 else 0)
            )
        }
        
        # Recent H2H weighting
        if len(h2h_matches) > 0:
            recent_matches = h2h_matches.iloc[-self.window_size:]
            weights = np.exp(np.linspace(-1, 0, len(recent_matches)))
            features['h2h_recent_weighted_goals'] = (
                (recent_matches['teamGoals'] * weights).sum() / weights.sum()
            )
        
        return features
    
    def _get_default_features(self) -> Dict:
        """Return default features when no historical data available"""
        return {
            'h2h_matches_count': 0,
            'h2h_team_goals_mean': 0,
            'h2h_opp_goals_mean': 0,
            'h2h_total_goals_mean': 0,
            'h2h_goal_diff_mean': 0,
            'h2h_recent_weighted_goals': 0,
        }