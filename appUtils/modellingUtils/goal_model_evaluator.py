import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, f1_score, mean_absolute_error, 
                           mean_squared_error, r2_score, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import warnings
from scipy.stats import poisson  # Added for Poisson calculations
from scipy.optimize import minimize  # Added if you want to optimize gamma parameter

# Suppress deprecation warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

class GoalModelEvaluator:
    """
    Goal Model Evaluator class with both multi-model and single-model evaluation capabilities.
    
    Methods are marked with:
    [MULTI] - Used for comparing multiple model configurations
    [SINGLE] - Used for single model configuration analysis
    [BOTH] - Used by both multi and single model workflows
    """
    def __init__(self, data: pd.DataFrame, 
                 full_features: List[str],
                 reduced_features: List[str],
                 info_cols: List[str] = None):
        """
        [BOTH] Initialize the model evaluator
        Initialize the model evaluator
        
        Args:
            data: DataFrame containing all features and target
            full_features: List of column names for full feature set
            reduced_features: List of column names for reduced feature set
            info_cols: List of informational columns (team, opponent, date etc.)
        """
        # Create a clean copy of the data to avoid fragmentation
        self.data = data.copy()
        
        # Pre-create the game_id column to avoid fragmentation
        self.data['game_id'] = (self.data['Season'].astype(str) + '_' + 
                               self.data['game'].astype(str))
        
        # Ensure all feature columns are numeric
        for col in full_features + reduced_features:
            if col in self.data.columns and not pd.api.types.is_numeric_dtype(self.data[col]):
                try:
                    self.data[col] = pd.to_numeric(self.data[col])
                except:
                    print(f"Warning: Could not convert column {col} to numeric. Dropping from features.")
                    full_features = [f for f in full_features if f != col]
                    reduced_features = [f for f in reduced_features if f != col]
        
        self.full_features = full_features
        self.reduced_features = reduced_features
        self.info_cols = info_cols or ['team', 'opponent', 'Season', 'game', 'teamGoals', 'Date']
        self.thresholds = [0.5, 1.5, 2.5, 3.5, 4.5]

    def create_sample_weights(self, y: np.ndarray, weight_config: dict = None) -> np.ndarray:
        """
        [BOTH] Create sample weights with configurable parameters
        Create sample weights with configurable parameters, optimized for 1.5-2 goal range"""
        if weight_config is None:
            weight_config = {
                'base': 1.0,
                'over_1': 2.0,
                'over_2': 1.5,
                'over_3': 1.0,
                'over_4': 1.0
            }
        
        weights = np.ones_like(y) * weight_config['base']
        
        # Apply weights in reverse order to ensure proper layering
        if 'over_4' in weight_config:
            weights[y > 4] = weight_config['over_4']
        if 'over_3' in weight_config:
            weights[y > 3] = weight_config['over_3']
        if 'over_2' in weight_config:
            weights[y > 2] = weight_config['over_2']
        if 'over_1' in weight_config:
            weights[y > 1] = weight_config['over_1']
            
        return weights

    def _create_xgb_model(self, params: dict = None) -> xgb.XGBRegressor:
        """
        [BOTH] Create XGBoost model with configurable parameters
        Create XGBoost model with configurable parameters"""
        default_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'enable_categorical': False
        }
        
        if params:
            default_params.update(params)
            
        return xgb.XGBRegressor(**default_params)
    
    def split_data_random(self, test_size: float = 0.2, 
                            random_state: int = 42,
                            keep_pairs: bool = False) -> Tuple[Dict, Dict]:
        """
        [BOTH] Split data randomly with option to keep match pairs together
        Split data randomly with option to keep match pairs together
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            keep_pairs: If True, ensures both teams from same game/season stay together
        """
        if keep_pairs:
            # Create unique game identifiers and split at game level
            unique_games = self.data['game_id'].unique()
            train_games, test_games = train_test_split(
                unique_games, 
                test_size=test_size,
                random_state=random_state
            )
            train_mask = self.data['game_id'].isin(train_games)
            test_mask = self.data['game_id'].isin(test_games)
        else:
            # Regular stratified split
            train_indices, test_indices = train_test_split(
                np.arange(len(self.data)),
                test_size=test_size,
                random_state=random_state,
                stratify=self.data['teamGoals']
            )
            # Convert numpy array indices to pandas Series with boolean mask
            train_mask = pd.Series(False, index=self.data.index)
            test_mask = pd.Series(False, index=self.data.index)
            train_mask.iloc[train_indices] = True
            test_mask.iloc[test_indices] = True
        
        return self._prepare_split_data(train_mask, test_mask)
    
    def split_data_temporal(self, split_date: str) -> Tuple[Dict, Dict]:
        """
        [BOTH] Split data based on a specific date
        Split data based on a specific date"""
        split_date = pd.to_datetime(split_date)
        train_mask = pd.to_datetime(self.data['Date']) < split_date
        test_mask = pd.to_datetime(self.data['Date']) >= split_date
        
        return self._prepare_split_data(train_mask, test_mask)
    
    def _prepare_split_data(self, train_mask: pd.Series, 
                            test_mask: pd.Series) -> Tuple[Dict, Dict]:
        """
        [BOTH] Prepare training and test data with both feature sets
        Prepare training and test data with both feature sets"""
        train_data = {
            'full': {
                'X': self.data.loc[train_mask, self.full_features].copy(),
                'y': self.data.loc[train_mask, 'teamGoals'].copy()
            },
            'reduced': {
                'X': self.data.loc[train_mask, self.reduced_features].copy(),
                'y': self.data.loc[train_mask, 'teamGoals'].copy()
            }
        }
        
        test_data = {
            'full': {
                'X': self.data.loc[test_mask, self.full_features].copy(),
                'y': self.data.loc[test_mask, 'teamGoals'].copy()
            },
            'reduced': {
                'X': self.data.loc[test_mask, self.reduced_features].copy(),
                'y': self.data.loc[test_mask, 'teamGoals'].copy()
            },
            'indices': self.data[test_mask].index  # Changed this line
        }
        
        return train_data, test_data
    
    def train_and_evaluate(self, train_data: Dict, 
                          test_data: Dict) -> Tuple[Dict, pd.DataFrame, Dict]:
        """Train models and evaluate performance
        [MULTI] Train and evaluate multiple model configurations
        For testing different features"""
        # Ensure all features are numeric
        for data_type in ['full', 'reduced']:
            for col in train_data[data_type]['X'].select_dtypes(include=['object']).columns:
                train_data[data_type]['X'][col] = pd.to_numeric(train_data[data_type]['X'][col], errors='coerce')
                test_data[data_type]['X'][col] = pd.to_numeric(test_data[data_type]['X'][col], errors='coerce')
        
        models = {
            'full_standard': self._train_model(train_data['full'], balanced=False),
            'full_balanced': self._train_model(train_data['full'], balanced=True),
            'reduced_standard': self._train_model(train_data['reduced'], balanced=False),
            'reduced_balanced': self._train_model(train_data['reduced'], balanced=True)
        }
        
        predictions = self._generate_predictions(models, test_data)
        metrics = self._calculate_metrics(test_data, predictions)
        
        return models, predictions, metrics
    
    def _train_model(self, data: Dict, balanced: bool) -> xgb.XGBRegressor:
        """
        [MULTI] Train a single XGBoost model for multi-model comparison
        Train a single XGBoost model"""
        model = self._create_xgb_model()
        
        if balanced:
            sample_weights = self.create_sample_weights(data['y'])
            model.fit(data['X'], data['y'], sample_weight=sample_weights)
        else:
            model.fit(data['X'], data['y'])
            
        return model
    
    def _generate_predictions(self, models: Dict, test_data: Dict) -> pd.DataFrame:
        """
        [MULTI] Generate predictions for all models in multi-model compariso
        Generate predictions for all models"""
        predictions = pd.DataFrame(index=test_data['indices'])
        
        for model_name, model in models.items():
            feature_set = 'full' if 'full' in model_name else 'reduced'
            predictions[f'pred_{model_name}'] = model.predict(test_data[feature_set]['X'])
        
        predictions['actual'] = test_data['full']['y']
        
        return predictions
    
    def _calculate_metrics(self, test_data: Dict, 
                          predictions: pd.DataFrame) -> Dict:
        """
        [MULTI] Calculate comprehensive metrics for all models
        Calculate comprehensive metrics for all models"""
        metrics = {threshold: {} for threshold in self.thresholds}
        model_columns = [col for col in predictions.columns if col.startswith('pred_')]
        
        for threshold in self.thresholds:
            y_true_binary = predictions['actual'] > threshold
            
            for model_col in model_columns:
                y_pred = predictions[model_col]
                y_pred_binary = y_pred > threshold
                
                tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
                
                metrics[threshold][model_col] = {
                    'accuracy': (tp + tn) / (tp + tn + fp + fn),
                    'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                    'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                    'f1': f1_score(y_true_binary, y_pred_binary),
                    'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                    'rmse': np.sqrt(mean_squared_error(predictions['actual'], y_pred)),
                    'mae': mean_absolute_error(predictions['actual'], y_pred),
                    'r2': r2_score(predictions['actual'], y_pred),
                    'confusion_matrix': {
                        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
                    }
                }
        
        return metrics

    def visualize_results(self, predictions: pd.DataFrame, metrics: Dict):
        """
        [MULTI] Create comprehensive visualizations for multiple model comparison"
        Create comprehensive visualizations of model performance
        """
        plt.style.use('seaborn')
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # 1. Predicted vs Actual for all models
        ax1 = axes[0, 0]
        model_columns = [col for col in predictions.columns if col.startswith('pred_')]
        colors = ['blue', 'green', 'red', 'purple']
        for model_col, color in zip(model_columns, colors):
            ax1.scatter(predictions['actual'], predictions[model_col], 
                     alpha=0.3, label=model_col.replace('pred_', ''),
                     color=color)
        
        max_val = max(max(predictions['actual']), max(predictions[model_columns].max()))
        ax1.plot([0, max_val], [0, max_val], 'r--')
        ax1.set_xlabel('Actual Goals')
        ax1.set_ylabel('Predicted Goals')
        ax1.set_title('Predicted vs Actual Goals')
        ax1.legend()
        
        # 2. Metrics across thresholds
        ax2 = axes[0, 1]
        for model_col, color in zip(model_columns, colors):
            accuracies = [metrics[t][model_col]['accuracy'] for t in self.thresholds]
            ax2.plot(self.thresholds, accuracies, marker='o', label=f"{model_col.replace('pred_', '')}_accuracy", color=color)
            
            precisions = [metrics[t][model_col]['precision'] for t in self.thresholds]
            ax2.plot(self.thresholds, precisions, marker='s', label=f"{model_col.replace('pred_', '')}_precision", 
                    color=color, linestyle='--')
        
        ax2.set_xlabel('Goal Threshold')
        ax2.set_ylabel('Score')
        ax2.set_title('Model Performance Across Thresholds')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 3. Error Distribution
        ax3 = axes[1, 0]
        for model_col, color in zip(model_columns, colors):
            errors = predictions[model_col] - predictions['actual']
            ax3.hist(errors, bins=20, alpha=0.5, color=color,
                    label=f"{model_col.replace('pred_', '')}\nMean: {errors.mean():.3f}\nStd: {errors.std():.3f}")
        ax3.set_xlabel('Prediction Error')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Error Distribution Comparison')
        ax3.legend()
        
        # 4. Summary Table
        ax4 = axes[1, 1]
        ax4.axis('off')
        summary_data = []
        metrics_to_show = ['rmse', 'mae', 'r2', 'accuracy']
        for metric in metrics_to_show:
            row = [f"{metrics[2.5][col][metric]:.3f}" for col in model_columns]
            summary_data.append(row)
        
        table = ax4.table(cellText=summary_data,
                         rowLabels=['RMSE', 'MAE', 'R²', 'Accuracy'],
                         colLabels=[col.replace('pred_', '') for col in model_columns],
                         loc='center',
                         cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax4.set_title('Model Performance Summary (2.5 goals threshold)')
        
        plt.tight_layout()
        plt.show()
        
    def print_detailed_metrics(self, metrics: Dict):
        """
        [MULTI] Print detailed metrics for all models and thresholds
        Print detailed metrics for all models and thresholds"""
        print("\n=== Overall Model Performance (All Models) ===")
        col_width = 35
        headers = ['Full Standard', 'Full Balanced', 'Reduced Standard', 'Reduced Balanced']
        model_keys = ['full_standard', 'full_balanced', 'reduced_standard', 'reduced_balanced']
        
        # Print headers
        header_line = ''.join([f"{header:<{col_width}}" for header in headers])
        print(header_line)
        print('-' * (col_width * len(headers)))

        # Overall metrics first
        metrics_2_5 = metrics[2.5]  # Using 2.5 as reference point for overall metrics
        for metric in ['rmse', 'mae', 'r2', 'accuracy']:
            metric_name = metric.upper() if metric.upper() != 'R2' else 'R² Score'
            print(f"\n{metric_name}:")
            values = [f"{metrics_2_5[f'pred_{key}'][metric]:.4f}" for key in model_keys]
            print(''.join([f"{val:<{col_width}}" for val in values]))

        # Threshold-specific metrics
        print("\n=== Threshold Performance ===")
        print("Class Definition for all thresholds:")
        print("Positive (1): Over X goals")
        print("Negative (0): Under X goals\n")

        for threshold in self.thresholds:
            print(f"\n{'='*120}")
            print(f"Threshold: Over {threshold} Goals")
            print('='*120)
            
            threshold_metrics = metrics[threshold]
            
            # Confusion Matrix Components
            print("\nConfusion Matrix Components:")
            for component in ['tp', 'tn', 'fp', 'fn']:
                component_names = {
                    'tp': 'True Positives (Correctly predicted Over)',
                    'tn': 'True Negatives (Correctly predicted Under)',
                    'fp': 'False Positives (Under predicted as Over)',
                    'fn': 'False Negatives (Over predicted as Under)'
                }
                values = [str(threshold_metrics[f'pred_{key}']['confusion_matrix'][component])
                         for key in model_keys]
                print(f"{component_names[component]}:")
                print(''.join([f"{val:<{col_width}}" for val in values]))
            
            # Classification Metrics
            print("\nClassification Performance Metrics:")
            for metric in ['accuracy', 'precision', 'recall', 'specificity', 'f1']:
                metric_names = {
                    'accuracy': 'Accuracy',
                    'precision': 'Precision (Of predicted Overs, % correct)',
                    'recall': 'Recall/Sensitivity (Of actual Overs, % caught)',
                    'specificity': 'Specificity (Of actual Unders, % caught)',
                    'f1': 'F1 Score'
                }
                values = [f"{threshold_metrics[f'pred_{key}'][metric]:.4f}"
                         for key in model_keys]
                print(f"{metric_names[metric]}:")
                print(''.join([f"{val:<{col_width}}" for val in values]))
            
            # Additional Metrics
            print("\nAdditional Metrics:")
            cm = threshold_metrics[f'pred_{model_keys[0]}']['confusion_matrix']
            actual_over = cm['tp'] + cm['fn']
            total_games = cm['tp'] + cm['tn'] + cm['fp'] + cm['fn']
            print(f"Total Games: {total_games}")
            print(f"Games Over {threshold}: {actual_over}")
            print(f"Games Under {threshold}: {total_games - actual_over}")

    
    def predict_with_configuration(self,
                                split_date: Optional[str] = None,
                                test_size: float = 0.2,
                                random_state: int = 42,
                                keep_pairs: bool = True,
                                use_full_features: bool = False,  # Default to False as we're using reduced
                                use_weights: bool = True,
                                weight_config: Optional[dict] = None,
                                xgb_params: Optional[dict] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        [SINGLE] Make predictions using a specific model configuration
        Make predictions using a specific model configuration
        
        Args:
            split_date: Optional date string (YYYY-MM-DD) for temporal split. If provided, 
                    uses temporal split instead of random split
            test_size: Proportion of data to use for testing (only used if split_date is None)
            random_state: Random seed for reproducibility (only used if split_date is None)
            keep_pairs: Whether to keep match pairs together in split (only used if split_date is None)
            use_full_features: Whether to use full or reduced feature set (defaults to False)
            use_weights: Whether to use sample weights
            weight_config: Optional dictionary of weight configuration parameters
            xgb_params: Optional dictionary of XGBoost parameters
        
        Returns:
            Tuple containing predictions DataFrame and test data dictionary
        """
        # Get data split based on method
        if split_date is not None:
            # Use temporal split
            train_data, test_data = self.split_data_temporal(split_date)
        else:
            # Use random split
            train_data, test_data = self.split_data_random(
                test_size=test_size,
                random_state=random_state,
                keep_pairs=keep_pairs
            )
        
        # Train model with specific configuration
        model = self._create_xgb_model(xgb_params)
        
        # Only apply weights if explicitly requested and config provided
        feature_set = 'full' if use_full_features else 'reduced'
        if use_weights and weight_config is not None:
            sample_weights = self.create_sample_weights(
                train_data[feature_set]['y'],
                weight_config
            )
            model.fit(train_data[feature_set]['X'], 
                    train_data[feature_set]['y'],
                    sample_weight=sample_weights)
        else:
            # No weights applied
            model.fit(train_data[feature_set]['X'],
                    train_data[feature_set]['y'])
        
        # Make predictions
        predictions = pd.DataFrame(index=test_data['indices'])
        predictions['actual'] = test_data[feature_set]['y']
        predictions['Predicted_Goals'] = model.predict(test_data[feature_set]['X'])
        
        # Add metadata columns
        info_mask = test_data['indices']
        for col in ['team', 'opponent', 'Season', 'game', 'Date']:
            predictions[col] = self.data.loc[info_mask, col]
        
        return predictions, test_data
    
    def visualize_single_model_results(self, predictions: pd.DataFrame, test_data: Dict):
        """[SINGLE] Create visualizations for single model configuration results"""
        plt.style.use('seaborn')
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # 1. Predicted vs Actual scatter plot
        ax1 = axes[0, 0]
        ax1.scatter(predictions['actual'], predictions['Predicted_Goals'], 
                   alpha=0.5, color='blue')
        
        max_val = max(max(predictions['actual']), max(predictions['Predicted_Goals']))
        ax1.plot([0, max_val], [0, max_val], 'r--')
        ax1.set_xlabel('Actual Goals')
        ax1.set_ylabel('Predicted Goals')
        ax1.set_title('Predicted vs Actual Goals')
        
        # 2. Error Distribution
        ax2 = axes[0, 1]
        errors = predictions['Predicted_Goals'] - predictions['actual']
        ax2.hist(errors, bins=20, alpha=0.7, color='blue')
        ax2.axvline(x=0, color='r', linestyle='--')
        ax2.set_xlabel('Prediction Error')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Error Distribution\nMean: {errors.mean():.3f}, Std: {errors.std():.3f}')
        
        # 3. Goals Distribution
        ax3 = axes[1, 0]
        ax3.hist(predictions['actual'], bins=range(0, int(max_val) + 2), 
                alpha=0.5, label='Actual', color='blue')
        ax3.hist(predictions['Predicted_Goals'], bins=range(0, int(max_val) + 2), 
                alpha=0.5, label='Predicted', color='red')
        ax3.set_xlabel('Goals')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Actual vs Predicted Goals')
        ax3.legend()
        
        # 4. Performance Metrics
        ax4 = axes[1, 1]
        ax4.axis('off')
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(predictions['actual'], 
                                             predictions['Predicted_Goals'])),
            'MAE': mean_absolute_error(predictions['actual'], 
                                     predictions['Predicted_Goals']),
            'R²': r2_score(predictions['actual'], 
                          predictions['Predicted_Goals'])
        }
        
        # Create metrics table
        cell_text = [[f"{value:.3f}"] for value in metrics.values()]
        table = ax4.table(cellText=cell_text,
                         rowLabels=list(metrics.keys()),
                         colLabels=['Value'],
                         loc='center',
                         cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        ax4.set_title('Model Performance Metrics')
        
        plt.tight_layout()
        plt.show()
        
        # Print additional analysis
        print("\nDetailed Performance Analysis:")
        print("=" * 50)
        print(f"Total Games: {len(predictions)}")
        for threshold in [0.5, 1.5, 2.5, 3.5, 4.5]:
            actual_over = (predictions['actual'] > threshold).mean()
            pred_over = (predictions['Predicted_Goals'] > threshold).mean()
            print(f"\nOver {threshold} goals:")
            print(f"Actual: {actual_over:.3f}")
            print(f"Predicted: {pred_over:.3f}")

class PoissonProbabilityCalculator:
    def __init__(self, gamma: float = 0.15):
        """
        Initialize calculator with correlation parameter
        
        Args:
            gamma: Correlation parameter for bivariate Poisson model
        """
        self.gamma = gamma
        self.thresholds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
        self.max_goals = 10

    def prepare_match_data(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert predictions DataFrame to match-level data, maintaining home/away team order
        based on game field format 'Home vs Away'
        
        Args:
            predictions_df: DataFrame with predictions from GoalModelEvaluator
        Returns:
            DataFrame with paired team predictions, preserving home/away order
        """
        # Create game identifier
        predictions_df['game_id'] = (predictions_df['Season'].astype(str) + '_' + 
                                predictions_df['game'].astype(str))
        
        matches = []
        for game_id in predictions_df['game_id'].unique():
            game_df = predictions_df[predictions_df['game_id'] == game_id]
            
            if len(game_df) == 2:  # Ensure we have both teams
                # Extract home and away teams from game field
                game_name = game_df.iloc[0]['game']  # Both rows have same game name
                home_team, away_team = game_name.split(' vs ')
                
                # Find corresponding rows
                home_row = game_df[game_df['team'] == home_team].iloc[0]
                away_row = game_df[game_df['team'] == away_team].iloc[0]
                
                # Which prediction column to use depends on what's available
                pred_cols = [col for col in game_df.columns if col.startswith('pred_')]
                pred_col = pred_cols[0] if pred_cols else 'Predicted_Goals'
                
                match_dict = {
                    'Date': home_row['Date'],
                    'Season': home_row['Season'],
                    'Game': home_row['game'],
                    'Team1': home_team,  # Home team from game field
                    'Team2': away_team,  # Away team from game field
                    'Actual_Goals1': home_row['actual'] if 'actual' in game_df.columns else home_row['teamGoals'],
                    'Actual_Goals2': away_row['actual'] if 'actual' in game_df.columns else away_row['teamGoals'],
                    'Predicted_Goals1': home_row[pred_col],
                    'Predicted_Goals2': away_row[pred_col]
                }
                matches.append(match_dict)
        
        return pd.DataFrame(matches)
        

    def calculate_match_probabilities(self, match_df: pd.DataFrame) -> List[Dict]:
        """
        Calculate detailed probabilities for each match with home/away team distinction
        
        Args:
            match_df: DataFrame with paired team predictions (Team1=Home, Team2=Away)
        Returns:
            List of dictionaries with detailed probability calculations
        """
        results = []
        for _, match in match_df.iterrows():
            # Ensure positive values for lambda (expected goals)
            lambda_home = max(0.01, float(match['Predicted_Goals1']))  # Home team
            lambda_away = max(0.01, float(match['Predicted_Goals2']))  # Away team
            
            # Calculate probability matrix
            # Note: Dimension ordering is [home_goals][away_goals] to match conventional notation
            prob_matrix = np.zeros((self.max_goals + 1, self.max_goals + 1))
            for home_goals in range(self.max_goals + 1):
                for away_goals in range(self.max_goals + 1):
                    p_home = poisson.pmf(home_goals, lambda_home)
                    p_away = poisson.pmf(away_goals, lambda_away)
                    
                    # Safe correlation calculation
                    if lambda_home > 0 and lambda_away > 0:
                        corr_term = 1 + self.gamma * (home_goals - lambda_home) * (away_goals - lambda_away) / \
                                np.sqrt(lambda_home * lambda_away)
                    else:
                        corr_term = 1
                    
                    prob_matrix[home_goals, away_goals] = p_home * p_away * max(corr_term, 0)
            
            # Ensure probabilities sum to 1
            total_prob = prob_matrix.sum()
            if total_prob > 0:
                prob_matrix = prob_matrix / total_prob
            
            # Calculate threshold probabilities with home/away distinction
            threshold_probs = {}
            for threshold in self.thresholds:
                total_probs = np.zeros_like(prob_matrix)
                for i in range(prob_matrix.shape[0]):
                    for j in range(prob_matrix.shape[1]):
                        if i + j > threshold:  # Total goals threshold
                            total_probs[i, j] = prob_matrix[i, j]
                
                over_prob = total_probs.sum()
                # Ensure probabilities are valid
                over_prob = min(max(over_prob, 0), 1)
                threshold_probs[threshold] = {
                    'over': over_prob,
                    'under': 1 - over_prob,
                    # Add match outcome probabilities
                    'home_win': np.sum(prob_matrix[np.triu_indices_from(prob_matrix, k=1)]),  # Home team scores more
                    'draw': np.sum(prob_matrix[np.diag_indices_from(prob_matrix)]),  # Equal scores
                    'away_win': np.sum(prob_matrix[np.tril_indices_from(prob_matrix, k=-1)])  # Away team scores more
                }
            
            results.append({
                'Date': match['Date'],
                'Team1': match['Team1'],  # Home team
                'Team2': match['Team2'],  # Away team
                'expected_goals_home': lambda_home,
                'expected_goals_away': lambda_away,
                'actual_goals_home': match['Actual_Goals1'],
                'actual_goals_away': match['Actual_Goals2'],
                'score_distribution': prob_matrix,  # [home_goals][away_goals]
                'threshold_probabilities': threshold_probs,
                'match_probabilities': {
                    'home_win': np.sum(prob_matrix[np.triu_indices_from(prob_matrix, k=1)]),
                    'draw': np.sum(prob_matrix[np.diag_indices_from(prob_matrix)]),
                    'away_win': np.sum(prob_matrix[np.tril_indices_from(prob_matrix, k=-1)])
                }
            })
        
        return results

    def analyze_multi_threshold_predictions(self, results: List[Dict]) -> pd.DataFrame:
        """
        Analyze predictions at individual game level for all thresholds
        """
        game_analysis = []
        
        for game in results:
            base_info = {
                'Date': game['Date'],
                'Match': f"{game['Team1']} vs {game['Team2']}",
                'Actual_Total': game['actual_goals_home'] + game['actual_goals_away'],
                'Predicted_Total': game['expected_goals_home'] + game['expected_goals_away'],
                'Team1': game['Team1'],
                'Team2': game['Team2'],
                'Team1_Expected': game['expected_goals_home'],
                'Team2_Expected': game['expected_goals_away'],
                'Team1_Actual': game['actual_goals_home'],
                'Team2_Actual': game['actual_goals_away']
            }
            
            for threshold in self.thresholds:
                # Get probabilities
                over_prob = game['threshold_probabilities'][threshold]['over']
                under_prob = game['threshold_probabilities'][threshold]['under']
                
                # Actual outcomes
                actual_total = base_info['Actual_Total']
                is_actually_over = (actual_total > threshold)
                is_actually_under = (actual_total <= threshold)
                
                # Predictions
                is_predicted_over = (over_prob > 0.5)
                is_predicted_under = (under_prob > 0.5)
                
                # Calculate correctness
                correct_over = (is_predicted_over == is_actually_over)
                correct_under = (is_predicted_under == is_actually_under)
                
                # Get match outcome probabilities if available
                match_probs = game['threshold_probabilities'][threshold]
                
                threshold_info = {
                    # Raw probabilities
                    f'Over_{threshold}_Prob': over_prob,
                    f'Under_{threshold}_Prob': under_prob,
                    
                    # Match outcome probabilities
                    f'Home_Win_Prob_{threshold}': match_probs.get('home_win', None),
                    f'Draw_Prob_{threshold}': match_probs.get('draw', None),
                    f'Away_Win_Prob_{threshold}': match_probs.get('away_win', None),
                    
                    # Predictions and actuals
                    f'Predicted_Over_{threshold}': is_predicted_over,
                    f'Predicted_Under_{threshold}': is_predicted_under,
                    f'Actually_Over_{threshold}': is_actually_over,
                    f'Actually_Under_{threshold}': is_actually_under,
                    
                    # Correctness columns
                    f'Correct_Over_{threshold}': correct_over,
                    f'Correct_Under_{threshold}': correct_under,
                    f'Correct_{threshold}': correct_over,  # For backward compatibility
                    
                    # Confidence
                    f'Confidence_{threshold}': abs(over_prob - 0.5) * 2
                }
                
                base_info.update(threshold_info)
            
            game_analysis.append(base_info)
        
        return pd.DataFrame(game_analysis)

    def visualize_multi_threshold_predictions(self, analysis_df: pd.DataFrame) -> None:
        """
        Create visualizations for multi-threshold prediction analysis
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        plt.style.use('seaborn')
        fig = plt.figure(figsize=(20, 15))

        # 1. Overall Accuracy by Threshold (Bar chart for Over and Under)
        plt.subplot(2, 2, 1)
        over_accuracies = []
        under_accuracies = []
        over_counts = []
        under_counts = []
        threshold_labels = []

        for threshold in self.thresholds:
            # Get games that went over/under
            actual_over_mask = analysis_df[f'Actually_Over_{threshold}']
            actual_under_mask = analysis_df[f'Actually_Under_{threshold}']
            
            # Calculate accuracy for games that actually went over
            over_games = analysis_df[actual_over_mask]
            if len(over_games) > 0:
                over_acc = (over_games[f'Predicted_Over_{threshold}'] == True).mean()
                over_accuracies.append(over_acc)
                over_counts.append(len(over_games))
            else:
                over_accuracies.append(0)
                over_counts.append(0)
            
            # Calculate accuracy for games that actually went under
            under_games = analysis_df[actual_under_mask]
            if len(under_games) > 0:
                under_acc = (under_games[f'Predicted_Under_{threshold}'] == True).mean()
                under_accuracies.append(under_acc)
                under_counts.append(len(under_games))
            else:
                under_accuracies.append(0)
                under_counts.append(0)
                
            threshold_labels.append(str(threshold))

        x_positions = np.arange(len(self.thresholds))
        bar_width = 0.35

        plt.bar(x_positions, over_accuracies, width=bar_width, alpha=0.6, 
                label='Accuracy on Actual Overs', color='skyblue')
        plt.bar(x_positions + bar_width, under_accuracies, width=bar_width, alpha=0.6, 
                label='Accuracy on Actual Unders', color='lightgreen')

        plt.axhline(y=0.5, color='r', linestyle='--', label='Random Guess')
        plt.xticks(x_positions + bar_width / 2, threshold_labels, rotation=45)
        plt.ylabel('Accuracy')
        plt.title('Prediction Accuracy by Threshold (Over vs Under)')
        plt.legend()

        # Add value labels with counts
        for i, (acc, count) in enumerate(zip(over_accuracies, over_counts)):
            plt.text(i, acc, f'{acc:.3f}\n(n={count})', ha='center', va='bottom')
        for i, (acc, count) in enumerate(zip(under_accuracies, under_counts)):
            plt.text(i + bar_width, acc, f'{acc:.3f}\n(n={count})', ha='center', va='bottom')

        # Rest of visualization code remains the same...
        plt.tight_layout()
        plt.show()

        # Print detailed analysis
        print("\nDetailed Threshold Analysis:")
        print("=" * 80)

        for threshold in self.thresholds:
            print(f"\nThreshold: {threshold}")
            print("-" * 50)
            
            actual_over_mask = analysis_df[f'Actually_Over_{threshold}']
            actual_under_mask = analysis_df[f'Actually_Under_{threshold}']
            
            over_games = analysis_df[actual_over_mask]
            under_games = analysis_df[actual_under_mask]
            
            print(f"Games that went Over {threshold}: {len(over_games)}")
            if len(over_games) > 0:
                over_acc = (over_games[f'Predicted_Over_{threshold}'] == True).mean()
                print(f"  - Correctly predicted Over: {over_acc:.3f}")
                print(f"  - Average predicted probability of Over: {over_games[f'Over_{threshold}_Prob'].mean():.3f}")
            
            print(f"\nGames that went Under {threshold}: {len(under_games)}")
            if len(under_games) > 0:
                under_acc = (under_games[f'Predicted_Under_{threshold}'] == True).mean()
                print(f"  - Correctly predicted Under: {under_acc:.3f}")
                print(f"  - Average predicted probability of Under: {under_games[f'Under_{threshold}_Prob'].mean():.3f}")

            # High confidence analysis
            high_conf_mask = analysis_df[f'Confidence_{threshold}'] > 0.75
            if high_conf_mask.sum() > 0:
                high_conf_df = analysis_df[high_conf_mask]
                
                high_conf_over = high_conf_df[high_conf_df[f'Actually_Over_{threshold}'] == True]
                high_conf_under = high_conf_df[high_conf_df[f'Actually_Under_{threshold}'] == True]
                
                print(f"\nHigh Confidence Predictions:")
                if len(high_conf_over) > 0:
                    over_acc = (high_conf_over[f'Predicted_Over_{threshold}'] == True).mean()
                    print(f"When Actually Over - Correct: {over_acc:.3f} (n={len(high_conf_over)})")
                
                if len(high_conf_under) > 0:
                    under_acc = (high_conf_under[f'Predicted_Under_{threshold}'] == True).mean()
                    print(f"When Actually Under - Correct: {under_acc:.3f} (n={len(high_conf_under)})")



    # ---- Not used yet but could be good to optimise gamma which is how independent goals are of each other for the teams ---
            
    def optimize_gamma(self, match_df: pd.DataFrame, method: str = 'likelihood') -> float:
        """
        Find optimal gamma parameter using historical match data
        
        Args:
            match_df: DataFrame with actual goals scored by both teams
            method: 'likelihood' or 'calibration'
        
        Returns:
            Optimal gamma value
        """
        def negative_log_likelihood(gamma):
            self.gamma = gamma
            total_nll = 0
            
            for _, match in match_df.iterrows():
                g1 = match['Actual_Goals1']
                g2 = match['Actual_Goals2']
                
                # Calculate probability of observed outcome using actual means as lambdas
                lambda1 = match_df['Actual_Goals1'].mean()
                lambda2 = match_df['Actual_Goals2'].mean()
                
                p1 = poisson.pmf(g1, lambda1)
                p2 = poisson.pmf(g2, lambda2)
                
                corr_term = 1 + gamma * (g1 - lambda1) * (g2 - lambda2) / \
                            np.sqrt(lambda1 * lambda2)
                prob = p1 * p2 * max(corr_term, 0)
                
                # Add to negative log-likelihood
                total_nll -= np.log(prob + 1e-10)
                
            return total_nll
        
        def calibration_error(gamma):
            self.gamma = gamma
            
            # Calculate total goals for each match
            actual_totals = match_df['Actual_Goals1'] + match_df['Actual_Goals2']
            mean_total = actual_totals.mean()
            
            # Calculate error for different thresholds
            total_error = 0
            for threshold in self.thresholds:
                actual_overs = (actual_totals > threshold).mean()
                # Use historical average for predictions
                pred_overs = (mean_total > threshold)
                error = (actual_overs - pred_overs) ** 2
                total_error += error
                
            return total_error

        # Choose optimization function based on method
        obj_func = negative_log_likelihood if method == 'likelihood' else calibration_error
        
        # Optimize gamma
        result = minimize(obj_func, 
                        x0=0.15,
                        bounds=[(0, 0.5)],
                        method='L-BFGS-B')
        
        optimal_gamma = result.x[0]
        self.gamma = optimal_gamma
        
        return optimal_gamma

    def compare_gamma_values(self, match_df: pd.DataFrame, 
                            gammas: List[float] = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]) -> pd.DataFrame:
        """
        Compare performance of different gamma values
        
        Args:
            match_df: DataFrame with actual goals scored by both teams
            gammas: List of gamma values to test
        
        Returns:
            DataFrame with performance metrics for each gamma
        """
        results = []
        actual_totals = match_df['Actual_Goals1'] + match_df['Actual_Goals2']
        
        for gamma in gammas:
            self.gamma = gamma
            metrics = {}
            
            # Calculate metrics for each threshold
            for threshold in self.thresholds:
                actual_overs = (actual_totals > threshold).mean()
                metrics[f'actual_rate_{threshold}'] = actual_overs
                
                # Calculate how gamma affects goal independence
                corrected_rate = actual_overs * (1 + gamma)  # Simplified correlation effect
                metrics[f'adjusted_rate_{threshold}'] = corrected_rate
                metrics[f'bias_{threshold}'] = corrected_rate - actual_overs
            
            metrics['gamma'] = gamma
            results.append(metrics)
        
        return pd.DataFrame(results)