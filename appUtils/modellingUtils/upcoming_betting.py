import pandas as pd
import numpy as np
from datetime import datetime

def map_market_to_probs(market_type: str, prob_cols: list) -> tuple:
    """Maps market type to probability columns"""
    try:
        # Extract goal line from market type (e.g., '05' -> '0.5')
        goal_num = int(market_type.split('_')[-1])
        goal_line = f"{goal_num/10:.1f}"
        
        # Find matching probability columns
        over_col = f"Over_{goal_line}_Prob"
        under_col = f"Under_{goal_line}_Prob"
        
        if over_col in prob_cols and under_col in prob_cols:
            return over_col, under_col
            
    except Exception as e:
        print(f"Error mapping market {market_type}: {str(e)}")
    
    return None, None

def calculate_kelly(prob: float, odds: float, bankroll: float) -> float:
    """Calculates pure Kelly stake"""
    b = odds - 1
    p = prob
    q = 1 - p
    kelly = (b * p - q) / b
    return max(0, kelly * bankroll)

def analyze_upcoming_value_bets(
    predictions_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    initial_bankroll: float = 100.0,
    max_daily_exposure: float = 0.15,
    edge_threshold: float = 0.03,
    min_probability: float = 0.5
) -> tuple:
    """
    Analyzes potential value bets for upcoming matches with proportional Kelly allocation
    
    Args:
        predictions_df: DataFrame with model predictions
        odds_df: DataFrame with odds history
        initial_bankroll: Starting bankroll amount
        max_daily_exposure: Maximum proportion of bankroll to risk per day
        edge_threshold: Minimum edge required for a bet
        min_probability: Minimum probability required for a bet
        
    Returns:
        tuple: (DataFrame of potential bets, DataFrame of market analysis)
    """
    print("\nStarting value bet analysis...")
    
    # Get probability columns
    prob_cols = [col for col in predictions_df.columns if '_Prob' in col]
    print(f"Found {len(prob_cols)} probability columns")
    
    # Prepare odds data
    latest_odds = odds_df.sort_values('timestamp').groupby(
        ['eventName', 'marketType']
    ).last().reset_index()
    
    # Track odds history with minimum prices
    odds_history = []
    for (event, market), group in odds_df.groupby(['eventName', 'marketType']):
        group = group.sort_values('timestamp')
        odds_history.append({
            'event': event,
            'market': market,
            'min_over': group['over'].min(),
            'max_over': group['over'].max(),
            'min_under': group['under'].min(),
            'max_under': group['under'].max(),
            'current_over': group['over'].iloc[-1],
            'current_under': group['under'].iloc[-1],
            'price_updates': len(group),
            'last_update': group['timestamp'].iloc[-1]
        })
    
    odds_history_df = pd.DataFrame(odds_history)
    
    # Initialize containers
    potential_bets = []
    market_analysis = []
    current_bankroll = initial_bankroll
    
    # Process each match date
    for date, day_predictions in predictions_df.groupby('Date'):
        print(f"\nAnalyzing bets for {date}")
        day_bets = []
        daily_limit = max_daily_exposure * current_bankroll
        
        # Process each match
        for _, pred_row in day_predictions.iterrows():
            match = pred_row['Match']
            print(f"Analyzing {match}")
            
            match_odds = latest_odds[latest_odds['eventName'] == match]
            if len(match_odds) == 0:
                print(f"No odds found for {match}")
                continue
            
            # Process each market
            for _, odds_row in match_odds.iterrows():
                market_type = odds_row['marketType']
                print(f"Processing market: {market_type}")
                
                # Map market to probability columns
                over_col, under_col = map_market_to_probs(market_type, prob_cols)
                
                if not over_col or not under_col:
                    print(f"No matching probability columns for {market_type}")
                    continue
                
                # Get probabilities and calculate edges
                over_prob = pred_row[over_col]
                under_prob = pred_row[under_col]
                
                implied_over_prob = 1 / odds_row['over']
                implied_under_prob = 1 / odds_row['under']
                
                over_edge = over_prob - implied_over_prob
                under_edge = under_prob - implied_under_prob
                
                # Store market analysis
                market_info = {
                    'Date': date,
                    'Match': match,
                    'Market': market_type,
                    'Over_Prob': over_prob,
                    'Under_Prob': under_prob,
                    'Current_Over_Odds': odds_row['over'],
                    'Current_Under_Odds': odds_row['under'],
                    'Over_Edge': over_edge,
                    'Under_Edge': under_edge,
                    'Last_Updated': odds_row['timestamp']
                }
                
                # Add historical context if available
                history = odds_history_df[
                    (odds_history_df['event'] == match) & 
                    (odds_history_df['market'] == market_type)
                ]
                if len(history) > 0:
                    market_info.update({
                        'Min_Over_Odds': history['min_over'].iloc[0],
                        'Max_Over_Odds': history['max_over'].iloc[0],
                        'Min_Under_Odds': history['min_under'].iloc[0],
                        'Max_Under_Odds': history['max_under'].iloc[0],
                        'Price_Updates': history['price_updates'].iloc[0],
                        'Last_Update': history['last_update'].iloc[0]
                    })
                
                market_analysis.append(market_info)
                
                # Calculate pure Kelly stakes for potential value bets
                if over_edge > edge_threshold and over_prob > min_probability:
                    pure_kelly = calculate_kelly(
                        prob=over_prob,
                        odds=odds_row['over'],
                        bankroll=current_bankroll
                    )
                    
                    day_bets.append({
                        'Date': date,
                        'Match': match,
                        'Market': market_type,
                        'Bet_Type': 'Over',
                        'Odds': odds_row['over'],
                        'Probability': over_prob,
                        'Edge': over_edge,
                        'Pure_Kelly': pure_kelly,
                        'Expected_Value': over_prob * odds_row['over'] - 1,
                        'Last_Updated': odds_row['timestamp']
                    })
                    
                elif under_edge > edge_threshold and under_prob > min_probability:
                    pure_kelly = calculate_kelly(
                        prob=under_prob,
                        odds=odds_row['under'],
                        bankroll=current_bankroll
                    )
                    
                    day_bets.append({
                        'Date': date,
                        'Match': match,
                        'Market': market_type,
                        'Bet_Type': 'Under',
                        'Odds': odds_row['under'],
                        'Probability': under_prob,
                        'Edge': under_edge,
                        'Pure_Kelly': pure_kelly,
                        'Expected_Value': under_prob * odds_row['under'] - 1,
                        'Last_Updated': odds_row['timestamp']
                    })
        
        # Calculate total pure Kelly for the day
        total_kelly = sum(bet['Pure_Kelly'] for bet in day_bets)
        
        # Allocate stakes proportionally within daily exposure
        if total_kelly > 0:  # Avoid division by zero
            for bet in day_bets:
                # Calculate proportional stake based on Kelly ratios
                kelly_fraction = bet['Pure_Kelly'] / total_kelly
                bet['Stake'] = kelly_fraction * daily_limit
                
                if bet['Stake'] >= 0.01:  # Minimum stake threshold
                    potential_bets.append(bet)
                    print(f"Added {bet['Bet_Type']} bet on {bet['Match']} - {bet['Market']}")
    
    print(f"\nAnalysis complete - Found {len(potential_bets)} potential bets")
    return pd.DataFrame(potential_bets), pd.DataFrame(market_analysis)

def analyze_odds_movement(current: float, min_odds: float, value_target: float) -> dict:
    """
    Analyzes odds movement relative to value target
    
    Args:
        current: Current odds
        min_odds: Minimum odds seen
        value_target: Odds needed for value bet
        
    Returns:
        Dictionary containing movement analysis
    """
    movement = current - min_odds
    
    return {
        'movement': movement,
        'min_odds': min_odds,
        'current_odds': current,
        'movement_percentage': (movement / min_odds) * 100 if min_odds != 0 else 0
    }

def calculate_value_odds(probability: float, edge_threshold: float) -> float:
    """
    Calculates the minimum odds needed for a value bet given probability and edge threshold
    
    Args:
        probability: Model's predicted probability
        edge_threshold: Minimum required edge
        
    Returns:
        Minimum odds needed for a value bet
    """
    try:
        value_odds = 1 / (probability - edge_threshold)
        return value_odds if value_odds > 1 else float('inf')
    except:
        return float('inf')

def generate_upcoming_bets_report(
    potential_bets: pd.DataFrame,
    market_analysis: pd.DataFrame,
    initial_bankroll: float,
    max_daily_exposure: float = 0.15,
    edge_threshold: float = 0.03
) -> str:
    """
    Generates a concise report focusing on potential betting opportunities,
    organized by match with markets as sub-sections
    """
    report = []
    report.append("# Current Value Bets\n")
    
    # Track value bets already placed
    placed_bets = set()
    if len(potential_bets) > 0:
        total_stake = potential_bets['Stake'].sum()
        report.append(f"**Total Stake**: £{total_stake:.2f} ({(total_stake/initial_bankroll)*100:.1f}% of bankroll)")
        
        # Track matches and markets with active bets
        for _, bet in potential_bets.iterrows():
            placed_bets.add((bet['Match'], bet['Market'], bet['Bet_Type']))
    
    # Group analysis by date and match
    report.append("\n# Potential Opportunities\n")
    for date, day_markets in market_analysis.groupby('Date'):
        report.append(f"## {date}\n")
        
        # Group by match
        for match, match_markets in day_markets.groupby('Match'):
            has_opportunities = False
            match_content = []
            match_content.append(f"### {match}")
            
            # Process each market
            for _, market in match_markets.iterrows():
                market_opportunities = []
                
                # Check Over market
                if (market['Over_Prob'] > 0.5 and 
                    (match, market['Market'], 'Over') not in placed_bets):
                    implied_prob_over = 1 / market['Current_Over_Odds']
                    edge_over = market['Over_Edge']
                    if edge_over > 0:  # Show any positive edge for consideration
                        market_opportunities.append(
                            f"- Over @ {market['Current_Over_Odds']:.2f} "
                            f"(Edge: {edge_over*100:.1f}%, "
                            f"Model: {market['Over_Prob']*100:.1f}%, "
                            f"Market: {implied_prob_over*100:.1f}%)"
                        )
                
                # Check Under market
                if (market['Under_Prob'] > 0.5 and 
                    (match, market['Market'], 'Under') not in placed_bets):
                    implied_prob_under = 1 / market['Current_Under_Odds']
                    edge_under = market['Under_Edge']
                    if edge_under > 0:  # Show any positive edge for consideration
                        market_opportunities.append(
                            f"- Under @ {market['Current_Under_Odds']:.2f} "
                            f"(Edge: {edge_under*100:.1f}%, "
                            f"Model: {market['Under_Prob']*100:.1f}%, "
                            f"Market: {implied_prob_under*100:.1f}%)"
                        )
                
                # If we found opportunities for this market, add them
                if market_opportunities:
                    has_opportunities = True
                    market_content = [f"#### {market['Market'].replace('OVER_UNDER_', 'Over/Under ').replace('_', '.')}"]
                    market_content.extend(market_opportunities)
                    match_content.extend(market_content)
            
            # Only add match section if it has opportunities
            if has_opportunities:
                report.extend(match_content)
                report.append("")  # Add spacing between matches
    
    return "\n".join(report)