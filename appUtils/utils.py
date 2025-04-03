import pandas as pd
import os
import logging

DATA_DIR = os.path.join("Data And Scripts", "Data")

def save_historical_odds(odds_df: pd.DataFrame, market_type: str) -> None:
    try:
        filename = f'historical_odds_{market_type}.csv'
        filepath = os.path.join(DATA_DIR, filename)
        
        # Ensure we have a clean DataFrame with consistent columns
        required_columns = ['eventName', 'marketType', 'date', 'under', 'over', 'timestamp']
        current_odds = odds_df[required_columns].copy()
        
        # Add timestamp if missing
        if 'timestamp' not in current_odds.columns:
            current_odds['timestamp'] = pd.Timestamp.now()
        
        if os.path.exists(filepath):
            # Load historical data
            historical_df = pd.read_csv(filepath)
            historical_df['timestamp'] = pd.to_datetime(historical_df['timestamp'])
            
            # Ensure historical data has same columns
            historical_df = historical_df[required_columns]
            
            # Convert both DataFrames to simple structures for clean concatenation
            historical_simple = pd.DataFrame(historical_df.values, columns=historical_df.columns)
            current_simple = pd.DataFrame(current_odds.values, columns=current_odds.columns)

            # Combine historical and current data
            combined_df = pd.concat([historical_simple, current_simple], axis=0, ignore_index=True)
            
            # Remove exact duplicates (same event, market, date, odds, and timestamp)
            combined_df = combined_df.drop_duplicates(keep='last')
            
            # For same event/market combinations with different odds/timestamps,
            # keep all records to maintain price history
            combined_df = combined_df.sort_values(['eventName', 'marketType', 'timestamp'])
            
        else:
            combined_df = current_odds
        
        # Convert numeric columns
        for col in ['over', 'under']:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

        # Ensure timestamp is in consistent format
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S.%f')

        combined_df = combined_df.dropna()

        # Save everything without any filtering
        combined_df.to_csv(filepath, index=False)

        logging.info(f"Saved historical {market_type} odds ({len(combined_df)} records)")

    except Exception as e:
        logging.error(f"Error saving historical {market_type} odds: {str(e)}")
        raise

