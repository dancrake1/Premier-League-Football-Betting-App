import logging
import datetime
import json
import pandas as pd
import requests
import urllib
import re
from typing import List, Dict
import os

class BetfairAPI:
    def __init__(self, username: str, password: str, app_key: str, cert_path: str, key_path: str):
        self.username = username
        self.password = urllib.parse.quote(password)
        self.app_key = app_key
        self.cert_path = cert_path
        self.key_path = key_path
        self.session_token = None
        self.bet_url = "https://api.betfair.com/exchange/betting/json-rpc/v1"
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('betfair.log'),
                logging.StreamHandler()
            ]
        )
        
        # Team name mappings
        self.name_mapping = {
            'Leeds': 'Leeds United',
            'Leicester': 'Leicester City',
            'Man Utd': 'Manchester Utd',
            'Man City': 'Manchester City',
            'Nottm Forest': "Nott'ham Forest",
            'Newcastle': 'Newcastle Utd',
            'Norwich': 'Norwich City',
            'Sheff Utd': 'Sheffield United',
            'Luton': 'Luton Town',
            'Ipswich': 'Ipswich Town'
        }
        
        self.login()

    def login(self) -> None:
        """Login to Betfair API and get session token"""
        try:
            payload = f'username={self.username}&password={self.password}'
            headers = {
                'X-Application': self.app_key,
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            resp = requests.post(
                'https://identitysso-cert.betfair.com/api/certlogin',
                data=payload,
                cert=(self.cert_path, self.key_path),
                headers=headers
            )
            
            if resp.status_code == 200:
                resp_json = resp.json()
                self.session_token = resp_json['sessionToken']
                logging.info("Successfully logged in to Betfair")
            else:
                logging.error(f"Login failed with status code: {resp.status_code}")
                raise Exception("Betfair login failed")
                
        except Exception as e:
            logging.error(f"Error during login: {str(e)}")
            raise

    def get_headers(self) -> Dict:
        """Get headers for API requests"""
        return {
            'X-Application': self.app_key,
            'X-Authentication': self.session_token,
            'content-type': 'application/json'
        }

    def chunks(self, lst: List, n: int):
        """Split list into chunks of size n"""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def get_events(self, from_date: datetime.date, to_date: datetime.date) -> pd.DataFrame:
        """Get Premier League events between dates"""
        try:
            from_time = from_date.strftime('%Y-%m-%dT%H:%M:%SZ')
            to_time = to_date.strftime('%Y-%m-%dT%H:%M:%SZ')
            
            request_data = {
                "jsonrpc": "2.0",
                "method": "SportsAPING/v1.0/listEvents",
                "params": {
                    "filter": {
                        "eventTypeIds": [1],
                        "competitionIds": [10932509],
                        "marketStartTime": {
                            "from": from_time,
                            "to": to_time
                        },
                        "countryCode": ["GB"]
                    },
                    "maxResults": "600"
                },
                "id": 1
            }
            
            response = requests.post(
                self.bet_url,
                data=json.dumps(request_data),
                headers=self.get_headers()
            )
            
            if response.status_code != 200:
                raise Exception(f"API request failed with status code: {response.status_code}")
                
            events = response.json()
            all_events = events['result']
            
            extracted_data = [
                {
                    'id': event['event']['id'],
                    'name': event['event']['name'],
                    'date': event['event']['openDate']
                }
                for event in all_events if event['event']['name'] != 'Daily Goals'
            ]
            
            return pd.DataFrame(extracted_data)
            
        except Exception as e:
            logging.error(f"Error getting events: {str(e)}")
            raise

    def get_market_catalogue(self, event_list: List[str]) -> List:
        """Get market catalogue for events"""
        request_data = {
            "jsonrpc": "2.0",
            "method": "SportsAPING/v1.0/listMarketCatalogue",
            "params": {
                "filter": {
                    "eventIds": event_list,
                },
                "maxResults": "400",
                "marketProjection": [
                    "EVENT",
                    "RUNNER_DESCRIPTION",
                    "RUNNER_METADATA",
                    "MARKET_START_TIME"
                ]
            },
            "id": 1
        }
        
        response = requests.post(
            self.bet_url,
            data=json.dumps(request_data),
            headers=self.get_headers()
        )
        
        if response.status_code != 200:
            raise Exception(f"API request failed with status code: {response.status_code}")
            
        return response.json()['result']

    def get_market_books(self, market_ids: List[str]) -> List:
        """Get market books for market IDs"""
        request_data = {
            "jsonrpc": "2.0",
            "method": "SportsAPING/v1.0/listMarketBook",
            "params": {
                "marketIds": market_ids,
                "priceProjection": {
                    "priceData": ["EX_BEST_OFFERS"],
                    "maxResults": "400",
                }
            },
            "id": 1
        }
        
        response = requests.post(
            self.bet_url,
            data=json.dumps(request_data),
            headers=self.get_headers()
        )
        
        if response.status_code != 200:
            raise Exception(f"API request failed with status code: {response.status_code}")
            
        return response.json()['result']

    def get_odds(self, market_type: str = 'goals') -> pd.DataFrame:
        """Get odds for either goals or cards markets"""
        try:
            # Get events for next 7 days
            today = datetime.date.today()
            events_df = self.get_events(today, today + datetime.timedelta(days=7))
            
            if events_df.empty:
                return pd.DataFrame()
                
            # Get market catalogue in batches
            event_ids = events_df.id.tolist()
            event_books = []
            for batch in self.chunks(event_ids, 2):
                event_books.extend(self.get_market_catalogue(batch))
            
            # Filter for relevant markets
            pattern = r'Over/Under \d+\.5 Goals' if market_type == 'goals' else r'Cards Over/Under \d+\.5'
            events_o_u = [event for event in event_books if re.match(pattern, event['marketName'])]
            market_ids = [event['marketId'] for event in events_o_u]
            
            # Get market books in batches
            market_books = []
            for batch in self.chunks(market_ids, 20):
                market_books.extend(self.get_market_books(batch))
            
            # Extract market data
            markets = pd.DataFrame([
                {
                    'id': event['marketId'],
                    'name': event['event']['name'],
                    'marketName': event['marketName'],
                    'date': event['event']['openDate'],
                    **{
                        ('overId' if 'Over' in runner['runnerName'] else 'underId'): 
                        int(runner['selectionId']) for runner in event['runners']
                    }
                }
                for event in events_o_u
            ])
            
            if markets.empty:
                return pd.DataFrame()
            
            # Extract price data
            prices = pd.DataFrame([
                {
                    'marketId': book['marketId'],
                    'selectionId': runner['selectionId'],
                    'maxAvailableToBackPrice': max(
                        [price['price'] for price in runner['ex']['availableToBack']]
                    ) if runner['ex']['availableToBack'] else None
                }
                for book in market_books
                for runner in book['runners']
            ])
            
            # Merge data
            df = pd.merge(
                markets, prices,
                how='left',
                left_on=['id', 'underId'],
                right_on=['marketId', 'selectionId']
            ).drop(columns=['marketId', 'selectionId']).rename(columns={'maxAvailableToBackPrice': 'under'})
            
            df = pd.merge(
                df, prices,
                how='left',
                left_on=['id', 'overId'],
                right_on=['marketId', 'selectionId']
            ).drop(columns=['marketId', 'selectionId']).rename(columns={'maxAvailableToBackPrice': 'over'})
            
            # Clean team names
            for old, new in self.name_mapping.items():
                df['name'] = df['name'].str.replace(old, new)
            df['name'] = df['name'].str.replace(' v ', ' vs ')
            
            # Clean market names
            def generate_replacement(value):
                cleaned = value.replace('Over/Under ', '').replace('.', '').replace(' ', '_')
                if market_type == 'goals':
                    cleaned = cleaned.split('_')[0]
                else:
                    cleaned = cleaned.split('_')[-1]
                return f'OVER_UNDER_{cleaned.upper()}'
            
            df['marketName'] = df['marketName'].apply(generate_replacement)
            
            # Rename columns and select final columns
            df = df.rename(columns={
                'name': 'eventName',
                'marketName': 'marketType'
            })[['eventName', 'marketType', 'date', 'under', 'over']]
            
            # Add timestamp
            df['timestamp'] = pd.Timestamp.now()
            
            return df
            
        except Exception as e:
            logging.error(f"Error getting {market_type} odds: {str(e)}")
            return pd.DataFrame()

    def save_odds(self, odds_df: pd.DataFrame, market_type: str) -> None:
        """Save odds to CSV file"""
        try:
            filename = f'upcoming_odds_{market_type}.csv'
            filepath = os.path.join('Data And Scripts/Data', filename)
            odds_df.to_csv(filepath, index=False)
            logging.info(f"Saved {market_type} odds to {filepath}")
        except Exception as e:
            logging.error(f"Error saving {market_type} odds: {str(e)}")

def get_odds(self, market_type: str = 'goals') -> pd.DataFrame:
    """Get odds for either goals or cards markets"""
    try:
        # If session token is expired, login again
        if self.session_token is None:
            self.login()
            
        # Get events for next 7 days
        today = datetime.date.today()
        events_df = self.get_events(today, today + datetime.timedelta(days=7))
        
        if events_df.empty:
            return pd.DataFrame()
            
        # Rest of your existing get_odds code...
            
    except requests.exceptions.RequestException as e:
        if "NOT_AUTHORIZED" in str(e):
            logging.warning("Session expired, attempting to re-login")
            self.login()  # Re-login and try again
            return self.get_odds(market_type)
        else:
            logging.error(f"Error in get_odds: {str(e)}")
            raise
    except Exception as e:
        logging.error(f"Error getting {market_type} odds: {str(e)}")
        return pd.DataFrame()
