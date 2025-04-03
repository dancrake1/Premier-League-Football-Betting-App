import pandas as pd
from bs4 import BeautifulSoup
import requests
import os
from datetime import datetime
import unidecode
import time
import json
import argparse
import logging

LOGPATH = '/Users/danielcrake/Desktop/Football Betting 2025/logs'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{LOGPATH}/skysports_scraper.log'),
        logging.StreamHandler()
    ]
)

def create_session():
    """Create a requests session with headers"""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    return session

def get_gameweek_dates(gameweek, season, fixtures_file='Data And Scripts/Data/fixtures.csv'):
    """Get all dates for a given gameweek in the current season"""
    try:
        fixtures = pd.read_csv(fixtures_file)
        logging.info(f"Looking for matches in Season {season}, Gameweek {gameweek}")
        
        # Filter for current season and gameweek
        gameweek_fixtures = fixtures[(fixtures['Season'] == season) & (fixtures['Wk'] == gameweek)]
        
        # Log the filtered data
        logging.info(f"Found {len(gameweek_fixtures)} matches for this gameweek")
        if len(gameweek_fixtures) > 0:
            logging.info(f"Match details:\n{gameweek_fixtures[['Date', 'Home', 'Away']].to_string()}")
        
        dates = pd.to_datetime(gameweek_fixtures['Date']).dt.strftime('%Y-%m-%d').tolist()
        logging.info(f"Unique dates for this gameweek: {dates}")
        return dates
    except Exception as e:
        logging.error(f"Error getting gameweek dates: {e}")
        return []

def get_match_urls(session, date_str):
    """Get Premier League match URLs for a specific date"""
    # Convert date to required format (YYYY-MM-DD)
    url = f'https://www.skysports.com/football-scores-fixtures/{date_str}'
    
    logging.info(f"Getting dates for URL: {url}")

    try:
        response = session.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find Premier League section using new class structure
        tournaments = soup.find_all('div', class_='ui-tournament-matches__tournament')
        premier_league_section = None
        
        for tournament in tournaments:
            tournament_name = tournament.find('h3', class_='ui-tournament-matches__tournament-name')
            if tournament_name and 'Premier League' in tournament_name.text:
                premier_league_section = tournament
                break
        
        if not premier_league_section:
            logging.warning(f"No Premier League matches found for {date_str}")
            return []

        # Get match URLs using new class structure
        links = premier_league_section.find_all('a', class_='ui-sport-match-score__wrapper')
        match_urls = [link.get('href') for link in links]
        
        # Convert to team URLs
        team_urls = []
        for url in match_urls:
            if url:  # Check if URL exists
                parts = url.split('/')
                parts.insert(-1, 'teams')
                team_url = 'https://www.skysports.com' + '/'.join(parts)
                team_urls.append(team_url)

        logging.info(f"Found {len(team_urls)} Premier League matches for {date_str}")   
        return team_urls
        
        
    
    except Exception as e:
        logging.error(f"Error getting match URLs: {e}")
        return []
    

def extract_players(team_container):
    """Extract player names from team container"""
    players = []
    # Only get the first dl element (starting lineup) and skip the substitutes
    starting_dl = team_container.find('dl', class_='sdc-site-team-lineup__players')
    if starting_dl:
        for player_element in starting_dl.find_all('dd', class_='sdc-site-team-lineup__player-name'):
            try:
                try: 
                    first_name = player_element.find('span', class_='sdc-site-team-lineup__player-initial')['title']
                except:
                    first_name = ''
                surname = player_element.find('span', class_='sdc-site-team-lineup__player-surname').text.strip()
                full_name = f'{first_name} {surname}'.strip()
                if full_name:
                    players.append(full_name)
            except Exception as e:
                continue
    return players

def normalize_team_names(name):
    """Normalize team names to match FBRef format"""
    team_name_map = {
        'Luton': 'Luton Town',
        'Newcastle': 'Newcastle Utd',
        'Man Utd': 'Manchester Utd',
        'Man City': 'Manchester City',
        'Ipswich': 'Ipswich Town',
        'N Forest': "Nott'ham Forest",
        'Sheffield Utd': 'Sheffield United',
        'C Palace': 'Crystal Palace',
        'Leicester': 'Leicester City',
        'Spurs': 'Tottenham'

    }
    return team_name_map.get(name, name)

def main():
    parser = argparse.ArgumentParser(description='Scrape Sky Sports lineup data')
    parser.add_argument('--date', required=True, help='Date in format YYYY-MM-DD')
    parser.add_argument('--season', required=True, type=int, help='Season year')
    parser.add_argument('--gameweek', required=True, type=int, help='Premier League gameweek number')
    
    args = parser.parse_args()
    logging.info(f"Starting scraper with date={args.date}, season={args.season}, gameweek={args.gameweek}")
    
    # Create data directory if needed
    if not os.path.exists('Data And Scripts/Data'):
        os.makedirs('Data And Scripts/Data')
    
    session = create_session()
    
    # Get dates for the specific gameweek and season
    gameweek_dates = get_gameweek_dates(args.gameweek, args.season)
    
    if len(gameweek_dates) == 0:
        gameweek_dates = [args.date]
        logging.info(f"No dates found in fixtures, using provided date: {args.date}")
    
    all_lineups = []
    processed_dates = set()
    
    for date in gameweek_dates:
        if date in processed_dates:
            logging.info(f"Skipping already processed date: {date}")
            continue
            
        processed_dates.add(date)
        logging.info(f"Processing matches for {date}")
        
        urls = get_match_urls(session, date)
        if not urls:
            logging.info(f"No match URLs found for {date}")
            continue
        
        logging.info(f"Found {len(urls)} match URLs for {date}: {urls}")
            
        for url in urls:
            try:
                response = session.get(url)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find team sections using new class structure
                team_sections = soup.find_all('div', class_='sdc-site-team-lineup__col')
                
                if len(team_sections) >= 2:
                    home_section = team_sections[0]
                    away_section = team_sections[1]
                    
                    # Get team names
                    home_team = home_section.find('h4', class_='sdc-site-team-lineup__team-name').text.strip()
                    away_team = away_section.find('h4', class_='sdc-site-team-lineup__team-name').text.strip()
                    
                    # Normalize team names
                    home_team = normalize_team_names(home_team)
                    away_team = normalize_team_names(away_team)
                    
                    # Get lineups
                    home_players = extract_players(home_section)
                    away_players = extract_players(away_section)
                    
                    if home_players and away_players:
                        for team, players in [(home_team, home_players), (away_team, away_players)]:
                            for player in players:
                                all_lineups.append({
                                    'Player': player,
                                    'team': team,
                                    'starting_11': 'Y',
                                    'game': f"{home_team} vs {away_team}",
                                    'Season': args.season,
                                    'Wk': args.gameweek
                                })
                
                time.sleep(2)
                
            except Exception as e:
                logging.error(f"Error processing {url}: {e}")
                continue
    
    if all_lineups:
        df = pd.DataFrame(all_lineups)
        output_file = f'Data And Scripts/Data/sky_lineups_gw{args.gameweek}.csv'
        df.to_csv(output_file, index=False)
        logging.info(f"Saved {len(all_lineups)} lineup entries to {output_file}")
    else:
        logging.warning("No lineup data collected")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Script failed: {e}")
        raise