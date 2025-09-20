import streamlit as st
import pandas as pd
import numpy as np
import requests 
import json
import ast
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
import plotly.express as px
import plotly.graph_objects as go
from GameReportsCourt import CourtCoordinates
import math
from numpy.random import randint

st.set_page_config(layout="wide", page_title="CBB Shot Predictor", page_icon="🏀")


def calculate_distance(x1, y1, x2, y2):
    """Calculate the distance between two points (x1, y1) and (x2, y2)."""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def generate_arc_points(p1, p2, apex, num_points=100):
    """Generate points on a quadratic Bezier curve (arc) between p1 and p2 with an apex."""
    t = np.linspace(0, 1, num_points)
    x = (1 - t)**2 * p1[0] + 2 * (1 - t) * t * apex[0] + t**2 * p2[0]
    y = (1 - t)**2 * p1[1] + 2 * (1 - t) * t * apex[1] + t**2 * p2[1]
    z = (1 - t)**2 * p1[2] + 2 * (1 - t) * t * apex[2] + t**2 * p2[2]
    return x, y, z

def create_nba_full_court(ax=None, court_color='#dfbb85',
                          lw=3, lines_color='black', lines_alpha=0.8,
                          paint_fill='', paint_alpha=0.8,
                          inner_arc=False):
    if ax is None:
        ax = plt.gca()

    # Create Pathes for Court Lines
    center_circle = Circle((94/2, 50/2), 6,
                           linewidth=lw, color=lines_color, lw=lw,
                           fill=False, alpha=lines_alpha)
    hoop_left = Circle((5.25, 50/2), 1.5 / 2,
                       linewidth=lw, color=lines_color, lw=lw,
                       fill=False, alpha=lines_alpha)
    hoop_right = Circle((94-5.25, 50/2), 1.5 / 2,
                        linewidth=lw, color=lines_color, lw=lw,
                        fill=False, alpha=lines_alpha)

    # Paint: 16 feet wide, 19 feet from baseline
    left_paint = Rectangle((0, 50/2 - 8), 19, 16,
                           fill=paint_fill, alpha=paint_alpha,
                           lw=lw, edgecolor=None)
    right_paint = Rectangle((94-19, 50/2 - 8), 19, 16,
                            fill=paint_fill, alpha=paint_alpha,
                            lw=lw, edgecolor=None)

    left_paint_border = Rectangle((0, 50/2 - 8), 19, 16,
                                  fill=False, alpha=lines_alpha,
                                  lw=lw, edgecolor=lines_color)
    right_paint_border = Rectangle((94-19, 50/2 - 8), 19, 16,
                                   fill=False, alpha=lines_alpha,
                                   lw=lw, edgecolor=lines_color)

    # Arcs at top of paint (6ft radius)
    left_arc = Arc((19, 50/2), 12, 12, theta1=-90, theta2=90,
                   color=lines_color, lw=lw, alpha=lines_alpha)
    right_arc = Arc((94-19, 50/2), 12, 12, theta1=90, theta2=-90,
                    color=lines_color, lw=lw, alpha=lines_alpha)

    # Lane markers (using standard spacing — could be customized more)
    # You can replicate lane markers symmetrically like before

    # Three-point lines (23.75 ft from basket, 22 ft in corners)
    corner_y = 3  # 22 ft from hoop to corner along baseline
    arc_radius = 23.75
    arc_diameter = arc_radius * 2
    three_pt_left = Arc((5.25, 50/2), arc_diameter, arc_diameter,
                        theta1=-69, theta2=69,
                        color=lines_color, lw=lw, alpha=lines_alpha)
    three_pt_right = Arc((94-5.25, 50/2), arc_diameter, arc_diameter,
                         theta1=180-69, theta2=180+69,
                         color=lines_color, lw=lw, alpha=lines_alpha)

    # 22-foot corner line (~14 feet from baseline to break point)
    ax.plot((0, 14), (corner_y, corner_y), color=lines_color, lw=lw, alpha=lines_alpha)
    ax.plot((0, 14), (50-corner_y, 50-corner_y), color=lines_color, lw=lw, alpha=lines_alpha)
    ax.plot((94-14, 94), (corner_y, corner_y), color=lines_color, lw=lw, alpha=lines_alpha)
    ax.plot((94-14, 94), (50-corner_y, 50-corner_y), color=lines_color, lw=lw, alpha=lines_alpha)

    # Add Patches
    ax.add_patch(left_paint)
    ax.add_patch(right_paint)
    ax.add_patch(left_paint_border)
    ax.add_patch(right_paint_border)
    ax.add_patch(center_circle)
    ax.add_patch(hoop_left)
    ax.add_patch(hoop_right)
    ax.add_patch(left_arc)
    ax.add_patch(right_arc)
    ax.add_patch(three_pt_left)
    ax.add_patch(three_pt_right)

    if inner_arc:
        left_inner_arc = Arc((19, 50/2), 12, 12, theta1=90, theta2=-90,
                             color=lines_color, lw=lw, alpha=lines_alpha, ls='--')
        right_inner_arc = Arc((94-19, 50/2), 12, 12, theta1=-90, theta2=90,
                              color=lines_color, lw=lw, alpha=lines_alpha, ls='--')
        ax.add_patch(left_inner_arc)
        ax.add_patch(right_inner_arc)

    # Restricted area (4 ft radius from center of basket)
    restricted_left = Arc((5.25, 50/2), 8, 8, theta1=-90, theta2=90,
                          color=lines_color, lw=lw, alpha=lines_alpha)
    restricted_right = Arc((94-5.25, 50/2), 8, 8, theta1=90, theta2=-90,
                           color=lines_color, lw=lw, alpha=lines_alpha)
    ax.add_patch(restricted_left)
    ax.add_patch(restricted_right)

    # Backboards
    ax.plot((4, 4), ((50/2) - 3, (50/2) + 3), color=lines_color, lw=lw*1.5, alpha=lines_alpha)
    ax.plot((94-4, 94-4), ((50/2) - 3, (50/2) + 3), color=lines_color, lw=lw*1.5, alpha=lines_alpha)
    ax.plot((4, 4.6), (50/2, 50/2), color=lines_color, lw=lw, alpha=lines_alpha)
    ax.plot((94-4, 94-4.6), (50/2, 50/2), color=lines_color, lw=lw, alpha=lines_alpha)

    # Halfcourt line
    ax.axvline(94/2, color=lines_color, lw=lw, alpha=lines_alpha)

    # Court border
    border = Rectangle((0.3, 0.3), 94 - 0.4, 50 - 0.4, fill=False, lw=3,
                       color='black', alpha=lines_alpha)
    ax.add_patch(border)

    # Plot Limit
    ax.set_xlim(0, 94)
    ax.set_ylim(0, 50)
    ax.set_facecolor(court_color)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    return ax


def pts_value(row):
    if (row['made'] == 1):
        if (row['action_type'] == '2pt'):
            return 2
        else:
            return 3
    else:
        return 0

def safe_literal_eval(val):
    if isinstance(val, str) and val.strip().startswith('['):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return None
    return val  

def make_get_request(url, token, params=None):
    headers = {
        "Authorization": f"Bearer {token}",
    }
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API error: {e}")
        return None

def fetch_all_player_locations(game_id):
    base_url = "https://api.shotquality.com"
    endpoint = "/player-locations"
    all_rows = []
    offset = 0
    limit = 1000

    while True:
        url = f"{base_url}{endpoint}/{game_id}"
        params = {"limit": limit, "offset": offset}
        response = make_get_request(url, token, params)
        
        if not response or 'player_locations' not in response or not response['player_locations']:
            break

        all_rows.extend(response['player_locations'])
        offset += limit

    return pd.DataFrame(all_rows)

@st.cache_data
def load_competitions():
    competitions_url = 'https://api.cbbanalytics.com/api/gs/competitions'
    response = make_get_request(competitions_url, token)
    if response :
        return pd.DataFrame(response)
    return pd.DataFrame()

@st.cache_data
def load_all_players(compid):
    players_url = f'https://api.cbbanalytics.com/api/gs/player-agg-stats-public?competitionId={compid}&divisionId=1&scope=season'
    response = make_get_request(players_url, token)
    if response :
        return pd.DataFrame(response)
    return pd.DataFrame()

@st.cache_data
def load_and_process_data(game_id, token):
    base_url = "https://api.shotquality.com"
    endpoints = {
        "Play By Play": "/play-by-play",
        "Player Stats": "/player-stats",
        "Players": "/players"
    }

    # Play-by-play data
    pbp_url = f"{base_url}{endpoints['Play By Play']}/{game_id}"
    pbp_params = {"limit": 1000, "feature_store": True}
    pbp_response = make_get_request(pbp_url, token, pbp_params)
    pbp_df = pd.DataFrame(pbp_response['plays'])

    # Merge assist & shot rows
    def merge_same_play_rows(group):
        shot_row = group[group['action_type'].isin(['2pt', '3pt'])]
        base = shot_row.iloc[0].copy() if not shot_row.empty else group.iloc[0].copy()
        secondary = group[group['play_id'] != base['play_id']]
        if not secondary.empty:
            second = secondary.iloc[0]
            base['second_action'] = second['action_type']
            base['second_player_id'] = second['player_id']
        else:
            base['second_action'] = None
            base['second_player_id'] = None
        return base

    pbp_df = pbp_df.groupby('parent_play_id').apply(merge_same_play_rows).reset_index(drop=True)

    # Fetch all player locations
    locdf = fetch_all_player_locations(game_id, token)

    # Merge play-by-play with location data
    fulldf = pd.merge(pbp_df, locdf, on='play_id',how='left')
    fulldf = fulldf.rename(columns={
        'player_id_x': 'shooter_id',
        'player_id_y': 'court_player_id'
    })

    # Player stats to get list of player_ids
    stats_url = f"{base_url}/player-stats/{game_id}"
    stats_response = make_get_request(stats_url, token, {"limit": 100})
    stats_df = pd.DataFrame(stats_response['player_stats'])

    allplayers = pd.DataFrame()
    ids = stats_df['player_id'].dropna().unique()

    for id in ids:
        player_url = f"{base_url}/players"
        player_response = make_get_request(player_url, token, {"player_id": id, "limit": 5})
        if player_response and 'players' in player_response and player_response['players']:
            player_df = pd.DataFrame(player_response['players'])
            allplayers = pd.concat([allplayers, player_df], ignore_index=True)

    # ID conversion to Int64 to support nulls
    fulldf['shooter_id'] = pd.to_numeric(fulldf['shooter_id'], errors='coerce').astype('Int64')
    fulldf['court_player_id'] = pd.to_numeric(fulldf['court_player_id'], errors='coerce').astype('Int64')
    fulldf['second_player_id'] = pd.to_numeric(fulldf['second_player_id'], errors='coerce').astype('Int64')
    allplayers['player_id'] = pd.to_numeric(allplayers['player_id'], errors='coerce').astype('Int64')

    # Map names
    id_to_name = allplayers.set_index('player_id')['player_name']
    id_to_height = allplayers.set_index('player_id')['height_inches']
    id_to_weight = allplayers.set_index('player_id')['weight_pounds']

    fulldf['shooter_name'] = fulldf['shooter_id'].map(id_to_name)
    fulldf['court_player_name'] = fulldf['court_player_id'].map(id_to_name)
    fulldf['secondary_name'] = fulldf['second_player_id'].map(id_to_name)
    fulldf['shooter_height'] = fulldf['shooter_id'].map(id_to_height)
    fulldf['shooter_weight'] = fulldf['shooter_id'].map(id_to_weight)


    return fulldf

# API base URL
base_url = "https://api.shotquality.com"

# Your API token
token = st.secrets["api"]["key"]

# Example endpoints

endpoints = {  
    "Games": "/games",  
    "Competition Seasons": "/competition-seasons",  
    "Teams": "/teams",  
    "Players": "/players",  
    "Play By Play": "/play-by-play",  
    "Player Stats": "/player-stats",  
    "Player Locations": "/player-locations",  
}
@st.cache_data
def load_sq_comps():
    url = f'https://api.shotquality.com/competition-seasons/?competition_name=NCAAM&limit=100'
    response = make_get_request(url, token)
    if response and 'competition_seasons' in response:
        return pd.DataFrame(response['competition_seasons'])
    return pd.DataFrame()

@st.cache_data
def load_player_data(player_name):
    url = f'https://api.shotquality.com/players/?player_name={player_name}'
    response = make_get_request(url, token)
    if response and 'players' in response:
        return pd.DataFrame(response['players'])
    return pd.DataFrame()

@st.cache_data
def fetch_games(compid):
    games_url = f"{base_url}{endpoints['Games']}/{compid}"
    games_params = {
        "limit": 10000
    }

    teamurl = 'https://api.shotquality.com/teams/'
    team_param = {  
        "limit": 10000
    } 
    team_response = make_get_request(teamurl, token, params=team_param) 
    teamlistdf = pd.DataFrame(team_response['teams'])
    teamlistdf = teamlistdf[teamlistdf['gender'] == 'M']
    team_name_dict = dict(zip(teamlistdf['team_id'], teamlistdf['team_name']))

    games_response = make_get_request(games_url, token, params=games_params)
    games = pd.DataFrame(games_response['games'])
    games = games[games['game_descriptors'].apply(lambda d: d.get('event_type_id') != 'Preseason')]
    games['home_team_name'] = games['home_team_id'].map(team_name_dict)
    games['away_team_name'] = games['away_team_id'].map(team_name_dict)
    games['game_datetime_utc'] = pd.to_datetime(games['game_datetime_utc'])
    games['game_datetime_utc'] = games['game_datetime_utc'].dt.strftime('%m-%d-%Y')
    return games

@st.cache_data
def load_game_with_freethrows(game):
    url = f"https://api.shotquality.com/play-by-play/{game}?feature_store=true"
    headers = {
        "accept": "application/json",
        "authorization": "Bearer 65e2cf34-9cef-4cec-990d-c16f62ed3de7"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # also good to catch HTTP errors

        data = response.json()
        pbp = pd.DataFrame(data['plays'])
    except requests.RequestException as e:
        st.error(f"Error fetching play-by-play data: {e}")
        return pd.DataFrame()
    return pbp
sqcomps = load_sq_comps()
comps = load_competitions()
comps = comps[(comps['startYear'] >= 2023) & (comps['gender'] == 'MALE')]
complist = comps['competitionName'].tolist()
selected_competition = st.sidebar.selectbox("Select Competition", complist)
comps = comps[comps['competitionName'] == selected_competition]
startYear = comps['startYear'].iloc[0]
sqcomps = sqcomps[sqcomps['season_start_year'] == startYear]
sqcompid = sqcomps['competition_season_id'].iloc[0]
compid = comps['competitionId'].iloc[0]
st.markdown("<h1 style='text-align: center; text-color: #000000; text-shadow: 2px 2px 5px #ffffff; font-size: 5em;'>CBB Shot Predictor</h1>", unsafe_allow_html=True)
allplayers = load_all_players(compid)
allplayers['teamFullName'] = allplayers['teamMarket'] + " " + allplayers['teamName']
playerlist = [f"{row['fullName']} | {row['teamFullName']}" for _, row in allplayers.iterrows()]
selected_player = st.selectbox("Select a player", playerlist)
team = selected_player.split(" | ")[1]

playerdf = load_player_data(selected_player.split(" | ")[0])

games = fetch_games(sqcompid)
games = games[(games['home_team_name'] == team.split(' ')[0]) | (games['away_team_name'] == team.split(' ')[0])]
gamelist = []

playersurl = f'https://api.cbbanalytics.com/api/gs/player-agg-stats-public?competitionId={compid}&divisionId=1&scope=season'
playersdata = requests.get(playersurl).json()
playersdf = pd.DataFrame(playersdata)
playerId = playersdf[playersdf['fullName'] == selected_player.split(" | ")[0]]['playerId'].iloc[0]
teamId = playersdf[playersdf['fullName'] == selected_player.split(" | ")[0]]['teamId'].iloc[0]
filteredplayersdf = playersdf[playersdf['fullName'] == selected_player.split(" | ")[0]]
playergamesurl = f'https://api.cbbanalytics.com/api/gs/player-game-stats?competitionId={compid}&playerId={playerId}&pass=false'
playergamesdata = requests.get(playergamesurl).json()
playergamesdf = pd.DataFrame(playergamesdata)
gamesplayed = playergamesdf['teamMarketAgst'].unique().tolist()
playergamesdf['gameDate'] = pd.to_datetime(playergamesdf['gameDate']).dt.strftime('%m-%d-%Y')
games = games[games['away_team_name'].isin(gamesplayed) | games['home_team_name'].isin(gamesplayed)]
for index, row in games.iterrows():
    gamelist.append(f"{row['home_team_name']} vs {row['away_team_name']} | {row['game_datetime_utc']} (ID: {row['game_id']})")


selected_games = st.sidebar.multiselect(f"Select games ({len(gamelist)})", gamelist, default=gamelist)
if selected_games:
    st.markdown(f"""
    <div style='text-align:center; border-radius:10px;'>
        <img src='https://storage.googleapis.com/cbb-image-files/player-headshots/{teamId}-{playerId}.png' width='180'/>
    </div>
    """, unsafe_allow_html=True)
    st.subheader(f'Year: {filteredplayersdf["classYr"].iloc[0]} | Position: {playerdf["position"].iloc[0]} | Height: {int(filteredplayersdf["height"].iloc[0]/12)}\'{int(filteredplayersdf["height"].iloc[0]%12)} | Weight: {playerdf["weight_pounds"].iloc[0]} lbs')
    allpbp = pd.DataFrame()
    offpbp = pd.DataFrame()
    defpbp = pd.DataFrame()
    ftpbp = pd.DataFrame()
    for game in selected_games:
        game_id = int(game.split("ID: ")[1].strip(")"))
        gamedate = game.split(" | ")[1].split(" (ID")[0]
        hometeam = game.split(" | ")[0].split(" vs ")[0]
        awayteam = game.split(" | ")[0].split(" vs ")[1]
        gamename = f"{hometeam} vs {awayteam}"
        gamedate = pd.to_datetime(gamedate).strftime('%m-%d-%Y')

        tempplayergamesdf = playergamesdf[playergamesdf['gameDate'] < gamedate]
        baseline_pct = 0.70
        baseline_attempts = 10
        total_made = baseline_pct * baseline_attempts
        total_attempts = baseline_attempts

        running_pct = []

        for _, row in tempplayergamesdf.iterrows():
            total_made += row['ftm']
            total_attempts += row['fta']
            running_pct.append(total_made / total_attempts)

        tempplayergamesdf['running_ft_pct'] = running_pct
        if not tempplayergamesdf.empty:
            ftpct = tempplayergamesdf['running_ft_pct'].iloc[-1]
        else:
            ftpct = baseline_pct


        pbp = load_game_with_freethrows(game_id)
        feature_df = pd.DataFrame([dict(f) if isinstance(f, dict) else {} for f in pbp['feature_store']])
        pbp = pd.concat([pbp.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)
        pbp_ft = pbp.copy()
        pbp_ft = pbp_ft[pbp_ft['action_type'] == 'freethrow']
        pbp_ft = pbp_ft[pbp_ft['player_id'] == playerdf['player_id'].iloc[0]]
        pbp['pts'] = np.where(pbp['action_type'] == '2pt',2,3)
        pbp['home_team'] = hometeam
        pbp['away_team'] = awayteam
        pbp['game_date'] = gamedate
        pbp['game_name'] = gamename
        pbp['ftpct'] = ftpct
        offdf = pbp[pbp['player_id'] == playerdf['player_id'].iloc[0]]
        offdf = offdf[offdf['action_type'].isin(['2pt','3pt'])]
        if 'Closest Defender ID'  in pbp.columns:
            defdf = pbp[pbp['Closest Defender ID'] == playerdf['player_id'].iloc[0]]
            defpbp = pd.concat([defpbp, defdf], ignore_index=True)
            defpbp['defender_name'] = playerdf['player_name'].iloc[0]
            defpbp['shooter_height'] = playerdf['height_inches'].iloc[0]
            defpbp['shooter_weight'] = playerdf['weight_pounds'].iloc[0]
        allpbp = pd.concat([allpbp, pbp], ignore_index=True)
        offpbp = pd.concat([offpbp, offdf], ignore_index=True)
        ftpbp = pd.concat([ftpbp, pbp_ft], ignore_index=True)
        ftpbp['player_name'] = playerdf['player_name'].iloc[0]
        offpbp['player_name'] = playerdf['player_name'].iloc[0]

    features = ['minutes',
    'seconds',
    'spacing',
    'Defenders in Paint',
    'Shooter in the Paint',
    'Closest Defender Behind',
    'Closest Defender Inside',
    'Closest Defender Blocking',
    'Shooter in Restricted Area',
    'Second-Closest Defender Behind',
    'Second-Closest Defender Inside',
    'Harmonic Mean Defender Distance',
    'Average Defender Distance (Feet)',
    'Defenders in the Restricted Area',
    'Second-Closest Defender Blocking',
    'Number of Defenders Closer to the Basket',
    'Number of Teammates in the Restricted Area',
    'Cosine of the Angle with the Closest Defender',
    'Cosine of the Angle with the Second-Closest Defender',
    "Height of the Closest Defender's Bounding Box (Pixels)",
    "Height of the Third-Closest Defender's Bounding Box (Pixels)",
    "Height of the Second-Closest Defender's Bounding Box (Pixels)",
    'Shot Distance (ft.)',
    'Distance to the Closest Defender (Feet)',
    'Distance to the Second-Closest Defender (Feet)',
    'pts',
    'alley_oop',
    'catch_and_shoot',
    'dunk',
    'floater',
    'in_paint',
    'jumpshot',
    'lay_up',
    'off_cut',
    'off_drive',
    'pick_and_roll',
    'post_up',
    'reverse',
    'step_back',
    'tip_in',
    'transition',
    'hook',
    'pull_up',
    'turnaround']

    model = pickle.load(open('xFG_model.pkl', 'rb'))

    descriptor_features = ['alley_oop',
    'catch_and_shoot',
    'dunk',
    'floater',
    'hook',
    'in_paint',
    'jumpshot',
    'lay_up',
    'off_cut',
    'off_drive',
    'pick_and_roll',
    'post_up',
    'pull_up',
    'reverse',
    'step_back',
    'tip_in',
    'transition',
    'turnaround']

    offpbp['play_descriptors_parsed'] = offpbp['play_descriptors'].apply(safe_literal_eval)
    defpbp['play_descriptors_parsed'] = defpbp['play_descriptors'].apply(safe_literal_eval)

    for desc in descriptor_features:
        offpbp[desc] = offpbp['play_descriptors_parsed'].apply(lambda x: int(desc in x))
        defpbp[desc] = defpbp['play_descriptors_parsed'].apply(lambda x: int(desc in x))

    input = offpbp[features]
    input2 = defpbp[features]
    input['spacing'] = input['spacing'].astype(float)
    offpbp['pts_value'] = offpbp.apply(pts_value, axis=1)
    input2['spacing'] = input2['spacing'].astype(float)
    defpbp['pts_value'] = defpbp.apply(pts_value, axis=1)


    preds = model.predict_proba(input)[:,1]
    preds2 = model.predict_proba(input2)[:,1]

    offpbp['xFG%'] = preds
    offpbp['xpts'] = offpbp['xFG%'] * offpbp['pts']
    defpbp['xFG%'] = preds2
    defpbp['xpts'] = defpbp['xFG%'] * defpbp['pts']

    player_xfg = pd.DataFrame(offpbp[offpbp['xFG%'].notna()].groupby('player_name')['xFG%'].mean())
    player_xfg['FG%'] = offpbp.groupby('player_name')['made'].mean()
    player_xfg['xFG%_OE'] = player_xfg['FG%'] - player_xfg['xFG%']
    player_xfg['shots'] = offpbp.groupby('player_name')['made'].count()

    player_xfg['pps'] = offpbp.groupby('player_name')['pts_value'].sum() / player_xfg['shots']
    player_xfg['x_pps'] = offpbp.groupby('player_name')['xpts'].mean()

    fg_points = offpbp.groupby('player_name')['pts_value'].sum()
    fts = ftpbp.groupby('player_name')['made'].count()
    player_xfg['ft_attempts'] = fts
    ft_points = ftpbp.groupby('player_name')['made'].sum()  # FT points are just made FTs (1 pt each)
    player_xfg['total_points'] = fg_points.add(ft_points, fill_value=0)
    player_xfg['fg_points'] = fg_points
    player_xfg['ft_points'] = ft_points
    x_fg_points = offpbp.groupby('player_name')['xpts'].sum()
    player_xfg['x_fg_points'] = x_fg_points
    ft_pct_by_player = offpbp.groupby('player_name')['ftpct'].mean()
    ft_attempts = ftpbp.groupby('player_name')['made'].count()
    ft_expected_points = ft_attempts * ft_pct_by_player
    player_xfg['x_ft_points'] = ft_expected_points
    x_shot_points = x_fg_points.add(ft_expected_points, fill_value=0)

    player_xfg['x_total_points'] = x_shot_points
    player_xfg['avg_shot_dist'] = offpbp[offpbp['xFG%'].notna()].groupby('player_name')['Shot Distance (ft.)'].mean()
    player_xfg['avg_def_height'] = offpbp[offpbp['xFG%'].notna()].groupby('player_name')['Height of the Closest Defender (Inches)'].mean()
    player_xfg['avg_def_weight'] = offpbp[offpbp['xFG%'].notna()].groupby('player_name')['Weight of the Closest Defender (Pounds)'].mean()
    player_xfg['avg_def_dist'] = offpbp[offpbp['xFG%'].notna()].groupby('player_name')['Distance to the Closest Defender (Feet)'].mean()


    def_xfg = pd.DataFrame(defpbp[defpbp['xFG%'].notna()].groupby('defender_name')['xFG%'].mean())
    def_xfg['FG%'] = defpbp[defpbp['xFG%'].notna()].groupby('defender_name')['made'].mean()
    def_xfg['xFG%_OE'] = def_xfg['FG%'] - def_xfg['xFG%']
    def_xfg['count'] = defpbp[defpbp['xFG%'].notna()].groupby('defender_name')['made'].count()
    def_xfg['pps'] = defpbp[(defpbp['xFG%'].notna())].groupby('defender_name')['pts_value'].sum()/def_xfg['count']
    def_xfg['x_pps'] = defpbp[defpbp['xFG%'].notna()].groupby('defender_name')['xpts'].mean()
    def_xfg['shot_points'] = defpbp[(defpbp['xFG%'].notna())].groupby('defender_name')['pts_value'].sum()
    def_xfg['x_shot_points'] = defpbp[(defpbp['xFG%'].notna())].groupby('defender_name')['xpts'].sum()
    def_xfg['avg_shot_dist'] = defpbp[defpbp['xFG%'].notna()].groupby('defender_name')['Shot Distance (ft.)'].mean()
    def_xfg['avg_off_height'] = defpbp[defpbp['xFG%'].notna()].groupby('defender_name')['shooter_height'].mean()
    def_xfg['avg_off_weight'] = defpbp[defpbp['xFG%'].notna()].groupby('defender_name')['shooter_weight'].mean()
    def_xfg['avg_def_dist'] = defpbp[defpbp['xFG%'].notna()].groupby('defender_name')['Distance to the Closest Defender (Feet)'].mean()


    # st.dataframe(player_xfg[player_xfg['shots'] > 5].sort_values(by='xFG%_OE', ascending=False))

    # st.dataframe(def_xfg[def_xfg['count'] > 5].sort_values(by='xFG%_OE', ascending=False))



    if not player_xfg.empty or not def_xfg.empty:
        st.markdown('---')
        colA, colB = st.columns(2)
        if not player_xfg.empty:
            pxfg = player_xfg[player_xfg['shots'] > 5].sort_values(by='xFG%_OE', ascending=False).reset_index()
            pxfg = pxfg.iloc[0] if not pxfg.empty else None
            with colA:
                st.markdown('### Offensive Shot Summary')
                if pxfg is not None:
                    st.markdown(f"""
                    <div style='font-size:18px; margin-bottom:8px;'><b>FG%:</b> {pxfg['FG%']:.4f}</div>
                    <div style='font-size:18px; margin-bottom:8px;'><b>xFG%:</b> {pxfg['xFG%']:.4f}</div>
                    <div style='font-size:18px; margin-bottom:8px;'><b>xFG% OE:</b> {pxfg['xFG%_OE']:.4f}</div>
                    <div style='font-size:18px; margin-bottom:8px;'><b>Shots:</b> {int(pxfg['shots'])}</div>
                    <div style='font-size:18px; margin-bottom:8px;'><b>PPS:</b> {pxfg['pps']:.2f}</div>
                    <div style='font-size:18px; margin-bottom:8px;'><b>xPPS:</b> {pxfg['x_pps']:.2f}</div>
                    <div style='font-size:18px; margin-bottom:8px;'><b>Points Scored:</b> {int(pxfg['total_points'])}</div>
                    <div style='font-size:18px; margin-bottom:8px;'><b>xPoints Scored:</b> {pxfg['x_total_points']:.2f}</div>
                    <div style='font-size:18px; margin-bottom:8px;'><b>Avg Shot Distance (ft):</b> {pxfg['avg_shot_dist']:.2f}</div>
                    <div style='font-size:18px; margin-bottom:8px;'><b>Avg Distance from Defender (ft):</b> {pxfg['avg_def_dist']:.2f}</div>
                    <div style='font-size:18px; margin-bottom:8px;'><b>Avg Defender Height (in):</b> {pxfg['avg_def_height']:.2f}</div>
                    <div style='font-size:18px; margin-bottom:8px;'><b>Avg Defender Weight (lbs):</b> {pxfg['avg_def_weight']:.2f}</div>

                    """, unsafe_allow_html=True)
        if not def_xfg.empty:
            dxfg = def_xfg[def_xfg['count'] > 5].sort_values(by='xFG%_OE', ascending=False).reset_index()
            dxfg = dxfg.iloc[0] if not dxfg.empty else None
            with colB:
                st.markdown('### Defensive Shot Summary')
                if dxfg is not None:
                    st.markdown(f"""
                    <div style='font-size:18px; margin-bottom:8px;'><b>FG%:</b> {dxfg['FG%']:.4f}</div>
                    <div style='font-size:18px; margin-bottom:8px;'><b>xFG%:</b> {dxfg['xFG%']:.4f}</div>
                    <div style='font-size:18px; margin-bottom:8px;'><b>xFG% OE:</b> {dxfg['xFG%_OE']:.4f}</div>
                    <div style='font-size:18px; margin-bottom:8px;'><b>Shots Defended:</b> {int(dxfg['count'])}</div>
                    <div style='font-size:18px; margin-bottom:8px;'><b>PPS:</b> {dxfg['pps']:.2f}</div>
                    <div style='font-size:18px; margin-bottom:8px;'><b>xPPS:</b> {dxfg['x_pps']:.2f}</div>
                    <div style='font-size:18px; margin-bottom:8px;'><b>Points Allowed:</b> {int(dxfg['shot_points'])}</div>
                    <div style='font-size:18px; margin-bottom:8px;'><b>xPoints Allowed:</b> {dxfg['x_shot_points']:.2f}</div>
                    <div style='font-size:18px; margin-bottom:8px;'><b>Avg Shot Distance (ft):</b> {dxfg['avg_shot_dist']:.2f}</div>
                    <div style='font-size:18px; margin-bottom:8px;'><b>Avg Distance from Shooter (ft):</b> {dxfg['avg_def_dist']:.2f}</div>
                    <div style='font-size:18px; margin-bottom:8px;'><b>Avg Shooter Height (in):</b> {dxfg['avg_off_height']:.2f}</div>
                    <div style='font-size:18px; margin-bottom:8px;'><b>Avg Shooter Weight (lbs):</b> {dxfg['avg_off_weight']:.2f}</div>
                    """, unsafe_allow_html=True)
        st.markdown('---')

    xFG = st.checkbox('xFG%',value=True)
    sameSide = st.checkbox('Same Side of Court')
    if sameSide:
        offpbp.loc[offpbp['shot_x'] < 47, 'shot_x'] = 94 - offpbp['shot_x']
    if xFG:
        courtbg = 'black'
        courtlines = 'white'
    else:
        courtbg = '#d2a679'
        courtlines = 'black'
    import plotly.express as px
    court = CourtCoordinates()
    court_lines_df = court.get_court_lines()

    fig = px.line_3d(
        data_frame=court_lines_df,
        x='x',
        y='y',
        z='z',
        line_group='line_group',
        color='color',
        color_discrete_map={
            'court': courtlines,
            'hoop': '#e47041',
            'net': '#D3D3D3',
            'backboard': 'gray'
        }
    )
    fig.update_traces(hovertemplate=None, hoverinfo='skip', showlegend=False)

    fig.update_traces(line=dict(width=5))
    fig.update_layout(    
        margin=dict(l=20, r=20, t=20, b=20),
        scene_aspectmode="data",
        height=600,
        scene_camera=dict(
            eye=dict(x=1.3, y=0, z=0.7)
        ),
        scene=dict(
            xaxis=dict(title='', showticklabels=False, showgrid=False),
            yaxis=dict(title='', showticklabels=False, showgrid=False),
            zaxis=dict(title='',  showticklabels=False, showgrid=False, showbackground=True, backgroundcolor=courtbg),
        ),
        showlegend=False,
        legend=dict(
            yanchor='top',
            y=0.05,
            x=0.2,
            xanchor='left',
            orientation='h',
            font=dict(size=15, color='gray'),
            bgcolor='rgba(0, 0, 0, 0)',
            title='',
            itemsizing='constant'
        )
    )
    if xFG:
        fig.add_trace(go.Scatter3d(
            x=offpbp['shot_y'], y=offpbp['shot_x'], z=[0]*len(offpbp),
            mode='markers+text',
            marker=dict(size=7, color=offpbp['xFG%'], colorscale='Hot', cmin=0, cmax=1, opacity=0.8),
            hovertemplate=
            'Shooter: %{customdata[4]}<br>' +
            'Shot Distance: %{customdata[0]:.1f} ft<br>' +
            'Defender Distance: %{customdata[1]:.1f} ft<br>' +
            'Shot Result: %{customdata[2]}<br>' +
            'Shot Type: %{customdata[3]}<br>' +
            'Shot Descriptors: %{customdata[5]}<br>' +
            'xFG%: %{customdata[7]:.2f}<br>' +
            'xPts: %{customdata[6]:.2f}<br>',

        customdata=np.stack((
            offpbp['Shot Distance (ft.)'],
            offpbp['Distance to the Closest Defender (Feet)'],
            np.where(offpbp['made'] == 1, 'Made', 'Missed'),
            offpbp['action_type'],
            offpbp['player_name'],
            offpbp['play_descriptors'],
            offpbp['xpts'],
            offpbp['xFG%']
        ), axis=-1)
        ))
    else:
        fig.add_trace(go.Scatter3d(
            x=offpbp['shot_y'], y=offpbp['shot_x'], z=[0]*len(offpbp),
            mode='markers+text',
            marker=dict(size=7, color=np.where(offpbp['made'] == 1, 'green', 'red'), opacity=0.8),
            hovertemplate=
            'Shooter: %{customdata[4]}<br>' +
            'Shot Distance: %{customdata[0]:.1f} ft<br>' +
            'Defender Distance: %{customdata[1]:.1f} ft<br>' +
            'Shot Result: %{customdata[2]}<br>' +
            'Shot Type: %{customdata[3]}<br>' +
            'Shot Descriptors: %{customdata[5]}<br>' +
            'xFG%: %{customdata[7]:.2f}<br>' +
            'xPts: %{customdata[6]:.2f}<br>',

        customdata=np.stack((
            offpbp['Shot Distance (ft.)'],
            offpbp['Distance to the Closest Defender (Feet)'],
            np.where(offpbp['made'] == 1, 'Made', 'Missed'),
            offpbp['action_type'],
            offpbp['player_name'],
            offpbp['play_descriptors'],
            offpbp['xpts'],
            offpbp['xFG%']
        ), axis=-1)
        ))

    x_values = []
    y_values = []
    z_values = []
    x_values2 = []
    y_values2 = []
    z_values2 = []
    offpbp_made = offpbp[offpbp['made'] == 1]
   
    for index, row in offpbp_made.iterrows():
        x_values.append(row['shot_y'])
        y_values.append(row['shot_x'])
        z_values.append(0)
        x_values2.append(court.hoop_loc_x)
        if row['shot_x'] <= 47:
            y_values2.append(court.hoop_loc_y)
        else:
            y_values2.append(court.court_length-court.hoop_loc_y)
        z_values2.append(court.hoop_loc_z)
    x_coords = x_values
    y_coords = y_values
    z_value = 0  # Fixed z value
    x_coords2 = x_values2
    y_coords2 = y_values2
    z_value2 = court.hoop_loc_z

    for i in range(len(offpbp_made)):
        x1 = x_coords[i]
        y1 = y_coords[i]
        x2 = x_coords2[i]
        y2 = y_coords2[i]
        p2 = np.array([x1, y1, z_value])
        p1 = np.array([x2, y2, z_value2])
        distance = calculate_distance(x1, y1, x2, y2)
        # Set arc height based on shot distance
        shot_dist = offpbp_made['Shot Distance (ft.)'].iloc[i]
        if shot_dist > 3:
            if shot_dist > 50:
                h = randint(255,305)/10
            elif shot_dist > 30:
                h = randint(230,280)/10
            elif shot_dist > 25:
                h = randint(180,230)/10
            elif shot_dist > 15:
                h = randint(180,230)/10
            else:
                h = randint(130,160)/10
            apex = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2), h])
            x, y, z = generate_arc_points(p1, p2, apex)
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(width=8, color='green'),
                opacity=0.5,
                hoverinfo='none',
            ))

    st.plotly_chart(fig, use_container_width=True)

    # Organize offensive plots in columns
    if not offpbp.empty:
        with st.expander('Offensive Plots'):
            col1, col2, col3 = st.columns(3)
            # 1. Line chart: xFG% and FG% by game date
            with col1:
                offpbp['game_date'] = pd.to_datetime(offpbp['game_date'])
                by_game = offpbp.groupby('game_date').agg({
                    'xFG%': 'mean',
                    'made': 'mean',
                    'pts_value': 'count'
                }).reset_index().sort_values('game_date')
                by_game['FG%'] = by_game['made']
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=by_game['game_date'], y=by_game['xFG%'], mode='lines+markers', name='xFG%', line=dict(color='orange')))
                fig1.add_trace(go.Scatter(x=by_game['game_date'], y=by_game['FG%'], mode='lines+markers', name='FG%', line=dict(color='green')))
                fig1.update_layout(title='xFG% and FG% by Game', xaxis_title='Game Date', yaxis_title='Percentage', legend_title='Metric')
                st.plotly_chart(fig1, use_container_width=True)

            with col3:
                offpbp['game_date'] = pd.to_datetime(offpbp['game_date'])
                by_game = offpbp.groupby('game_date').agg({
                    'xpts': 'sum',
                    'pts_value': 'sum',
                    'pts_value': 'count'
                }).reset_index().sort_values('game_date')
                # by_game['FG%'] = by_game['made']
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=by_game['game_date'], y=by_game['xpts'], mode='lines+markers', name='xpts', line=dict(color='orange')))
                fig1.add_trace(go.Scatter(x=by_game['game_date'], y=by_game['pts_value'], mode='lines+markers', name='pts_value', line=dict(color='green')))
                fig1.update_layout(title='Expected and Actual Field Goal PPG', xaxis_title='Game Date', yaxis_title='Points', legend_title='Metric')
                st.plotly_chart(fig1, use_container_width=True)
            # 2. Bar chart: Shot type frequency
            with col2:
                shot_types = offpbp['action_type'].value_counts().reset_index()
                shot_types.columns = ['Shot Type', 'Count']
                fig2 = px.bar(shot_types, x='Shot Type', y='Count', color='Count', color_continuous_scale='Blues', title='Shot Type Frequency')
                st.plotly_chart(fig2, use_container_width=True)
            col4, col5, col6 = st.columns(3)
            # 3. Scatter plot: Shot distance vs. xFG%, colored by make/miss
            with col4:
                fig3 = px.scatter(
                    offpbp, x='Shot Distance (ft.)', y='xFG%', color=offpbp['made'].map({True: 'Made', False: 'Missed'}),
                    hover_data=['game_date', 'action_type', 'play_descriptors'],
                    title='Shot Distance vs. xFG% (Colored by Result)',
                    color_discrete_map={'Made': 'green', 'Missed': 'red'}
                )
                st.plotly_chart(fig3, use_container_width=True)
            # 4. Histogram: Distribution of shot distances
            with col6:
                fig4 = px.histogram(offpbp, x='Shot Distance (ft.)', nbins=20, color_discrete_sequence=['#636EFA'])
                fig4.update_layout(title='Distribution of Shot Distances', xaxis_title='Shot Distance (ft.)', yaxis_title='Count')
                st.plotly_chart(fig4, use_container_width=True)
            # 5. Radar chart: Descriptor features profile
            with col5:
                desc_counts = {desc: offpbp[desc].sum() for desc in descriptor_features}
                radar_df = pd.DataFrame({'Descriptor': list(desc_counts.keys()), 'Count': list(desc_counts.values())})
                fig5 = go.Figure()
                fig5.add_trace(go.Scatterpolar(
                    r=radar_df['Count'],
                    theta=radar_df['Descriptor'],
                    fill='toself',
                    name='Descriptor Profile',
                    line=dict(color='green')
                ))
                fig5.update_layout(
                    polar=dict(radialaxis=dict(visible=True)),
                    title='Shot Descriptor Profile (Radar Chart)'
                )
                st.plotly_chart(fig5, use_container_width=True)

    # --- Defensive Plots ---
    if not defpbp.empty:
        with st.expander('Defensive Plots (as Closest Defender)'):
            dcol1, dcol2, dcol3 = st.columns(3)
            # 1. Line chart: Opponent xFG% and FG% by game date
            with dcol1:
                defpbp['game_date'] = pd.to_datetime(defpbp['game_date'])
                d_by_game = defpbp.groupby('game_date').agg({
                    'xFG%': 'mean',
                    'made': 'mean',
                    'pts_value': 'count'
                }).reset_index().sort_values('game_date')
                d_by_game['FG%'] = d_by_game['made']
                dfig1 = go.Figure()
                dfig1.add_trace(go.Scatter(x=d_by_game['game_date'], y=d_by_game['xFG%'], mode='lines+markers', name='xFG%', line=dict(color='orange')))
                dfig1.add_trace(go.Scatter(x=d_by_game['game_date'], y=d_by_game['FG%'], mode='lines+markers', name='FG%', line=dict(color='red')))
                dfig1.update_layout(title='Opponent xFG% and FG% by Game (Defender)', xaxis_title='Game Date', yaxis_title='Percentage', legend_title='Metric')
                st.plotly_chart(dfig1, use_container_width=True)
            with dcol3:
                defpbp['game_date'] = pd.to_datetime(defpbp['game_date'])
                d_by_game = defpbp.groupby('game_date').agg({
                    'xpts': 'sum',
                    'pts_value': 'sum',
                    'pts_value': 'count'
                }).reset_index().sort_values('game_date')
                dfig1 = go.Figure()
                dfig1.add_trace(go.Scatter(x=d_by_game['game_date'], y=d_by_game['xpts'], mode='lines+markers', name='xpts', line=dict(color='orange')))
                dfig1.add_trace(go.Scatter(x=d_by_game['game_date'], y=d_by_game['pts_value'], mode='lines+markers', name='pts_value', line=dict(color='red')))
                dfig1.update_layout(title='Opponent Expected and Actual Points by Game', xaxis_title='Game Date', yaxis_title='Points', legend_title='Metric')
                st.plotly_chart(dfig1, use_container_width=True)
            # 2. Bar chart: Opponent shot type frequency
            with dcol2:
                d_shot_types = defpbp['action_type'].value_counts().reset_index()
                d_shot_types.columns = ['Shot Type', 'Count']
                dfig2 = px.bar(d_shot_types, x='Shot Type', y='Count', color='Count', color_continuous_scale='Reds', title='Opponent Shot Type Frequency (Defender)')
                st.plotly_chart(dfig2, use_container_width=True)
            dcol4, dcol5, dcol6 = st.columns(3)
            # 3. Scatter plot: Opponent shot distance vs. xFG%
            with dcol4:
                dfig3 = px.scatter(
                    defpbp, x='Shot Distance (ft.)', y='xFG%', color=defpbp['made'].map({True: 'Made', False: 'Missed'}),
                    hover_data=['game_date', 'action_type', 'play_descriptors'],
                    title='Opponent Shot Distance vs. xFG% (Defender)',
                    color_discrete_map={'Made': 'green', 'Missed': 'red'}
                )
                st.plotly_chart(dfig3, use_container_width=True)
            # 4. Histogram: Distribution of opponent shot distances
            with dcol6:
                dfig4 = px.histogram(defpbp, x='Shot Distance (ft.)', nbins=20, color_discrete_sequence=['#EF553B'])
                dfig4.update_layout(title='Opponent Shot Distance Distribution (Defender)', xaxis_title='Shot Distance (ft.)', yaxis_title='Count')
                st.plotly_chart(dfig4, use_container_width=True)
            # 5. Radar chart: Defensive descriptor profile
            with dcol5:
                ddef_counts = {desc: defpbp[desc].sum() for desc in descriptor_features}
                ddef_radar_df = pd.DataFrame({'Descriptor': list(ddef_counts.keys()), 'Count': list(ddef_counts.values())})
                dfig5 = go.Figure()
                dfig5.add_trace(go.Scatterpolar(
                    r=ddef_radar_df['Count'],
                    theta=ddef_radar_df['Descriptor'],
                    fill='toself',
                    name='Defensive Descriptor Profile',
                    line=dict(color='red')
                ))
                dfig5.update_layout(
                    polar=dict(radialaxis=dict(visible=True)),
                    title='Defensive Descriptor Profile (Radar Chart)'
                )
                st.plotly_chart(dfig5, use_container_width=True)






