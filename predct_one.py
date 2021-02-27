import pandas as pd
import pickle
import json
import numpy as np
import pymc3 as pm
import utils_reg
import parse_game_lineup

with open("model\pymc.pkl", "rb") as input_file:
    pymc = pickle.load(input_file)

teams = pd.read_csv("data/working/teams.csv")
players = pd.read_csv("data/working/players.csv")
avg_minutes = pd.read_csv("data/working/avg_minutes.csv")


with open('data\\raw\\game_lineup\\20210227-IND-BOS.json') as jsonfile:
    data = json.load(jsonfile)
# checks whether there is expected lineups to avoid crash
has_info = bool(data['teamLineups'][0]['expected']) & bool(data['teamLineups'][1]
                                                           ['expected'])
# extract lineup for each team
lineups_df = pd.DataFrame()
for i, lineup in enumerate(data['teamLineups']):
    df = parse_game_lineup.get_expected_lineup(lineup, data)
    if i == 0:
        df['teamId'], df['abb'] = parse_game_lineup.get_away_team(data)
    elif i == 1:
        df['teamId'], df['abb'] = parse_game_lineup.get_home_team(data)
    lineups_df = lineups_df.append(df)


def feature_engineer_adhoc(daily_gamelog, teams):
    """
    Add: oppTeamAbbreviation
    oppTeamId
    teamI
    oppTeami
    """
    daily_gamelog['oppTeamAbbreviation'] = np.where(daily_gamelog['abb'] == daily_gamelog['homeTeamAbb'],
                                                    daily_gamelog['awayTeamAbb'], daily_gamelog['homeTeamAbb'])

    # Add Id of opponent team
    daily_gamelog = pd.merge(daily_gamelog, teams.drop(
        columns="teamI").rename(columns={'teamId': 'oppTeamId', 'abbreviation':     'oppTeamAbbreviation'}), on="oppTeamAbbreviation", how="left")
    # .drop(columns = "abbreviation_y")

    # Add team Index
    daily_gamelog = pd.merge(daily_gamelog, teams.drop(
        columns="abbreviation"), on="teamId", how="left")

    # # Add own team Index (es lo mismo?)
    # daily_gamelog = pd.merge(daily_gamelog, teams.drop(
    #     columns="abbreviation").rename(columns={'teamI': 'TeamI'}), on="teamId",    how="left")

    # Add opponent team Index
    daily_gamelog = pd.merge(daily_gamelog, teams.drop(
        columns="abbreviation").rename(columns={'teamI': 'oppTeamI', 'teamId':  'oppTeamId'}), on="oppTeamId", how="left")
    daily_gamelog['atHome'] = np.where(
        daily_gamelog['abb'] == daily_gamelog['homeTeamAbb'], 1, 0)
    return daily_gamelog


lineup2 = feature_engineer_adhoc(lineups_df, teams)

lineup3 = lineup2.merge(players, left_on='id', right_on='playerId', how="left")

# predict

with pymc['model']:
    pm.set_data(
        {'player_index': lineup3.i.values,
         'opp_team_index': lineup3.oppTeamI.values,
         'athome_var': lineup3.atHome,
         'y_shared': np.zeros(lineup3.shape[0])
         }
    )
    posterior = pm.sample_posterior_predictive(
        pymc['trace'], var_names=['pts_minute'])


posterior_team_long, winner_sample, posterior_team_winners_pct = utils_reg.simulate_from_posterior_sample(
    posterior=posterior, truth=lineup3, avg_minutes=avg_minutes)
