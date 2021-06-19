import pandas as pd
import pickle
import json
import numpy as np
import pymc3 as pm
import utils_reg
import parse_game_lineup
import os
import re

# LA PAPA. PODES CAMBIAR OTROS SCRIPTS SIN REINICIAR
%reload_ext autoreload
%autoreload 2


os.chdir("D:\Data Science\MySportsFeed\python_api")
# dates to import
dates_to_import = ['20210331', "20210401"]
pattern = '|'.join(dates_to_import)

# import model
with open("model\pymc.pkl", "rb") as input_file:
    pymc = pickle.load(input_file)

# indexes and data associated to the training process
teams = pd.read_csv("data/working/teams.csv")
players = pd.read_csv("data/working/players.csv")
avg_minutes = pd.read_csv("data/working/avg_minutes.csv")

dir = "D:\Data Science\MySportsFeed\python_api\data\\raw\game_lineup"

lineups_full = pd.DataFrame()

os.chdir(dir)
for filename in os.listdir(dir):
    # only files that match the dates decided
    if re.match('(' + pattern + ')\S*', filename):
        with open(filename) as jsonfile:
            data = json.load(jsonfile)

            # checks whether there is expected lineups to avoid crash
            try:
                has_info = bool(data['teamLineups'][0]['expected']) & bool(data['teamLineups'][1]
                                                                           ['expected'])
            # if file is null has_info fails but should also continue.
            except:
                continue
            if has_info == False:
                continue

            # extract lineup for each team
            lineups_df = pd.DataFrame()
            for i, lineup in enumerate(data['teamLineups']):
                df = parse_game_lineup.get_expected_lineup(lineup, data)
                if i == 0:
                    df['teamId'], df['abb'] = parse_game_lineup.get_away_team(
                        data)
                elif i == 1:
                    df['teamId'], df['abb'] = parse_game_lineup.get_home_team(
                        data)
                lineups_df = lineups_df.append(df)

            lineups_full = lineups_full.append(lineups_df)

lineups_full = lineups_full.rename(columns={'abb': 'abbreviation',
                                            "homeTeamAbb": "homeTeamAbbreviation",
                                            "awayTeamAbb": "awayTeamAbbreviation"})


lineup2 = utils_reg.feature_engineer_1(lineups_full, teams)
lineup2['atHome'] = np.where(
    lineup2['abbreviation'] == lineup2['homeTeamAbbreviation'], 1, 0)

lineup3 = lineup2.merge(players, left_on='id', right_on='playerId', how="left")
lineup3['i'] = np.where(lineup3['i'].isnull(),
                        608., lineup3['i'])  # add fix when player was not used in training (add index to max index possible)


with pymc['model']:
    pm.set_data(
        {'player_index': np.array(lineup3.i.values, dtype=np.int32),
         'opp_team_index': lineup3.oppTeamI.values,
         'athome_var': lineup3.atHome,
         'y_shared': np.zeros(lineup3.shape[0])
         }
    )
    posterior = pm.sample_posterior_predictive(
        pymc['trace'], var_names=['pts_minute'])


posterior_team_long, winner_sample, posterior_team_winners_pct = utils_reg.simulate_from_posterior_sample(
    posterior=posterior, truth=lineup3, avg_minutes=avg_minutes, teams=teams)
# TODO add fix when player was not used in training and doesn't have average minutes (add index to max index possible)
posterior_team_winners_pct['min_odd_needed'] = 1 / \
    posterior_team_winners_pct['pct_win']
