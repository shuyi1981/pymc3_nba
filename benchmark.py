from numpy.core.defchararray import join
from numpy.core.numeric import NaN
from numpy.lib.function_base import average
from numpy.lib.histograms import _unsigned_subtract
import pandas as pd
import numpy as np
from pathlib import Path
from pymc3.sampling import sample_posterior_predictive
from sklearn.linear_model import LinearRegression
import pymc3 as pm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.validation import _num_samples
import statsmodels.api as sm
import arviz as az
from theano.tensor.basic import numpy_scalar
import multiprocessing as mp

import utils_reg

# LA PAPA. PODES CAMBIAR OTROS SCRIPTS SIN REINICIAR
%reload_ext autoreload
%autoreload 2

train_sep = 58000
np.random.seed(35)

waic_section = False
multiplicative = True
charts_section = False
residuals_section = False
coef_inspection = False


file = "D:\Data Science\MySportsFeed\python_api\data\parsed\daily_player_gamelog.csv"
parsed_games = "D:\Data Science\MySportsFeed\python_api\data\parsed\games.csv"

daily_gamelog_raw = pd.read_csv(file)

daily_gamelog = daily_gamelog_raw.copy()

# GENERAL CLEANING
# REGULAR + PLAYOFFS TOGETHER
# ASSIGN INDEX TO EACH PLAYER AND TEAM
# I THINK THIS IS USEFUL TO PREDICT LATER (HAVING INDEX FOR ALL THE PLAYERS IN OR NOT IN TRAIN)
# NO NEW FEATURE BASED ON OBS (THAT'S WHY TEST IS OK)

# - filter obs with time played
# - calculate points per minute

daily_gamelog = utils_reg.clean_data_1(daily_gamelog)


daily_gamelog['pts_minute'].describe()
np.log(daily_gamelog.pts_minute+0.1).hist(bins=25)
daily_gamelog.head()
daily_gamelog.tail()

# - get an index for each playerId
# later used to generate a coeff per player
# players has playerId but also an index from 0 to n
players, last_name, num_players = utils_reg.get_player_index(daily_gamelog)

daily_gamelog = pd.merge(daily_gamelog, players.drop(
    columns="lastName"), on="playerId", how="left")


teams, abb, num_teams = utils_reg.get_team_index(daily_gamelog)

daily_gamelog = utils_reg.feature_engineer_1(daily_gamelog, teams)

teams.to_csv("data/working/teams.csv", index=False)

# SPLIT TRAIN AND TEST

train = daily_gamelog.loc[daily_gamelog.gameId < train_sep, :]
test = daily_gamelog.drop(train.index)

type(daily_gamelog['gameId'])
train.head()

# index to pass to model. One intercept per opp team
oppTeam_game = train.oppTeamI.values
# index to pass to model. One intercept per index
player_game = train.i.values

# - Empirical points per minute
# Dataframes are needed beyond the empirical points
games_played = train.groupby(
    ['playerId', 'lastName']).size().reset_index(name='n_games')

ranking = utils_reg.rank_players_by_points(train)
ranking = ranking.merge(games_played.drop(
    columns='lastName'), on="playerId", how="left").sort_values(by="minSeconds")

avg_minutes = utils_reg.get_avg_minutes(ranking)
# median_minutes = utils_reg.get_median_minutes(daily_gamelog_raw)


dummies = pd.get_dummies(player_game)

# - statsmodels lm
# ejemplo, un coef por jugador
# lo mismo pero para obtener directo mas metricas

# smlm = sm.GLM(train.pts_minute, np.ones(train.shape[0]))
smlm = sm.GLM(train.pts_minute, dummies)
# fits y deviance
results = smlm.fit()
print(results.summary())


# use the coefs to estimate points per game
sm_res = results.params.reset_index().merge(
    train, left_on="index", right_on="i", how="left")

sm_res2 = sm_res.merge(
    avg_minutes[['playerId', 'avg_minutes']],     on='playerId', how='left')

sm_res2['points'] = sm_res2.loc[:, 0] * sm_res2.avg_minutes

sm_team_points = sm_res2.groupby(
    ['gameId', 'teamId']).aggregate({'points': 'sum'}).reset_index()
pred = sm_team_points.sort_values(
    by=['points'], ascending=False, kind='mergesort').drop_duplicates(subset=['gameId'])

# team with most points is predicted winner, there aren't probabilities
to_odds = pred.copy()
to_odds['pct_win'] = 1

to_odds2, accuracy = utils_reg.get_accuracy(
    to_odds, parsed_games, 0.5)

to_odds2.to_csv('data/working/benchmark_train.csv', index=False)

# Out of Sample
sm_res_test = results.params.reset_index().merge(
    test, left_on="index", right_on="i", how="inner")

sm_res2_test = sm_res_test.merge(
    avg_minutes[['playerId', 'avg_minutes']],     on='playerId', how='left')
sm_res2_test['points'] = sm_res2_test.loc[:, 0] * sm_res2_test.avg_minutes
sm_team_points_test = sm_res2_test.groupby(
    ['gameId', 'teamId']).aggregate({'points': 'sum'}).reset_index()
pred_test = sm_team_points_test.sort_values(
    by=['points'], ascending=False, kind='mergesort').drop_duplicates(subset=['gameId'])

to_odds = pred_test.copy()
to_odds['pct_win'] = 1

to_odds2, accuracy = utils_reg.get_accuracy(
    to_odds, parsed_games, 0.5)

to_odds2.to_csv('data/working/benchmark_test.csv', index=False)
