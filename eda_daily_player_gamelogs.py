from numpy.core.defchararray import join
from numpy.core.numeric import NaN
from numpy.lib.function_base import average
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

import utils_reg

file = "D:\Data Science\MySportsFeed\python_api\data\parsed\daily_player_gamelog.csv"
games = "D:\Data Science\MySportsFeed\python_api\data\parsed\games_2020_2021.csv"

daily_gamelog = pd.read_csv(file)

# filter obs with time played
daily_gamelog = daily_gamelog[daily_gamelog['minSeconds'] > 0.0]
# points per minute
daily_gamelog['pts_minute'] = daily_gamelog['pts'] / \
    daily_gamelog['minSeconds']*60

daily_gamelog['pts_minute'].describe()


np.log(daily_gamelog.pts_minute+0.1).hist(bins=25)

# # get an index for each playerId
# # later uesd to generate a coeff per player
# players = daily_gamelog.playerId.unique()
# last_name = daily_gamelog[['playerId', 'lastName']].drop_duplicates()
# players = pd.DataFrame(players, columns = ['playerId'])
# players["i"] = players.index
# players = players.merge(last_name, on = "playerId", how = "left")
# num_players = len(players)

# daily_gamelog = pd.merge(daily_gamelog, players.drop(columns="lastName"), on = "playerId", how = "left")

# player_game = daily_gamelog.i.values


# Empirical points per minute TODO MIRAR UN POCO
games_played = daily_gamelog.groupby(
    ['playerId', 'lastName']).size().reset_index(name='n_games')

ranking = utils_reg.rank_players_by_points(daily_gamelog)
ranking = ranking.merge(games_played.drop(
    columns='lastName'), on="playerId", how="left")

ranking.tail()

avg_minutes = utils_reg.get_avg_minutes(ranking)

# Distribution of points per minute for high presence players
# Normal para los de muchos puntos?
ranking = ranking.sort_values(by="minSeconds")
top_minutes = ranking.iloc[-10:, :].loc[:, ['playerId']
                                        ].merge(daily_gamelog, on="playerId", how="inner")

sns.displot(top_minutes, x="pts_minute", col='lastName', hue="lastName")
g = sns.FacetGrid(top_minutes, col='lastName', col_wrap=3)
g.map(plt.hist, "pts_minute")
