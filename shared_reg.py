from numpy.core.defchararray import join
from numpy.core.numeric import NaN
from numpy.lib.arraysetops import unique
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
import pickle

import utils_reg

# LA PAPA. PODES CAMBIAR OTROS SCRIPTS SIN REINICIAR
from IPython import get_ipython
ipython = get_ipython()

if '__IPYTHON__' in globals():
    ipython.magic('load_ext autoreload')
    ipython.magic('autoreload 2')

train_sep = 63447
np.random.seed(35)

waic_section = False
multiplicative = True
charts_section = True
residuals_section = True
coef_inspection = True
save_model = True


file = "D:\Data Science\MySportsFeed\python_api\data\parsed\daily_player_gamelog.csv"

parsed_games = "D:\Data Science\MySportsFeed\python_api\data\parsed\\all_games.csv"

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

# daily_gamelog.iloc[0:5000, :].to_csv(
#     "data/working/daily_gamelog_reduced.csv", index=False)

players.to_csv("data/working/players.csv", index=False)

teams, abb, num_teams = utils_reg.get_team_index(daily_gamelog)

daily_gamelog = utils_reg.feature_engineer_1(daily_gamelog, teams)
# daily_gamelog.to_csv('data/working/debug.csv')
teams.to_csv("data/working/teams.csv", index=False)

daily_gamelog['rest_hours'].describe()
daily_gamelog.rest_hours.hist(bins=10)

# daily_gamelog.rest_hours/np.timedelta64(1)

#  tipos = [type(i) for i in daily_gamelog.startTime_date]
# set(tipos)
# tipos = [type(i) for i in daily_gamelog['diff']]
# set(tipos)

# daily_gamelog['diff'] = daily_gamelog.groupby(
#         'playerId').startTime_date.diff()
# import datetime
# daily_gamelog['diff'] = daily_gamelog['diff'].fillna(datetime.timedelta(3))

# daily_gamelog['diff']/np.timedelta64(1,'h')
# daily_gamelog.loc[daily_gamelog['diff'].apply(lambda x: isinstance(x,float)), :]

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

avg_minutes.to_csv("data/working/avg_minutes.csv", index=False)
# median_minutes = utils_reg.get_median_minutes(daily_gamelog_raw)


# # Bayesian, revisar forma funcional

# with pm.Model() as points_minute_model:
#     # global mode parameters
#     # intercept = pm.Flat("intercept")
#     sigma = pm.Uniform('sigma', lower=0, upper=10)
#     home_effect = pm.Normal('home_effect', mu=0, sigma=0.5)
#     # player specific parameters
#     points_players = pm.Normal(
#         "points_players", mu=0.1, sigma=0.1, shape=num_players)  # ??

#     oppTeamEffect = pm.Normal("oppTeamEffect", mu=0,
#                               sigma=0.1, shape=num_teams)

#     players_mu = points_players[player_game] + home_effect * \
#         train.atHome + oppTeamEffect[oppTeam_game]
#     # relation between variables ?
#     pts_minute = pm.Normal('pts_minute', mu=players_mu,
#                            sd=sigma, observed=train.pts_minute)
#     trace_points_minute_model = pm.sample(2000, tune=1000, cores=1)


# pm.traceplot(trace_points_minute_model)

# ROBUSTA  (por student (?))

with pm.Model() as points_minute_model:
    # global mode parameters
    # intercept = pm.Flat("intercept")
    player_index = pm.Data("player_index", player_game)
    opp_team_index = pm.Data("opp_team_index", oppTeam_game)
    athome_var = pm.Data("athome_var", train.atHome)
    rest_hours_var = pm.Data('rest_hours_var', np.log(train.rest_hours))
    y_shared = pm.Data("y_shared", train.pts_minute)

    sigma = pm.Uniform('sigma', lower=0, upper=10)
    home_effect = pm.Normal('home_effect', mu=0, sigma=0.5)
    rest_hours_effect = pm.Normal('rest_hours_effect', mu=0, sigma=0.2)
    nu = pm.Exponential('nu', 1./10, testval=5.)
    # player specific parameters
    points_players = pm.Normal(
        "points_players", mu=0.1, sigma=0.1, shape=num_players)  # ??

    # sigma_players = pm.Uniform('sigma_players', lower = 0, upper = 1, shape = num_players)

    oppTeamEffect = pm.Normal("oppTeamEffect", mu=0,
                              sigma=0.1, shape=num_teams)

    # players_mu = points_players[player_game] + home_effect * \
    #     daily_gamelog.atHome + oppTeamEffect[oppTeam_game]

    # multiplicative
    if multiplicative:
        players_mu = points_players[player_index]*home_effect ** \
            athome_var * oppTeamEffect[opp_team_index] * rest_hours_effect ** \
            rest_hours_var

    else:
        players_mu = points_players[player_index] + home_effect * \
            athome_var + oppTeamEffect[opp_team_index] + rest_hours_effect * \
            rest_hours_var

    # # relation between variables ?
    # pts_minute = pm.StudentT('pts_minute', nu=nu, mu=players_mu,
    #                          sigma=sigma, observed=daily_gamelog.pts_minute)

    pts_minute = pm.StudentT('pts_minute', nu=nu, mu=players_mu,
                             sigma=sigma, observed=y_shared)
    trace_points_minute_model = pm.sample(2000, tune=2000, cores=1)


# Save Model
if save_model:

    utils_reg.pickle_model(output_path='model\pymc.pkl', model=points_minute_model,
                           trace=trace_points_minute_model)

with open("model\pymc.pkl", "rb") as input_file:
    pymc = pickle.load(input_file)

# # LogNormal (ponele)
# with pm.Model() as points_minute_model:
#     # global mode parameters
#     # intercept = pm.Flat("intercept")
#     sigma = pm.Uniform('sigma', lower=0, upper=10)
#     home_effect = pm.Normal('home_effect', mu=0, sigma=0.5)
#     # player specific parameters
#     points_players = pm.Normal(
#         "points_players", mu=0, sigma=0.1, shape=num_players)  # ??

#     oppTeamEffect = pm.Normal("oppTeamEffect", mu=0,
#                               sigma=0.1, shape=num_teams)

#     players_mu = np.exp(points_players[player_game] + home_effect *
#                         daily_gamelog.atHome + oppTeamEffect[oppTeam_game])
#     # relation between variables ?
#     pts_minute = pm.Normal('pts_minute', mu=players_mu,
#                            sigma=sigma, observed=train.pts_minute)
#     trace_points_minute_model = pm.sample(2000, tune=1000, cores=1)


# en terminos de MSE y MAPE
# Normal y Student dieron casi igual
# Lognormal dio levemento peor (pero con coefs batante distintos)

# WAIC

if waic_section:

    waic = pm.waic(trace_points_minute_model, points_minute_model)

    waic.waic  # waic o elpd_waic
    -2 * waic.waic - waic.p_waic

    pm.summary(trace_points_minute_model).round(2)
    divergent = trace_points_minute_model['diverging']
    print("Number of Divergent %d" % divergent.nonzero()[0].size)
# posterior prediction
"""
Returns a dict where key = var
The values is a np matrix. 
[samples * obs]
As many as samples required (2k in this case, twice de trace samples I think).
Each of the rows has a sample from the posterior for each obs. In this case for each player/game 
"""
with points_minute_model:
    post_pred = pm.sample_posterior_predictive(
        trace_points_minute_model, var_names=['pts_minute'])

# check players
# Anthony Davis
len(list(post_pred.values())[0][0])
# len(list(post_pred_pickle.values())[0][0])


list(post_pred.values())[0][0][train['i'] == 8].mean()  # 0.69925
list(post_pred.values())[0][1][train['i'] == 8].mean()  # 0.7497


# Danny Green
list(post_pred.values())[0][0][train['i'] == 38].mean()
list(post_pred.values())[0][1][train['i'] == 38].mean()


if charts_section:

    charts = 8

    # residuals for particular players

    def res_player(i):
        res = list(post_pred.values())[0][0][train['i'] == i] - \
            train.loc[train['i'] == i, 'pts_minute']
        # sns.scatterplot(x = np.arange(0,len(res)), y = res, ax = axs[i-1])
        # axs[i-1].set(ylabel = "res")
        sns.histplot(x=res, ax=axs[i-1])

    fig, axs = plt.subplots(charts, figsize=(10, 20))
    for i in range(1, charts):
        res_player(i)

    sns.scatterplot(x=list(post_pred.values())[
                    0][0][train['i'] == 8], y=train[train['i'] == 8].pts_minute)


# check residuals
if residuals_section:

    for b in post_pred.values():
        avg_pred = pd.DataFrame(np.transpose(
            b)).add_prefix("sample").mean(axis=1)

    residuals = train.pts_minute - avg_pred

    toplot = residuals[(residuals < 1.5) & (residuals > -1.5)]

    # ALL
    sns.scatterplot(x=np.arange(0, len(residuals)),
                    y=residuals, hue=train.teamId)

    # Without extreme residuals
    sns.scatterplot(x=np.arange(0, len(toplot)),
                    y=toplot)

    plt.figure(figsize=(8, 60))
    pm.forestplot(trace_points_minute_model, var_names=[
                  'points_players', 'oppTeamEffect'])
    plt.figure(figsize=(8, 60))
    pm.forestplot(trace_points_minute_model, var_names=[
                  'rest_hours_effect'])

    # Metrics en training

    def mse(pred, true):
        MSE = ((true - pred)**2).mean()
        # print((true - pred)**2)
        return MSE

    def mape(pred, true):
        MAPE = abs(true-pred).mean()
        return MAPE

    utils_reg.mse(avg_pred, train.pts_minute)

    utils_reg.mape(avg_pred, train.pts_minute)
""" 
Cada sample juntarla con playerId, gameId, teamId.

"""

posterior_team_long, winner_sample, posterior_team_winners_pct = utils_reg.simulate_from_posterior_sample(
    posterior=post_pred, truth=train, avg_minutes=avg_minutes, teams=teams)

# returns comparison between prediction and truth
# in this case is using the training data
# first argument must have:
# - teamId
# - gameId
# - win %
post_pred_team_winners_pct_expanded, accuracy = utils_reg.get_accuracy(
    posterior_team_winners_pct, parsed_games, 0.5)
print(accuracy)
# Accuracy by bin
bin_accuracy = utils_reg.bin_results(post_pred_team_winners_pct_expanded, 11)

sns.barplot(x='pct_win', y='actual_win', data=bin_accuracy)
# train.head()
# Accuracy by bin

if coef_inspection:
    # ??? algunas metricas de los coef
    bfmi = np.max(pm.stats.bfmi(trace_points_minute_model))
    max_gr = max(np.max(gr_stats) for gr_stats in pm.stats.rhat(
        trace_points_minute_model).values()).values

    (
        pm.energyplot(trace_points_minute_model, legend=False, figsize=(6, 4)).set_title(
            f"BFMI = {bfmi}\nGelman-Rubin = {max_gr}"
        )
    )

    # arma tabla con resultados de coeficiente por jugador
    df_hpd = pd.DataFrame(
        pm.stats.hpd(trace_points_minute_model["points_players"]), columns=["hpd_low", "hpd_high"], index=players.playerId.values
    )

    df_median = pd.DataFrame(
        np.quantile(trace_points_minute_model["points_players"], 0.5, axis=0), columns=["hpd_median"], index=players.playerId.values
    )
    df_hpd = df_hpd.join(df_median).merge(
        last_name, left_index=True, right_on="playerId",  how="left")
    df_hpd["relative_lower"] = df_hpd.hpd_median - df_hpd.hpd_low
    df_hpd["relative_upper"] = df_hpd.hpd_high - df_hpd.hpd_median
    df_hpd = df_hpd.sort_values(by="hpd_median")
    df_hpd = df_hpd.reset_index()
    df_hpd["x"] = df_hpd.index + 0.5

    fig, axs = plt.subplots(figsize=(10, 4))
    axs.errorbar(
        df_hpd.x[np.arange(0, 100, 2)],
        df_hpd.hpd_median[np.arange(0, 100, 2)],
        yerr=(df_hpd[["relative_lower", "relative_upper"]
                     ].values[np.arange(0, 100, 2)]).T,
        fmt="o",
    )
    axs.set_title("HPD of Attack Strength, by Team")
    axs.set_xlabel("pl")
    axs.set_ylabel("Posterior Attack Strength")
    _ = axs.set_xticks(df_hpd.index[np.arange(0, 100, 2)] + 0.5)
    _ = axs.set_xticklabels(
        df_hpd["lastName"].values[np.arange(0, 100, 2)], rotation=45)

    #
    fig, axs = plt.subplots(figsize=(10, 4))
    sns.pointplot(x=df_hpd.x[np.arange(0, 100, 2)],
                  y=df_hpd.hpd_median[np.arange(0, 1000, 2)], ax=axs)
    sns.barplot(x=df_hpd.x[np.arange(0, 1000, 2)],
                y=df_hpd.relative_upper[np.arange(0, 1000, 2)], ax=axs)

# Out of sample
# CAMBIAR Y BORRAR TODO LO QUE NO HAGA FALTA

# parsed_games = "D:\Data Science\MySportsFeed\python_api\data\parsed\playoffs_games.csv"

with open("model\pymc.pkl", "rb") as input_file:
    pymc = pickle.load(input_file)

with pymc['model']:
    pm.set_data(
        {'player_index': test.i.values,
         'opp_team_index': test.oppTeamI.values,
         'athome_var': test.atHome,
         'y_shared': np.zeros(test.shape[0])
         }
    )
    posterior = pm.sample_posterior_predictive(
        pymc['trace'], var_names=['pts_minute'])


# check players
# Anthony Davis
len(list(posterior.values())[0][0])


list(posterior.values())[0][0][test['i'] == 8].mean()  # 0.69925
list(posterior.values())[0][1][test['i'] == 8].mean()  # 0.7497

# Danny Green
list(posterior.values())[0][0][test['i'] == 38].mean()
list(posterior.values())[0][1][test['i'] == 38].mean()

if charts_section:

    charts = 8

    def res_player(i):
        res = list(posterior.values())[0][0][test['i'] == i] - \
            test.loc[test['i'] == i, 'pts_minute']
        # sns.scatterplot(x = np.arange(0,len(res)), y = res, ax = axs[i-1])
        # axs[i-1].set(ylabel = "res")
        sns.histplot(x=res, ax=axs[i-1])

    fig, axs = plt.subplots(charts, figsize=(10, 20))
    for i in range(1, charts):
        res_player(i)

    sns.scatterplot(x=list(posterior.values())[
                    0][0][test['i'] == 8], y=test[test['i'] == 8].pts_minute)

if residuals_section:

    # check residuals
    for b in posterior.values():
        avg_pred_test = pd.DataFrame(np.transpose(b)).add_prefix(
            "sample").mean(axis=1).reset_index(drop=True)

    residuals = test.pts_minute.reset_index(
        drop=True) - avg_pred_test.reset_index(drop=True)

    toplot = residuals[(residuals < 1.5) & (residuals > -1.5)]

    sns.scatterplot(x=np.arange(0, len(residuals)),
                    y=residuals, hue=train.teamId)

    sns.scatterplot(x=np.arange(0, len(toplot)),
                    y=toplot)

    plt.figure(figsize=(8, 60))
    pm.forestplot(trace_points_minute_model, var_names=[
                  'points_players', 'oppTeamEffect'])

    utils_reg.mse(avg_pred_test, test.pts_minute)

    utils_reg.mape(avg_pred_test, test.pts_minute)

#------------------------------------------#

# predecir partidos
posterior_team_long, winner_sample, posterior_team_winners_pct = utils_reg.simulate_from_posterior_sample(
    posterior=posterior, truth=test, avg_minutes=avg_minutes, teams=teams)


# playoff_parsed_games = "D:\Data Science\MySportsFeed\python_api\data\parsed\playoffs_games.csv"
posterior_team_winners_pct_expanded, accuracy = utils_reg.get_accuracy(
    posterior_team_winners_pct, parsed_games, 0.5)
print(accuracy)


posterior_team_winners_pct_expanded.to_csv(
    "data/working/posterior_team_winners_pct_expanded.csv", index=False)
# Accuracy by bin
bin_accuracy = utils_reg.bin_results(posterior_team_winners_pct_expanded, 11)

sns.barplot(x='pct_win', y='actual_win', data=bin_accuracy)
