import utils_reg
import seaborn as sns
import pandas as pd
import numpy as np
from pandas._config.config import describe_option

test = True
benchmark = False


odds_reduced = pd.read_csv(
    "data/parsed/all_odds_reduced.csv")

abbs = pd.read_csv("data/parsed/abbreviations.csv")

# if test:
#     parsed_games = "D:\Data Science\MySportsFeed\python_api\data\parsed\playoffs_games.csv"
# else:
#     parsed_games = "D:\Data Science\MySportsFeed\python_api\data\parsed\games.csv"

parsed_games = "D:\Data Science\MySportsFeed\python_api\data\parsed\\all_games.csv"

truth = pd.read_csv(parsed_games)
truth['date'] = truth['startTime'].str[0:10]
id_date = truth.loc[:, ['id', 'date']]

if benchmark:
    predictions = pd.read_csv(
        "data/working/benchmark.csv")
else:
    predictions = pd.read_csv(
        "data/working/posterior_team_winners_pct_expanded.csv")

predictions = predictions.merge(id_date.rename(
    columns={'id': 'gameId'}), on='gameId', how='left')

teams = pd.read_csv("data/working/teams.csv")

predictions = predictions.merge(teams.drop(
    columns="teamI"), on='teamId', how='left')


predictions = predictions.merge(odds_reduced.rename(columns={
                                'newDate': 'date', 'Abb': 'abbreviation'}), on=['date', 'abbreviation'], how='left')

# this is for NaN
# predictions['mod_day'] = np.where(predictions['decimal'].isnull(), predictions['date'].str[-2:].astype(int)-1,predictions['date'].str[-2:].astype(int))

predictions['date'] = np.where(predictions['decimal'].isnull(), pd.to_datetime(
    predictions.date) + pd.DateOffset(-1), pd.to_datetime(predictions.date))
predictions['date'] = predictions['date'].astype(str)

predictions = predictions.drop(columns="decimal").merge(odds_reduced.rename(columns={
    'newDate': 'date', 'Abb': 'abbreviation'}), on=['date', 'abbreviation'], how='left')

########


def is_there_value(df, win_pct, odd):
    """
    if implicit even odd is lower than actual odd
    then there is value.
    Flag Bet.
    """
    bet = np.where(1/df[win_pct] < df[odd], 1, 0)
    return bet


def is_there_much_value(df, win_pct, odd):
    """
    if implicit even odd is lower than actual odd
    then there is value.
    Flag Bet.
    """
    bet = np.where(1/(df[win_pct]+0.1) < df[odd], 1, 0)
    return bet


def returns(df):
    """
    Calculate returns per observation
    """
    if df.bet == 0:
        return 0
    elif (df.bet == 1) & (not df.winner == df.teamId):
        return -1
    elif (df.bet == 1) & (df.winner == df.teamId):
        return df.decimal - 1


predictions['bet'] = is_there_value(predictions, 'pct_win', 'decimal')
predictions['bet'] = is_there_much_value(predictions, 'pct_win', 'decimal')


predictions['earning'] = predictions.apply(returns, axis=1)

predictions.earning.sum()
predictions.bet.sum()

predictions.earning.sum()/predictions.bet.sum()

predictions.loc[predictions.earning > 0, :]
predictions.loc[predictions.earning < 0, :]

predictions.loc[(predictions.earning != 0) & (predictions.pct_win < 0.1961), :]

aa = utils_reg.bin_results(predictions, 11)
print(aa)


def left_bound(int):
    return int.left


graph = sns.scatterplot(x=aa.pct_win.apply(left_bound), y=aa.actual_win)

bins = 11
predictions.groupby(pd.cut(predictions.pct_win, bins)).aggregate(
    {'earning': 'sum',
     'bet': 'sum'}).reset_index()


bb = utils_reg.bin_results(predictions[predictions.earning > 0], 11)
sns.scatterplot(x=bb.pct_win.apply(left_bound), y=bb.actual_win)


sns.histplot(x=predictions.loc[predictions.earning > 0, 'pct_win'],
             y=predictions.loc[predictions.earning > 0, 'correct'])
sns.histplot(
    x=predictions.loc[predictions.earning > 0, 'pct_win'], stat='count')

sns.histplot(x=predictions.loc[predictions.earning < 0, 'pct_win'])


predictions.loc[predictions.pct_win < 0.1, :]
predictions.loc[predictions.pct_win > 0.9, :]

aa.pct_win[0].left

b = aa.pct_win.apply(left_bound)
