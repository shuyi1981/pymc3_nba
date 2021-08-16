from utils.test import (format_predictions, read_predictions, is_there_value,
                        is_there_much_value, returns_select)
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
teams = pd.read_csv("data/working/teams.csv")

# if test:
#     parsed_games = "D:\Data Science\MySportsFeed\python_api\data\parsed\playoffs_games.csv"
# else:
#     parsed_games = "D:\Data Science\MySportsFeed\python_api\data\parsed\games.csv"

parsed_games = "D:\Data Science\MySportsFeed\python_api\data\parsed\\all_games.csv"

truth = pd.read_csv(parsed_games)
truth['date'] = truth['startTime'].str[0:10]
id_date = truth.loc[:, ['id', 'date']]

# TODO
# Deberia hacer una funcion genérica que scoree un df con predicciones

predictions = read_predictions(benchmark=benchmark)
predictions = format_predictions(predictions, id_date, teams, odds_reduced)

########
predictions['bet'] = is_there_value(predictions, 'pct_win', 'decimal')
predictions['bet_confident'] = is_there_much_value(
    predictions, 'pct_win', 'decimal', epsilon=0.1)
predictions['earning'] = returns_select(predictions)


# Analisis
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
