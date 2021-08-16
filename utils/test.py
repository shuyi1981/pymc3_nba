import pandas as pd
import numpy as np


def read_predictions(benchmark=False):
    """ Reads predictions. File must have:
    * gameId
    * teamId
    * winning probability
    that way formar_predictions() can merge with other tables
    """
    if benchmark:
        predictions = pd.read_csv(
            "data/working/benchmark.csv")
    else:
        predictions = pd.read_csv(
            "data/working/posterior_team_winners_pct_expanded.csv")
    return predictions


def format_predictions(predictions, id_date, teams, odds_reduced):

    predictions = predictions.merge(id_date.rename(
        columns={'id': 'gameId'}), on='gameId', how='left')

    predictions = predictions.merge(teams.drop(
        columns="teamI"), on='teamId', how='left')

    predictions = predictions.merge(odds_reduced.rename(columns={
                                    'newDate': 'date', 'Abb': 'abbreviation'}),     on=['date', 'abbreviation'], how='left')

    # this is for NaN
    # predictions['mod_day'] = np.where(predictions['decimal'].isnull(),    predictions['date'].str[-2:].astype(int)-1,predictions['date'].str[-2:].   astype(int))

    predictions['date'] = np.where(predictions['decimal'].isnull(), pd. to_datetime(
        predictions.date) + pd.DateOffset(-1), pd.to_datetime(predictions.date))
    predictions['date'] = predictions['date'].astype(str)

    predictions = predictions.drop(columns="decimal").merge(odds_reduced.rename(columns={
        'newDate': 'date', 'Abb': 'abbreviation'}), on=['date', 'abbreviation'],    how='left')

    return predictions


def is_there_value(df, win_pct, odd):
    """
    if implicit even odd is lower than actual odd
    then there is value.
    Flag Bet.
    """
    bet = np.where(1/df[win_pct] < df[odd], 1, 0)
    return bet


def is_there_much_value(df, win_pct, odd, epsilon=0.1):
    """
    if implicit even odd is lower than actual odd
    then there is value.
    Flag Bet.
    """
    bet = np.where(1/(df[win_pct]+epsilon) < df[odd], 1, 0)
    return bet


def returns_select(df):
    """
    Calculate returns per observation
    """
    conditions = [
        df.bet == 0,
        ((df.bet == 1) & ~(df.winner == df.teamId)),
        ((df.bet == 1) & (df.winner == df.teamId))
    ]

    choice = [
        0,
        -1,
        df.decimal - 1
    ]

    output = np.select(conditions, choice, default=0)
    return output
