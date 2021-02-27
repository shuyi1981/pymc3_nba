import pandas as pd
import numpy as np
import pickle


def get_player_index(daily_gamelog):
    r"""Returns
    - players : df with id, lastname and index
    - last_name : df with id and last_name
    - num_players : # of unique players 
    """
    players = daily_gamelog.playerId.unique()
    last_name = daily_gamelog[['playerId', 'lastName']].drop_duplicates()
    players = pd.DataFrame(players, columns=['playerId'])
    players["i"] = players.index
    players = players.merge(last_name, on="playerId", how="left")
    num_players = len(players)

    return players, last_name, num_players


def get_team_index(daily_gamelog):
    r"""Returns
    - teams : df with id, abbreviation and index
    - abbrevation : df with id and abbreviation
    - num_teams : # of unique teams 
    """
    teams = daily_gamelog.teamId.unique()
    abb = daily_gamelog[['teamId', 'abbreviation']].drop_duplicates()
    teams = pd.DataFrame(teams, columns=['teamId'])
    teams["teamI"] = teams.index
    teams = teams.merge(abb, on="teamId", how="left")
    num_teams = len(teams)

    return teams, abb, num_teams


def rank_players_by_points(daily_gamelog):
    """ Returns ranking: df ordered by points per minute """
    ranking = daily_gamelog.groupby(['playerId', 'lastName']).aggregate({'pts': 'sum',
                                                                         'minSeconds': 'sum'}).reset_index()

    ranking['pts_minute'] = ranking['pts']/ranking['minSeconds']*60
    ranking = ranking.sort_values(by="pts_minute")

    return ranking


def get_avg_minutes(ranking):
    """Returns df: average minutes per game per player"""
    avg_minutes = ranking.groupby('playerId').aggregate({'minSeconds': 'sum',
                                                         'n_games': 'sum'}).reset_index()
    avg_minutes['avg_minutes'] = avg_minutes['minSeconds'] / \
        60/avg_minutes['n_games']

    return avg_minutes


def get_median_minutes(ranking):
    """Returns df: median minutes per game per player"""
    median_minutes = ranking.groupby('playerId').aggregate(
        {'minSeconds': 'median'}).reset_index()
    median_minutes['median_minutes'] = median_minutes['minSeconds'] / \
        60
    return median_minutes


def get_accuracy(prediction, parsed_games, threshold):
    """ Returns:
    -----------
    - df with actual winner of each game
    - accuracy metric based on threshold

    Input:
    ------
    - df with gameId, teamId and pct_win (chance given by model)

    """
    truth_reduced = get_ground_truth(parsed_games)

    n_games = len(prediction.gameId.unique())
    prediction = prediction.merge(
        truth_reduced, left_on='gameId', right_on="id", how='left')

    prediction['correct'] = np.where(
        prediction['winner'] == prediction['teamId'], 1, 0)

    accuracy = prediction[prediction['pct_win']
                          > threshold].correct.sum()/n_games
    return prediction, accuracy


def get_ground_truth(parsed_games):
    "Reads parsed_games csv and returns the winner by gameId"
    truth = pd.read_csv(parsed_games)
    truth_reduced = truth[['id', 'awayTeamId',
                           'homeTeamId', 'awayScoreTotal', 'homeScoreTotal']]
    truth_reduced['winner'] = np.where(truth_reduced['homeScoreTotal'] > truth_reduced['awayScoreTotal'],
                                       truth_reduced['homeTeamId'], truth_reduced['awayTeamId'])
    truth_reduced = truth_reduced[['id', 'winner']]
    return truth_reduced


def bin_results(predictions, n_bins):
    """Split predictions in bins and returns df with accuracy by bin.
    11 bins is from 0 to 1 by 0.1"""
    bins = np.linspace(0, 1, n_bins)
    df = predictions.groupby(pd.cut(predictions.pct_win, bins)).aggregate(
        {'correct': 'sum'}).reset_index()
    size = predictions.groupby(
        pd.cut(predictions.pct_win, bins)).size().reset_index(name="n_games")
    output = df.join(size.drop(columns='pct_win'))
    output['actual_win'] = output['correct']/output['n_games']
    return output


def index_winner_by_game_sample(df):
    # https://stackoverflow.com/questions/50381064/select-the-max-row-per-group-pandas-performance-issue/50389889?noredirect=1#comment87795818_50389889
    return df.sort_values(by=['points'], ascending=False, kind='mergesort').drop_duplicates(subset=['gameId', 'sample'])


def clean_data_1(daily_gamelog):
    """
    Removes obs without minutes
    Calculates pts per minute
    Flag if playing at home
    """
    daily_gamelog = daily_gamelog[daily_gamelog['minSeconds'] > 0.0]
    daily_gamelog['pts_minute'] = daily_gamelog['pts'] / \
        daily_gamelog['minSeconds']*60
    daily_gamelog['atHome'] = np.where(
        daily_gamelog['abbreviation'] == daily_gamelog['homeTeamAbbreviation'], 1, 0)
    return daily_gamelog


def feature_engineer_1(daily_gamelog, teams):
    """
    Add: oppTeamAbbreviation
    oppTeamId
    teamI
    oppTeami
    """
    daily_gamelog['oppTeamAbbreviation'] = np.where(daily_gamelog['abbreviation'] == daily_gamelog['homeTeamAbbreviation'],
                                                    daily_gamelog['awayTeamAbbreviation'], daily_gamelog['homeTeamAbbreviation'])

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
    return daily_gamelog


def simulate_from_posterior_sample(posterior, truth, avg_minutes):

    identifier = truth.loc[:, ['playerId',
                               'gameId', 'teamId']].reset_index(drop=True)

    # - transponemos samples, renombramos y agregamos minutos promedio por jugador
    # loop pero hay un solo value en el dict. Me resulto util para usarlo pero no hay   una iteracion # real
    # AVG minutes
    for b in posterior.values():
        posterior_t = pd.DataFrame(np.transpose(b)).add_prefix("sample").join(
            identifier).merge(avg_minutes[['playerId', 'avg_minutes']],     on='playerId', how='left')

    # # Median minutes
    # for b in posterior.values():
    #     posterior_t = pd.DataFrame(np.transpose(b)).add_prefix("sample").join(
    #         identifier).merge(median_minutes[['playerId', 'median_minutes']],     on='playerId', how='left')

    # - calculamos puntos en partido.
    # pts_minute * avg_minutes
    sample_cols = [col for col in posterior_t.columns if 'sample' in col]
    posterior_t_mult = posterior_t.copy()
    for col in sample_cols:
        posterior_t_mult[col] = posterior_t[col] * posterior_t.avg_minutes

    # suma para cada equipo/partido los puntos en cada sample
    # dict para sumar cada columna que empieza con sample # individualmente
    agg_dict = {i: 'sum' for i in sample_cols}
    posterior_team_points = posterior_t_mult.groupby(
        ['gameId', 'teamId']).aggregate(agg_dict).reset_index()

    # posterior_team_points.head()

    posterior_team_long = posterior_team_points.melt(
        id_vars=['gameId', 'teamId'], var_name='sample', value_name="points")

    # posterior_team_long['sample'] = posterior_team_long['sample'].str.replace ("sample",'').astype(int64)

    # posterior_team_long.head()
    posterior_team_long['match_id'] = posterior_team_long.index

    # sns.displot(
    #     posterior_team_long[posterior_team_long['gameId'] == 53038], x="points",  hue="teamId")

    winner_sample = index_winner_by_game_sample(posterior_team_long)
    # winner_sample.head()

    # returns the % of wins for the sample for each team
    # % winning prediction
    posterior_team_winners = posterior_team_long.iloc[winner_sample.reset_index(
    ).match_id, :]
    posterior_team_winners_pct = posterior_team_winners.groupby(
        ['gameId', 'teamId']).size().reset_index(name='n_won')

    posterior_team_winners_pct['pct_win'] = posterior_team_winners_pct['n_won'] / \
        len(posterior_team_long['sample'].unique())

    return posterior_team_long, winner_sample, posterior_team_winners_pct


def mse(pred, true):
    pred = pred.reset_index(drop=True)
    true = true.reset_index(drop=True)
    MSE = ((true - pred)**2).mean()
    # print((true - pred)**2)
    return MSE


def mape(pred, true):
    pred = pred.reset_index(drop=True)
    true = true.reset_index(drop=True)

    MAPE = abs(true-pred).mean()
    return MAPE


def pickle_model(output_path: str, model, trace):
    """Pickles PyMC3 model and trace"""
    with open(output_path, "wb") as buff:
        pickle.dump({"model": model, "trace": trace}, buff)
