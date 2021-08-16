import json
import pathlib
import pandas as pd
from pandas.core.frame import DataFrame
from pathlib import Path


def main():

    dir = "D:\Data Science\MySportsFeed\python_api\data\\raw\game_lineup"
    pathlist = Path(dir).rglob("*.json")
    # pathlist = [dir + "\\" + '20210207-MIN-OKL.json',
    #             dir + "\\" + "20210207-POR-CHA.json", dir + "\\" + '20210207-SAC-LAC.json']

    # file = "D:\Data Science\MySportsFeed\python_api\data\\raw\game_lineup\\20191023-BOS-PHI.json"

    lineups_full = pd.DataFrame()

    for file in pathlist:

        with open(file) as jsonfile:
            data = json.load(jsonfile)

        if data is None:
            continue
        # checks whether there is expected lineups to avoid crash
        has_info = bool(data['teamLineups'][0]['expected']) & bool(data['teamLineups'][1]
                                                                   ['expected'])
        if has_info == False:
            continue

        # extract lineup for each team
        lineups_df = pd.DataFrame()
        for i, lineup in enumerate(data['teamLineups']):
            df = get_expected_lineup(lineup, data)
            if i == 0:
                df['teamId'], df['abb'] = get_away_team(data)
            elif i == 1:
                df['teamId'], df['abb'] = get_home_team(data)
            lineups_df = lineups_df.append(df)

        lineups_full = lineups_full.append(lineups_df)

    lineups_full.to_csv("../data/parsed/lineups.csv", index=False)


def get_gameId(data):
    return data['game'].get('id')


def get_away_team(data):
    id = data['game']['awayTeam'].get('id')
    abb = data['game']['awayTeam'].get('abbreviation')
    return (id, abb)


def get_home_team(data):
    id = data['game']['homeTeam'].get('id')
    abb = data['game']['homeTeam'].get('abbreviation')
    return (id, abb)


def get_expected_lineup(lineup, data):
    df = pd.DataFrame()
    for l in lineup['expected']['lineupPositions']:
        if bool(l.get('player')):
            expected_players = pd.DataFrame.from_dict(
                l['player'], orient='index')
            expected_players_t = expected_players.transpose()
            expected_start = pd.DataFrame(
                {'postion': [l['position']]})
            # expected_start_t = expected_start.transpose()
            expected_line = expected_start.join(expected_players_t)
            df = df.append(expected_line)
    df['gameId'] = get_gameId(data)
    df['awayTeamId'], df['awayTeamAbb'] = get_away_team(data)
    df['homeTeamId'], df['homeTeamAbb'] = get_home_team(data)
    return df


if __name__ == "__main__":
    result = main()

# lineups_df = pd.DataFrame()
# for i,lineup in enumerate(data['teamLineups']):
#     df = get_expected_lineup(lineup)
#     if i == 0:
#         df['teamId'], df['abb'] = get_away_team(data)
#     elif i == 1:
#         df['teamId'], df['abb'] = get_home_team(data)
#     lineups_df = lineups_df.append(df)


# expected = pd.DataFrame(
#         data['teamLineups'][0]['expected']['lineupPositions'])


# len(data['teamLineups'])


# for a,b in enumerate(['a','b']):
#     print(a,b)


# plays = pd.DataFrame.from_dict(data['plays'])

# data['plays'][3]['playStatus']
# data['plays'][41]['substitution']

# lineup = data['teamLineups'][0]

# df = pd.DataFrame()
# for i in lineup['expected']['lineupPositions']:
#     expected_players = pd.DataFrame.from_dict(
#     i['player'], orient= 'index')
#     expected_players_t = expected_players.transpose()
#     expected_start = pd.DataFrame(
#     {'postion': [i['position']]})
#     # expected_start_t = expected_start.transpose()
#     expected_line = expected_start.join(expected_players_t)
#     df = df.append(expected_line)


# for a,b in enumerate(lineup['expected']['lineupPositions']):
#     print(a,b)


# for i in lineup['expected']['lineupPositions']:
#     print(i['position'])


# expected_start = pd.DataFrame(
#     {'postion': [lineup['expected']['lineupPositions'][0]['position']]})

# expected_players = pd.DataFrame.from_dict(
#     lineup['expected']['lineupPositions'][0]['player'], orient= 'index')
# expected_players_t = expected_players.transpose()

# expected_line = expected_start.join(expected_players_t)
# df2 = df.append(expected_line)
