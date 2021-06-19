from datetime import time
import json
from numpy.core.fromnumeric import transpose
from numpy.lib.function_base import copy
from numpy.lib.shape_base import column_stack
import pandas as pd
from pathlib import Path


def main():

    dir = "D:\Data Science\MySportsFeed\python_api\data\\raw\daily_player_gamelogs"
    pathlist = Path(dir).rglob('*.json')

    # Final table (to be filled)
    player_gamelog = pd.DataFrame()

    # loop through json files
    for file in pathlist:

        with open(file) as jsonfile:
            data = json.load(jsonfile)

        print(f"file is {file}")
        if data is None:
            continue
        # checks whether there is expected lineups to avoid crash
        has_info = bool(data['gamelogs'])
        if has_info == False:
            continue

        # df with all data from a particular file (to be filled)
        json_df = pd.DataFrame()
        # information in is gamelogs dictionary key
        # each function extracts the data and generates a row per player per game
        # TODO any way to not join everything like this?
        for gamelog in data['gamelogs']:
            row = get_game(gamelog).join(get_player(gamelog)).join(get_team(gamelog))\
                .join(get_fieldgoals(gamelog)).join(get_freethrows(gamelog)).join(get_rebounds(gamelog))\
                .join(get_offense(gamelog)).join(get_defense(gamelog)).join(get_miscellaneous(gamelog))
            json_df = json_df.append(row)

        player_gamelog = player_gamelog.append(json_df)

    player_gamelog.to_csv("data/parsed/daily_player_gamelog.csv", index=False)

#### Functions ####
# They extract the necessary data from json keys
# TODO should I do it in another way? Also put them in another file?


def get_game(gamelog):
    """ 
    Returns a dataframe with game info.
    1  row, 4 columns
    - game id
    - start time
    - awayteam
    - hometeam 
    """
    # Couldn't create the dataframe straight as columns (need an index (?))
    # easiest way was to make it as just one column and then transpose
    # TODO make this in just one step
    df = pd.DataFrame.from_dict(gamelog['game'], orient="index")
    df_t = df.transpose()
    df_t = df_t.rename(columns={'id': 'gameId'})
    return df_t


def get_player(gamelog):
    """ 
    Returns a dataframe with player info.
    1  row, 5 columns
    - player id
    - first name
    - last Name
    - position
    - jersey number
    """
    df = pd.DataFrame.from_dict(gamelog['player'], orient="index")
    df_t = df.transpose()
    df_t = df_t.rename(columns={'id': 'playerId'})
    return df_t


def get_team(gamelog):
    """ 
    Returns a dataframe with team info.
    1  row, 2 columns
    - team id
    - team
    """
    df = pd.DataFrame.from_dict(gamelog['team'], orient="index")
    df_t = df.transpose()
    df_t = df_t.rename(columns={'id': 'teamId'})
    return df_t


def get_fieldgoals(gamelog):
    """ 
    Returns a dataframe with field goals info.
    1  row, X columns (many columns, same amount for each type of field goal shot)
    - att
    - attpgame (same as att?)
    - made
    - madepgame (same as made?)
    - pct
    """
    df = pd.DataFrame.from_dict(gamelog['stats']['fieldGoals'], orient="index")
    df_t = df.transpose()
    return df_t


def get_freethrows(gamelog):
    """ 
    Returns a dataframe with freethros info.
    1  row, 5 columns 
    - att
    - attpgame (same as att?)
    - made
    - madepgame (same as made?)
    - pct
    """
    df = pd.DataFrame.from_dict(gamelog['stats']['freeThrows'], orient="index")
    df_t = df.transpose()
    return df_t


def get_rebounds(gamelog):
    """ 
    Returns a dataframe with rebounds info.
    1  row, 6 columns 
    - off
    - offpgame (same as off?)
    - def
    - defpgame (same as def?)
    - reb
    - rebpgame
    """
    df = pd.DataFrame.from_dict(gamelog['stats']['rebounds'], orient="index")
    df_t = df.transpose()
    return df_t


def get_offense(gamelog):
    """ 
    Returns a dataframe with offense info.
    1  row, 4 columns 
    - assists
    - assistspgame (same as assists?)
    - points
    - pointspgame (same as points?)
    """
    df = pd.DataFrame.from_dict(gamelog['stats']['offense'], orient="index")
    df_t = df.transpose()
    return df_t


def get_defense(gamelog):
    """ 
    Returns a dataframe with defense info.
    1  row, 4 columns 
    - tov
    - tovpgame (same as tov?)
    - stl
    - stlpgame (same as points?)
    - blk
    - blkpgame
    - blk against
    - blk againstpgame
    """
    df = pd.DataFrame.from_dict(gamelog['stats']['defense'], orient="index")
    df_t = df.transpose()
    return df_t


def get_miscellaneous(gamelog):
    """ 
    Returns a dataframe with other info such as fouls, ejections, starter, time played, plusminus.
    """
    df = pd.DataFrame.from_dict(
        gamelog['stats']['miscellaneous'], orient="index")
    df_t = df.transpose()
    return df_t


# In the end for the get functions to be read
if __name__ == "__main__":
    main()
