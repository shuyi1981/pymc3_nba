## Param ##
import re
import time
import os
import numpy as np
import pandas as pd
import json
from api import *
from numpy.core.defchararray import index

version = "2.0"
league = "nba"
season = "2019-2020-regular"
format = "json"
feed = "game_playbyplay"
output = "data\\raw\\"
max_tries = 5

# parse_seasonal_games.py must be run first to get the 'games' variable.

### code ###

if not os.path.exists(output + feed):
    os.makedirs(output + feed)


msf = authenticate_api()

games = pd.read_csv("data/parsed/games.csv")

to_download = games[['id', 'startTime', 'awayTeamAbb', 'homeTeamAbb']]
to_download['year'] = to_download.startTime.str[0:4]
to_download['month'] = to_download.startTime.str[5:7]
to_download['day'] = to_download.startTime.str[8:10]
# to_download['day'] = np.where(to_download.startTime.str[11:13].astype(int) > 7, to_download.startTime.str[8:10], (to_download.startTime.str[8:10].astype(int)-1).astype(str))

# to_download['day'] = if to_download.startTime.str[12:14] != '00':
#                         to_download.startTime.str[9:11]
#                      else:
#                          str(int(to_download.startTime.str[9:11])-1)

to_download['gameName'] = to_download['year'] + to_download['month'] +\
    to_download['day'] + "-" +\
    to_download['awayTeamAbb'] + "-" + to_download['homeTeamAbb']

if os.path.isfile(output + feed + "\\" + 'log.csv'):
    already_downloaded = pd.read_csv(
        output + feed + "\\" + 'log.csv', names=['id', 'gameName', 'status'], index_col=None)
else:
    already_downloaded = pd.DataFrame(columns=['id', 'gameName', 'status'])


for game, gameName in zip(to_download.id, to_download.gameName):
    # with open(output + feed + "\\" + 'log.csv', 'w') as fp:
    #     fp.write(str(game))
    completed = already_downloaded[already_downloaded['status']
                                   == "downloaded"]
    if game not in completed.id.unique():
        trial = 1
        while trial < max_tries:
            try:
                data = get_data(version=version, league=league, season=season,
                                feed=feed, format=format, api=msf, game=game)
                status = "downloaded"
                trial = max_tries
            except:
                print(game, gameName)
                status = "failed"
                trial = trial + 1
                time.sleep(1)

        with open(output + feed + "\\" + 'log.csv', 'a') as fp:
            fp.write("{id},{gameName},{status}\n".format(
                id=game, gameName=gameName, status=status))

        with open(output + feed + "\\" + gameName + '.json', 'w') as fp:
            json.dump(data, fp)


# test if everything got downloaded

bajados = [re.sub(".json", "", i) for i in os.listdir(output + feed)]
estado = [d in bajados for d in list(to_download['gameName'])]

analyze = pd.DataFrame(list(zip(
    to_download['gameName'], to_download['id'], falta)), columns=['gN', 'id', 'bajado'])

# these are the ones missing
analyze[analyze['bajado'] == False]

# Manually download in case of need.
# Take care of changing the gameName
data = get_data(version=version, league=league, season=season,
                feed=feed, format=format, api=msf, game=57078)
with open(output + feed + "\\" + '20200815-PHI-HOU' + '.json', 'w') as fp:
    json.dump(data, fp)
