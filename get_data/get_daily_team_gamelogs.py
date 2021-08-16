## Param ##
import datetime
import re
import time
import os
import numpy as np
import pandas as pd
import json
from utils.api import *
from numpy.core.defchararray import index
version = "2.0"
league = "nba"
season = "2019-2020-regular"
format = "json"
feed = "daily_team_gamelogs"
output = "..\data\\raw\\"
max_tries = 5
startdate = '2019-10-22'
enddate = '2020-08-16'

""" 
This feed must be downloaded by date and not by game id
We create a range of dates with +1 day at each end to control for timezones
"""

### code ###

if not os.path.exists(output + feed):
    os.makedirs(output + feed)


dates_to_download = pd.date_range(startdate, enddate).strftime('%Y%m%d')

msf = authenticate_api()

if os.path.isfile(output + feed + "\\" + 'log.csv'):
    already_downloaded = pd.read_csv(
        output + feed + "\\" + 'log.csv', names=['date', 'status'], index_col=None)
else:
    already_downloaded = pd.DataFrame(columns=['date', 'status'])


for date in dates_to_download:
    # with open(output + feed + "\\" + 'log.csv', 'w') as fp:
    #     fp.write(str(game))
    completed = already_downloaded[already_downloaded['status']
                                   == " downloaded"]
    if date not in completed.date.unique():
        trial = 1
        while trial < max_tries:
            try:
                data = get_data(version=version, league=league, season=season,
                                feed=feed, format=format, api=msf, date=date)
                status = "downloaded"
                trial = max_tries
            except:
                print(date)
                status = "failed"
                trial = trial + 1
                time.sleep(1)

        with open(output + feed + "\\" + 'log.csv', 'a') as fp:
            fp.write("{date},{status}\n".format(date=date, status=status))

        with open(output + feed + "\\" + date + '.json', 'w') as fp:
            json.dump(data, fp)
