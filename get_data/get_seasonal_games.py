
## Param ##
import os
import json
from utils.api import *
version = "2.0"
league = "nba"
# season = "2019-2020-regular"
season = "2020-2021-regular"
format = "json"
feed = "seasonal_games"
output = "..\\data\\raw\\"

### code ###

if not os.path.exists(output + feed):
    os.makedirs(output + feed)


msf = authenticate_api()
data = get_data(version=version, league=league, season=season,
                feed=feed, format=format, api=msf)

with open(output + feed + "\\" + season + '.json', 'w') as fp:
    json.dump(data, fp)
