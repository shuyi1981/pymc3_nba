
## Param ##
version = "2.0"
league = "nba"
season = "2019-2020-regular"
format = "json"
feed = "seasonal_team_stats"
output = "data\\raw\\"

### code ###
from api import *
import json
import os

if not os.path.exists(output + feed):
    os.makedirs(output + feed)


msf = authenticate_api()
data = get_data(version=version,league=league, season=season, feed=feed, format=format, api=msf)

with open(output + feed + "\\" + season + '.json', 'w') as fp:
    json.dump(data, fp)



