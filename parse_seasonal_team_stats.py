import json
import pandas as pd

file = "D:\Data Science\MySportsFeed\python_api\data\\raw\seasonal_team_stats\\2019-2020-regular.json"

with open(file) as jsonfile:
   data = json.load(jsonfile)


# inspect
data.keys()
data['references']
data['teamStatsTotals'][0]

# TODO
# this is just a copy of another code

import pandas as pd
plays = pd.DataFrame.from_dict(data['teamStatsTotals'][1])

plays.head()

schedule = []
for row in data['teamStatsTotals']:
    schedule.append(row['team'])
schedule_df = pd.DataFrame(schedule)
# extract information from column dictionaries

score = []
for row in data['teamStatsTotals']:
    score.append(row['stats'])
score_df = pd.DataFrame(score)