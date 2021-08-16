import json
import pandas as pd

file = "D:\Data Science\MySportsFeed\python_api\data\\raw\game_playbyplay\\20191023-BOS-PHI.json"

with open(file) as jsonfile:
   data = json.load(jsonfile)


# inspect
data.keys()
data['references'].keys()
data['references']['playerReferences']

plays = pd.DataFrame.from_dict(data['plays'])

data['plays'][3]['playStatus']
data['plays'][41]['substitution']