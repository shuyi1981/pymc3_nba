import pandas as pd
import arrow
import numpy as np

data = pd.read_csv("data/working/daily_gamelog_reduced.csv")

data = data.loc[data['minSeconds'] > 300, ['gameId', 'startTime', 'playerId']]

data['date2'] = data.apply(lambda row: arrow.get(row['startTime']), axis=1)


data.columns

data

type(data['startTime'][0])

now = arrow.now()
now

arrow.get('2019-10-23T00:00:00.000Z')
arrow.get('2019-10-24T02:00:00.000Z').to('US/Pacific')


aa = data.loc[data['playerId'].isin([17208, 9097]), :]

bb = aa.sort_values(by=['playerId', 'date2'])
bb['rest'] = bb.groupby('playerId').date2.diff()/np.timedelta64(1, 'h')
# https://stackoverflow.com/a/53339320
# me queda alguna duda de como funciona pero anda
bb['rest'] = bb['rest'].fillna(bb.groupby(
    'playerId')['rest'].transform('mean'))


bb.rest/np.timedelta64(1, 'h')
