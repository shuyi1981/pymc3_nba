import requests
from requests.models import HTTPBasicAuth
import ast
from datetime import datetime, timedelta
import pandas as pd

# get credentials
file = open("cred.txt", "r")
contents = file.read()
cred = ast.literal_eval(contents)
file.close()

credentials = HTTPBasicAuth(cred['userid'], cred['pass'])

# hit API
FROM = "2020-01-01"
date = datetime.fromisoformat(FROM)
print(date)

# no permite consultas de + de 30 dias
# loop over dates
monthly_bets = []
while date < datetime.now():
    TO = date + timedelta(30)
    response = requests.get(f"https://api.pinnacle.com/v3/bets?betlist=SETTLED&fromDate={date.strftime('%Y-%m-%d')}T00:00:00Z&toDate={TO.strftime('%Y-%m-%d')}T00:00:00Z",
                            auth=credentials)
    monthly_bets.append(response)
    date = TO

# nos quedamos con la data
flat_list = [
    item
    for sublist in monthly_bets
    if 'straightBets' in sublist.json()
    for item in sublist.json()['straightBets']
]


df = pd.DataFrame(flat_list)
df.to_csv("../online_results/settled_bets.csv", index=False)
