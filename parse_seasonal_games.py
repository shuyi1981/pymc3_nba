import json
import pandas as pd
from pathlib import Path

dir = "D:\Data Science\MySportsFeed\python_api\data\\raw\seasonal_games"
pathlist = Path(dir).rglob("*.json")
# file = "D:\Data Science\MySportsFeed\python_api\data\\raw\seasonal_games\\2020-playoffs.json"

games_full = pd.DataFrame()

for file in pathlist:

    with open(file) as jsonfile:
        data = json.load(jsonfile)

    # inspect
    data.keys()
    data['references'].keys()

    # generate reference tables
    """
    Extract team and stadium information from json for each match
    """
    team_references = pd.DataFrame.from_dict(
        data['references']['teamReferences'])
    venue_references = pd.DataFrame.from_dict(
        data['references']['venueReferences'])

    # generate seasonal games table
    """ 
    Games list has two dictionaries. Schedule and score. We parse them individually and join them   into games tables 
    """
    schedule = []
    for row in data['games']:
        schedule.append(row['schedule'])
    schedule_df = pd.DataFrame(schedule)
    # extract information from column dictionaries
    schedule_df['awayTeamId'] = [d.get('id') for d in schedule_df.awayTeam]
    schedule_df['awayTeamAbb'] = [d.get('abbreviation')
                                  for d in schedule_df.awayTeam]
    schedule_df['homeTeamId'] = [d.get('id') for d in schedule_df.homeTeam]
    schedule_df['homeTeamAbb'] = [d.get('abbreviation')
                                  for d in schedule_df.homeTeam]
    schedule_df['venueId'] = [d.get('id') for d in schedule_df.venue]

    score = []
    for row in data['games']:
        score.append(row['score'])
    score_df = pd.DataFrame(score)

    for row, d in enumerate(score_df.quarters):
        for i in range(len(d)):
            score_df.loc[row, 'homeScore{n}'.format(n=i+1)] = d[i]['homeScore']
            score_df.loc[row, 'awayScore{n}'.format(n=i+1)] = d[i]['awayScore']

    games = schedule_df.join(score_df)

    games_full = games_full.append(games)

games_full.to_csv("data/parsed/all_games.csv")

# games.to_csv("data/parsed/playoffs_games.csv")
