import pandas as pd
import numpy as np
from pathlib import Path

dir = "D:\Data Science\MySportsFeed\python_api\data\\raw\odds"
pathlist = Path(dir).rglob("*.xlsx")

harcode_index_year_2019 = 1005
abbs = pd.read_csv("data/parsed/abbreviations.csv")


def american_to_decimal(american):
    if american >= 0:
        decimal = 1 + american/100
    else:
        decimal = 1 - (100 / american)
    return decimal


all_odds = pd.DataFrame()
all_odds_reduced = pd.DataFrame()

for file in pathlist:

    first_year = file.__str__()[-12:-8]
    second_year = '20' + file.__str__()[-7:-5]

    odds = pd.read_excel(
        file, sheet_name="Sheet1")

    odds['decimal'] = odds['ML'].apply(american_to_decimal)

    odds['day'] = odds['Date'].astype(str).str[-2:]
    odds['month'] = odds['Date'].astype(str).str[:-2]
    odds['month'] = odds['month'].str.zfill(2)

    if first_year == "2019":
        odds['year'] = np.where(
            odds.index <= harcode_index_year_2019, 2019, 2020)
    else:
        odds['year'] = np.where(odds.month.astype(
            int) > 9, first_year, second_year)

    odds['newDate'] = odds['year'].astype(
        str) + "-" + odds['month'].astype(str) + "-" + odds['day'].astype(str)

    odds = odds.merge(abbs, on="Team", how="left")
    odds.head()

    odds_reduced = odds.loc[:, ['newDate', 'Abb', 'decimal']]

    all_odds = all_odds.append(odds)
    all_odds_reduced = all_odds_reduced.append(odds_reduced)


# export
all_odds.to_csv("data/parsed/all_odds.csv", index=False)
all_odds_reduced.to_csv("data/parsed/all_odds_reduced.csv", index=False)
