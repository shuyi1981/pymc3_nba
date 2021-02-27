import csv
import pandas as pd
import re

file = "D:\Data Science\MySportsFeed\python_api\\results\seasonal_games-nba-2019-2020-regular.csv"



with open(file, newline='') as csvfile:
   spamreader = csv.reader(csvfile, delimiter=',')
   header = next(spamreader)[0]
   print(header)
   df = pd.DataFrame(columns = list(header.split(',')))
   if header != None:
       for row in spamreader:
           try:
               temp = row[0].replace('[', '').replace(']','').split(',')
               df.loc[len(df)] = temp
           except:
                pass
            
        
df.head()
       
