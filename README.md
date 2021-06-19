# pymc3_nba
Using PYMC3 to model NBA matches

### Get data
* get_daily_player_gamelogs.py: data by player/game  req:api (train)
* get_seasonal_games.py: scheduled games for the season req:api 
* manually download odds from https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba/nbaoddsarchives.htm
----
* parse_daily_player_gamelogs: all player/games in one clean table (train)
* parse_seasonal_games.py: all games from the season
* parse_odds.py: write clean csv with odds
----
### Model and test
* shared_reg: feature engineering + bayesian model
* odds: calculate return on test data using odds.
----
### Predict live matches
* get_daily_lineup.py: expected lineup per game
* parse_game_lineup.py: write clean csv with lineups
* predict_one: predict one particular game
* predict_batch: predict all matches for selected dates


#### others
* benchmark: results from a model with one intercept (dummy) per player (not bayesian) 
* out_of_training.ipynb: notebook showing what happens if you include a player that wasn't seen in training

