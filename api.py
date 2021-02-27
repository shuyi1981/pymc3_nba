from ohmysportsfeedspy import MySportsFeeds

def authenticate_api():
    msf = MySportsFeeds(version='2.1', store_type = None)
    msf.authenticate("4eec6849-46c9-43d2-9280-711e0c", "MYSPORTSFEEDS")
    return msf

def get_data(version,league, season, feed, format, api, **kwargs):
    return api.msf_get_data(version=version,league=league,season=season,feed=feed,format=format, force = True, **kwargs )
    
    
def get_data_pbp(version,league, season, feed, format, api, **kwargs):
    api.msf_get_data(version=version,league=league,season=season,feed=feed,format=format, force = True, game = game )



