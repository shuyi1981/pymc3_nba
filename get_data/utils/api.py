from ohmysportsfeedspy import MySportsFeeds


def authenticate_api():
    msf = MySportsFeeds(version='2.1', store_type=None)
    msf.authenticate("baa67f75-9630-4666-942c-a70cd7", "MYSPORTSFEEDS")
    return msf


def get_data(version, league, season, feed, format, api, **kwargs):
    return api.msf_get_data(version=version, league=league, season=season, feed=feed, format=format, force=True, **kwargs)


def get_data_pbp(version, league, season, feed, format, api, **kwargs):
    api.msf_get_data(version=version, league=league, season=season,
                     feed=feed, format=format, force=True, game=game)
