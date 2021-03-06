class Error(Exception):
    """Base class for other exceptions"""
    pass

class InvalidTweet(Error):
    """Raised when the tweet is invalid by not having the defined keys by the official Twitter API"""
    pass

class NotEnglishTweet(Error):
    "Raised when the tweet is not in English ('lang' parameter of JSON is not 'en')"
    pass