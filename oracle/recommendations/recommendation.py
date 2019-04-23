class Recommendation:
    '''Recommendation Object that Holds Query Metadata'''

    def __init__(self, qry=None):
        self.data = {}
        if qry is not None:
            self.data[qry] = RecommendationItem()


class RecommendationItem:
    '''Recommendation Item for Automated Query Suggestion'''

    def __init__(self, feature=None, index=[]):
        self.index_hash = {}
        if feature is not None:
            self.index_hash[feature] = index
