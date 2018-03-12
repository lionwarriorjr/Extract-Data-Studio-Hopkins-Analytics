class Module:
    
    '''Module interfaces with Core by generating index sets of filtered tables on relevant queries'''
    
    def __init__(self, params=None):
        self.params = params
        
    def set_module(self):
        """
        @return True if this Module is a filter module else False
        """
        raise NotImplementedError()    
        
    def get_lexicon(self):
        """
        @return keyword set for this Module
        """
        raise NotImplementedError()    
    
    def execute(self, iset, data, is_grouped=False):
        """
        Operates over shared reference to underlying database
            can parse self.qry to set self.params of this Module
            @return index set for filtered result
        """
        raise NotImplementedError()