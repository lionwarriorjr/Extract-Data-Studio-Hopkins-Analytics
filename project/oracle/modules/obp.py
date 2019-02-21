import numpy as np
from scipy import stats
import pandas as pd
from oracle.modules.module import Module

class OBP(Module):
    
    def set_module(self):
        return False
    
    def get_lexicon(self):
        result = set()
        result.add('OBP')
        result.add('on-base percentage')
        return result
    
    '''if self.is_filter return an index set else return a table'''
    def execute(self, iset, data, params={}, is_grouped=False):
        if is_grouped:
            data = data.apply(lambda g: g).reset_index(drop=True)
        data = data.iloc[iset,:]
        calc = pd.DataFrame([0.0])
        if data.shape[0] > 0:
            calc = data[((data['event'] == 'Single') | (data['event'] == 'Double') | (data['event'] == 'Triple') | 
                         (data['event'] == 'Home Run') | (data['event'] == 'Walk') | 
                         (data['event'] == 'Field Error'))].shape[0] / data.shape[0]
            calc = pd.DataFrame([calc])
        return calc