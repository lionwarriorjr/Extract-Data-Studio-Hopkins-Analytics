import pandas as pd
import sys
from .module import Module
import numpy as np

class FPS(Module):

    def set_module(self):
        return False

    def get_lexicon(self):
        result = set()
        result.add('FPS')
        result.add('first pitch swing ratio')
        return result

    '''if self.is_filter return an index set else return a table'''
    def execute(self, iset, data, params={}, is_grouped=False):
        if is_grouped:
            return None
        data = data.iloc[iset,:]
        calc = pd.DataFrame([0.0])
        if data.shape[0] > 0:
            no_swing = ['ball', 'called_strike']
            take = 0
            swing = 0
            at_bat_id = -1
            for index, row in data.iterrows():
                # first pitch
                if at_bat_id != row["at_bat_id"]:
                    if row["description"] in no_swing:
                        take+=1
                    else:
                        swing+=1
                    at_bat_id = row["at_bat_id"]
            calc = float(swing)/float(take + swing)
            calc = 'YES' if calc > 0.33 else 'NO'
            calc = pd.DataFrame([calc])
        return calc