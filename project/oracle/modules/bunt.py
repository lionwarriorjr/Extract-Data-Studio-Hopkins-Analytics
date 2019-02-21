import pandas as pd
import sys
from .module import Module
import numpy as np

class Bunt(Module):

    def set_module(self):
        return False

    def get_lexicon(self):
        result = set()
        result.add('bunt ratio')
        return result

    '''if self.is_filter return an index set else return a table'''
    def execute(self, iset, data, params={}, is_grouped=False):
        if is_grouped:
            return None
        df = data.iloc[iset,:]
        calc = pd.DataFrame([0.0])
        if df.shape[0] > 0:
            bunt = 0
            at_bat = 0
            at_bat_id = -1
            for index, row in df.iterrows():
                batter_id = row["batter"]
                if at_bat_id != row["at_bat_id"]:
                    if "bunt" in row["atbat_des"]:
                        bunt+=1
                    at_bat+=1
                    at_bat_id = row["at_bat_id"]
            calc = float(bunt)/float(at_bat)
            calc = 'YES' if calc > 0.10 else 'NO'
            calc = pd.DataFrame([calc])
        return calc