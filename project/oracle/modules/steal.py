import numpy as np
from scipy import stats
import pandas as pd
from oracle.modules.module import Module
import collections

class Steal(Module):

    def set_module(self):
        return False
    
    def get_lexicon(self):
        result = set()
        result.add('stealing')
        result.add('steal ratio')
        return result
    
    '''if self.is_filter return an index set else return a table'''
    def execute(self, iset, data, params={}, is_grouped=False):
        if is_grouped:
            return None
        df = data.iloc[iset,:]
        calc = pd.DataFrame([0.0])
        if 'batter' in params:
            batter = params['batter']
        else:
            return calc
        if df.shape[0] > 0:
            swings = collections.Counter()
            batter_up = ""
            total = 0
            successful_steal = 0
            attempted_steal = 0
            for index, row in df.iterrows():
                if not pd.isnull(row['on_2b']):
                    continue
                # new bat
                if batter_up != row["at_bat_id"]:
                    # sucessfully stole a base
                    if batter + " steals" in row["atbat_des"]:
                        successful_steal+=1
                    # got caught stealing
                    if batter + " caught stealing" in row["atbat_des"]:
                        attempted_steal+=1
                    total+=1
                batter_up = row["at_bat_id"]
            # percentage of stealing or attempting to steal
            #perc_steal = (successful_steal+attempted_steal)/total
            total_steals = successful_steal + attempted_steal
            calc = 'YES' if total_steals > 20 else 'NO'
            calc = pd.DataFrame([calc])
        return calc