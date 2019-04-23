import sys
import numpy as np
from scipy import stats
import pandas as pd
from oracle.modules.module import Module
import collections

class Swing(Module):

    def set_module(self):
        return False
    
    def get_lexicon(self):
        result = set()
        result.add('swing type')
        return result
    
    '''if self.is_filter return an index set else return a table'''
    def execute(self, iset, data, params={}, is_grouped=False):
        if is_grouped:
            return None
        df = data.iloc[iset,:]
        calc = pd.DataFrame([""])
        if df.shape[0] > 0:
            swings = collections.Counter()
            batter_up = ""
            swing = ""
            total = 0
            event = ""
            events = ["single","double","triple","field_out",
                      "double_play","force_out","home_run","grounded_into_double_play"]
            for index, row in df.iterrows():
                # new batter, we want to look at what happened at last pitch
                if batter_up != row["at_bat_id"]:
                    if event in events:
                        # increments that swing - could have multiple swings as declared below
                        # s is each of the swings that we declared (if there is just one)
                        # this for loop will only run once.
                        for s in swing:
                            swings[s]+=1
                            # increment total swings
                            total+=1
                    batter_up = row["at_bat_id"]
                # parse where to swing
                swing = ""
                for word in str(row["atbat_des"]).split(" "):
                    if word.lower() in ["left", "third"]:
                        # set a specific value for whatever kind of swing it was
                        swing = ["Left hitter"]
                        # now that we have found the word, we can exit out of this
                        # loop and stop looking for more words
                        break
                    elif word.lower() in ["pitcher", "center"]:
                        swing = ["Middle hitter"]
                        break
                    elif word.lower() in ["first", "right"]:
                        swing = ["Right hitter"]
                        break
                    elif word.lower() == "shortstop":
                        swing = ["Left hitter", "Middle hitter"]
                    elif word.lower() == "second":
                        swing = ["Middle hitter", "Right hitter"]
                event = row["event"]

            hitter = "spray hitter"
            if swings:
                most_common_swing = swings.most_common(1)[0]
                # check out thresholds
                if float(most_common_swing[1])/total > 0.4:
                    hitter = most_common_swing[0]
            calc = pd.DataFrame([hitter])
        
        return calc