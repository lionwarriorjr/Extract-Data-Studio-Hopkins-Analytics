import pandas as pd
import sys
import collections
from collections import defaultdict
import numpy as np
from oracle.modules.module import Module

class PowerSequence(Module):

    def set_module(self):
        return False
    
    def get_lexicon(self):
        result = set()
        result.add('power sequence')
        return result
    
    '''if self.is_filter return an index set else return a table'''
    def execute(self, iset, data, params={}, is_grouped=False):
        if is_grouped:
            return None
        df = data.iloc[iset,:]
        calc = pd.DataFrame([""])
        if df.shape[0] > 0:
            pitches = collections.Counter()
            sequence_ids = defaultdict(list)
            batter_up = -1
            event = ""
            second_last_at_bat_pitch_type = ""
            last_at_bat_pitch_type = ""
            second_last_at_bat_id = ""
            last_at_bat_id = ""
            for index, row in df.iterrows():
                # new batter, we want to look at what happened at last pitch
                if batter_up != row["at_bat_id"]:
                    # we only care if this guy struck out
                    if event == "Strikeout":
                        power_sequence = str(second_last_at_bat_pitch_type) + " " + str(last_at_bat_pitch_type)
                        # if we've seen this sequence before, we incrememnt by 1.
                        # if we haven't, default is 0
                        pitches[power_sequence]+=1
                        sequence_ids[power_sequence].append(str(second_last_at_bat_id) + " " + str(last_at_bat_id))
                    second_last_at_bat_pitch_type = ""
                    last_at_bat_pitch_type = ""
                    second_last_at_bat_id = ""
                    last_at_bat_id = ""
                    event = ""
                    batter_up = row["at_bat_id"]
                else:
                    # we want last two at bats at all times in case they are the last pitch
                    # puts them into zone based on four squares - zone 1-4.
                    # possbile TODO: account for right/left hitters
                    x = row["px"]
                    z = row["pz"]
                    horiz, vert = 1, 1
                    if x < -1.25 or x > 1.25:
                        horiz = 3
                    if -1.25 < x < 0:
                        horiz = 1
                    if 0 < x < 1.25:
                        horiz = 2
                    if z<0.75 or z>3.58:
                        vert = 3
                    if 0.75 < z < 2.176:
                        vert = 1
                    if 2.17 < z < 3.58:
                        vert = 2
                    if (horiz,vert) == (1,1):
                        zone = 'Zone 3'
                    if (horiz,vert) == (2,1):
                        zone = 'Zone 4'
                    if (horiz,vert) == (1,2):
                        zone = 'Zone 2'
                    if (horiz,vert) == (2,2):
                        zone = 'Zone 1'
                    if horiz == 3 or vert == 3:
                        zone = "ball"
                    if zone != 'ball':
                        second_last_at_bat_pitch_type = last_at_bat_pitch_type
                        last_at_bat_pitch_type = str(row["pitch_type"]) + " " + str(zone)
                        second_last_at_bat_id = last_at_bat_id
                        last_at_bat_id = index
                        event = row["event"]
            # best power sequence
            if pitches:
                power_sequence = str(pitches.most_common(1)[0][0])
                #print('best_power_sequence:')
                #print(sequence_ids[power_sequence])
                calc = pd.DataFrame([power_sequence])
        
        return calc