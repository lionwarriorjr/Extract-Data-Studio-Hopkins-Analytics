import numpy as np
from scipy import stats
import pandas as pd
import json
import collections
from oracle.modules.steal import Steal
from oracle.modules.bunt import Bunt
from oracle.modules.swing import Swing
from oracle.modules.fpt import FPT
from oracle.modules.fps import FPS
from oracle.modules.power_sequence import PowerSequence
from oracle.modules.zones import ClassifyZones
from oracle.modules.module import Module

class Flashcard(Module):

    def set_module(self):
        return False
    
    def get_lexicon(self):
        result = set()
        result.add('flashcard')
        return result
    
    def run_flashcard(self, batting, appearances, batter):
        print('evaluating flashcard module')
        flashcard = {}
        flashcard['batter'] = batter
        stand = str(batting.loc[batting.index[0],'stand'])
        flashcard['stand'] = stand
        steal = Steal()
        perc_steal = steal.execute(range(len(appearances.index)), appearances, {'batter': batter}, False) 
        flashcard['perc_steal'] = perc_steal.iloc[0,0]
        bunt = Bunt()
        bunt_pct = bunt.execute(range(len(batting.index)), batting, {}, False)
        flashcard['bunt'] = bunt_pct.iloc[0,0]
        fps = FPS()
        fps_ratio = fps.execute(range(len(batting.index)), batting, {}, False)
        flashcard['fps'] = fps_ratio.iloc[0,0]
        fpt = FPT()
        fpt_ratio = fpt.execute(range(len(batting.index)), batting, {}, False)
        flashcard['fpt'] = fpt_ratio.iloc[0,0]
        swing = Swing()
        swing_pct = swing.execute(range(len(batting.index)), batting, {}, False)
        flashcard['swing_type'] = swing_pct.iloc[0,0]
        power = PowerSequence()
        power_sequence = power.execute(range(len(batting.index)), batting, {}, False)
        flashcard['powerSequence'] = power_sequence.iloc[0,0].strip()
        classify_zones = ClassifyZones()
        zones = classify_zones.execute(range(len(batting.index)), batting, {}, False)
        flashcard['zones'] = zones
        with open('flashcard.json', 'w') as outfile:
            json.dump(flashcard, outfile)
        print(flashcard)
        # call visualization subroutine (Simon and Randy's code)
    
    '''if self.is_filter return an index set else return a table'''
    def execute(self, iset, data, params={}, is_grouped=False):
        if is_grouped:
            return None
        df = data.iloc[iset,:]
        if data.shape[0] > 0:
            batter = params['batter']
            appearances = df
            df = df[df['batter_name'] == batter]
            if df.shape[0] > 0:
                batting_left = df[df['stand'] == 'L']
                batting_right = df[df['stand'] == 'R']
                if batting_left.shape[0] > 0 and batting_left.shape[0] > 0.2 * df.shape[0]:
                    self.run_flashcard(batting_left, appearances, batter)
                if batting_right.shape[0] > 0 and batting_right.shape[0] > 0.2 * df.shape[0]:
                    self.run_flashcard(batting_right, appearances, batter)