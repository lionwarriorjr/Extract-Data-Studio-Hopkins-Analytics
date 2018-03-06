import os
import sys
import math
import numpy as np
from scipy import stats
import pandas as pd
import nltk
from collections import defaultdict
import re
import itertools
import jellyfish
from functools import reduce
import plotly

plotly.tools.set_credentials_file(username='smohan', api_key='UGKT8GURMTb64LlCxpBm') # Plotly API Credentials

desc_filename = os.path.join(os.getcwd(), 'sample_query.csv')
X = pd.read_csv(desc_filename) # read season data and store in config.X
desc_filename = os.path.join(os.getcwd(), 'sample_query_pitch.csv')
pitch = pd.read_csv(desc_filename) # read season data for pitch table and store in config.pitch
desc_filename = os.path.join(os.getcwd(), 'sample_query_atbat.csv')
atbat = pd.read_csv(desc_filename) # read season data for atbat table and store in config.atbat
desc_filename = os.path.join(os.getcwd(), 'sample_query_player.csv')
player = pd.read_csv(desc_filename) # read season data for player table and store in config.player
base = 'X'
domains = {'pitch': pitch, 'atbat': atbat, 'player': player, 'X': X}

# domain specific feature coercing
X['inning'] = X['inning'].astype(str)
atbat['inning'] = atbat['inning'].astype(str)
pitch['inning'] = pitch['inning'].astype(str)
X['o'] = X['o'].astype(str)
atbat['o'] = atbat['o'].astype(str)
X['Date'] = pd.to_datetime(X['Date'])
atbat['Date'] = pd.to_datetime(atbat['Date'])
months = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
          7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
X['month'] = [months[t.month] for t in X.Date]
atbat['month'] = [months[t.month] for t in atbat.Date]

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
    
    def execute(self, iset, name, is_grouped=False):
        """
        Operates over shared reference to underlying database
            can parse self.qry to set self.params of this Module
            @return index set for filtered result
        """
        raise NotImplementedError()    
    
module_keywords = {}
#module_parsers = pybktree.BKTree(pybktree.hamming_distance, [])
module_hash = defaultdict(list)
modules_reversed = defaultdict(set)

# initialize keyword set
keywords = defaultdict(str)
keywords['and'] = 'AND'
keywords['or'] = 'OR'
keywords['not'] = 'NOT'
keywords['of'] = '->'
keywords['that'] = '->'
keywords['in'] = '=>*(%filter%)'
keywords['on'] = '=>*(%filter%)'
keywords['for'] = '=>*(%filter%)'
keywords['from'] = '=>*(%filter%)'
keywords['when'] = '=>*(%filter%)'
keywords['where'] = '=>*(%filter%)'
keywords['with'] = '=>*(%filter%)'
keywords['against'] = '=>*(%filter%)'
keywords['by'] = '=>*(%by%)'
keywords['along'] = '=>*(%by%)'
keywords['over'] = '=>*(%over%)'
keywords['under'] = '=>*(%under%)'
keywords['above'] = '=>*(%over%)'
keywords['below'] = '=>*(%under%)'
keywords['between'] = '=>*(%between%)'
keywords['except'] = '=>*(%except%)'
keywords['without'] = '=>*(%except%)'
keywords['near'] = '=>*(%near%)'
keywords['through'] = '=>*(%range%)'
keywords['until'] = '=>*(%until%)'
keywords['to'] = '=>*(%to%)'
keywords['after'] = '=>*(%after%)'
keywords['before'] = '=>*(%before%)'
keywords['compare'] = '=>*(%compare%)'
numericFilters = ['above', 'below', 'over', 'under', 'between']
FILTER = 'filter'

filters = set()
GENITIVE = '->'

# initialize stoplist set
stoplist = {}
stoplist['V'] = set()
stoplist['V'].add('is')
stoplist['V'].add('are')
stoplist['V'].add('am')
stoplist['V'].add('be')
stoplist['V'].add('was')
stoplist['V'].add('were')
stoplist['V'].add('do')
stoplist['V'].add('does')
stoplist['V'].add('did')

def is_actor(tag):
    '''Return part of speech tags that map to noun phrases in the query'''
    return tag in ['NN', 'NNP', 'NNS', 'PRP', 'CD']

def is_desc(tag):
    '''Return part of speech tags that map to modifiers in the query'''
    return tag in ['CD', 'JJ', 'JJR', 'JJS', 'RBR', 'RBS', 'VBG']

def is_prep(tag):
    '''Return part of speech tags that map to prepositions in the query'''
    return tag in ['IN', 'WRB']

def is_verb(tag):    
    '''Return part of speech tags that map to verb phrases in the query'''
    return re.match(r'VB.*', tag[1]) is not None and tag[0] not in stoplist['V']

def is_genitive(tag):
    '''Return part of speech tags that map to possessive modifiers in the query'''
    return tag[0] == '->' or tag[1] in ['POS', 'PRP$']

def is_adv(tag):
    '''Return part of speech tags that map to adverbs in the query'''
    return tag == 'RB'

def is_gerund(tag):
    '''Return part of speech tags that map to gerunds in the query'''
    return tag == 'VBG'

def is_conj(tag):
    '''Return part of speech tags that map to conjunctions in the query'''
    return tag in ['and', 'or', 'not']

def initialize():
    # initialize filters set to contain all keywords that are not conjunctions
    for f_key in keywords:
        if keywords[f_key] != '->' and not is_conj(f_key):
            filters.add(keywords[f_key])
    filtered = domains[base].copy()
    return filtered

filtered = initialize() # filtered updated automatically on user query
joinedTables = set() # current tables being joined in the query
time_series = filtered

qry = '' # store current user query as config.qry
entityFilters = []
conjunctive = {}

module_tables = {'filtered': filtered, 'time_series': time_series}

# Initialize preset substitution list
subs = defaultdict(str)
subs['(time)->(percentage)'] = '(*PCT*)'
subs['(time)->(percent)'] = '(*PCT*)'

# add to domain-specific background information to the system's domain knowledge
# account for common modifiers that reference features in the dataset

tables = {}
tables['month'] = 'atbat'
tables['team_id_pitcher'] = 'atbat'
tables['team_id_batter'] = 'atbat'
tables['type'] = 'pitch'
tables['pitch_type'] = 'pitch'
tables['event'] = 'pitch'
tables['stand'] = 'atbat'
tables['inning'] = 'atbat'
tables['inning_side'] = 'pitch'
tables['o'] = 'atbat'
tables['px'] = 'pitch'
tables['pz'] = 'pitch'
tables['start_speed'] = 'pitch'
tables['end_speed'] = 'pitch'
tables['pfx_x'] = 'pitch'
tables['pfx_z'] = 'pitch'
tables['ax'] = 'pitch'
tables['az'] = 'pitch'
tables['break_angle'] = 'pitch'
tables['break_length'] = 'pitch'
tables['type_confidence'] = 'pitch'
tables['batter'] = 'atbat'
tables['pitcher'] = 'pitch'
tables['batter_name'] = 'atbat'
tables['pitcher_name'] = 'pitch'

DOMAIN_KNOWLEDGE = defaultdict(lambda : defaultdict(list))
DOMAIN_KNOWLEDGE['month']['month'] = list(X['month'].unique())
DOMAIN_KNOWLEDGE['team_id_pitcher']['astros'] = ['houmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['rangers'] = ['texmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['angels'] = ['anamlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['diamondbacks'] = ['arimlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['white sox'] = ['chamlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['reds'] = ['cinmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['rockies'] = ['colmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['tigers'] = ['detmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['brewers'] = ['milmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['yankees'] = ['nyamlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['athletics'] = ['oakmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['phillies'] = ['phimlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['pirates'] = ['pitmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['mariners'] = ['seamlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['rays'] = ['tbamlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['blue jays'] = ['tormlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['nationals'] = ['wasmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['cubs'] = ['chnmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['dodgers'] = ['lanmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['marlins'] = ['miamlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['padres'] = ['sdnmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['orioles'] = ['balmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['royals'] = ['kcamlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['twins'] = ['minmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['mets'] = ['nynmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['cardinals'] = ['slnmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['giants'] = ['sfnmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['texas'] = ['texmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['los angeles'] = ['anamlb', 'lanmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['arizona'] = ['arimlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['chicago'] = ['chamlb','chnmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['cincinatti'] = ['cinmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['colorado'] = ['colmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['detroit'] = ['detmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['milwaukee'] = ['milmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['new york'] = ['nyamlb', 'nynmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['oakland'] = ['oakmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['philadelphia'] = ['phimlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['pittsburgh'] = ['pitmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['seattle'] = ['seamlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['tampa bay'] = ['tbamlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['toronto'] = ['tormlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['washington'] = ['wasmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['miami'] = ['miamlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['san diego'] = ['sdnmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['baltimore'] = ['balmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['kansas city'] = ['kcamlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['minnesota'] = ['minmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['st louis'] = ['stlmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['san francisco'] = ['sfnmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['pitchers'] = list(set(reduce(lambda x, y: x + y, 
                                                           DOMAIN_KNOWLEDGE['team_id_pitcher'].values(), [])))
DOMAIN_KNOWLEDGE['team_id_batter']['batters'] = DOMAIN_KNOWLEDGE['team_id_pitcher']['pitchers']
DOMAIN_KNOWLEDGE['type']['strike'].append('S')
DOMAIN_KNOWLEDGE['type']['ball'].append('B')
DOMAIN_KNOWLEDGE['type']['in-play'].append('X')
DOMAIN_KNOWLEDGE['pitch_type']['pitch type'] = list(X['pitch_type'].unique())
DOMAIN_KNOWLEDGE['pitch_type']['fastball'] = ['FA', 'FF', 'FT', 'FC', 'FS']
DOMAIN_KNOWLEDGE['pitch_type']['4-seam'] = ['FF']
DOMAIN_KNOWLEDGE['pitch_type']['four-seam'] = ['FF']
DOMAIN_KNOWLEDGE['pitch_type']['two-seam'] = ['FT']
DOMAIN_KNOWLEDGE['pitch_type']['2-seam'] = ['FT']
DOMAIN_KNOWLEDGE['pitch_type']['cutter'] = ['FC']
DOMAIN_KNOWLEDGE['pitch_type']['sinker'] = ['SI']
DOMAIN_KNOWLEDGE['pitch_type']['split'] = ['SF']
DOMAIN_KNOWLEDGE['pitch_type']['fingered'] = ['SF']
DOMAIN_KNOWLEDGE['pitch_type']['slider'] = ['SL']
DOMAIN_KNOWLEDGE['pitch_type']['changeup'] = ['CH']
DOMAIN_KNOWLEDGE['pitch_type']['curveball'] = ['CB', 'CU']
DOMAIN_KNOWLEDGE['pitch_type']['knuckleball'] = ['KC', 'KN']
DOMAIN_KNOWLEDGE['pitch_type']['knucklers'] = ['KC', 'KN']
DOMAIN_KNOWLEDGE['pitch_type']['eephus'] = ['EP']
DOMAIN_KNOWLEDGE['event']['groundout'] = ['Groundout']
DOMAIN_KNOWLEDGE['event']['groundball'] = ['Groundout']
DOMAIN_KNOWLEDGE['event']['strikeout'] = ['Strikeout']
DOMAIN_KNOWLEDGE['event']['homerun'] = ['Home Run']
DOMAIN_KNOWLEDGE['event']['walk'] = ['Walk']
DOMAIN_KNOWLEDGE['event']['single'] = ['Single']
DOMAIN_KNOWLEDGE['event']['double'] = ['Double']
DOMAIN_KNOWLEDGE['event']['triple'] = ['Triple']
DOMAIN_KNOWLEDGE['event']['lineout'] = ['Lineout']
DOMAIN_KNOWLEDGE['event']['flyout'] = ['Flyout']
DOMAIN_KNOWLEDGE['event']['flyball'] = ['Flyout']
DOMAIN_KNOWLEDGE['event']['pop-out'] = ['Pop Out']
DOMAIN_KNOWLEDGE['event']['bunt'] = ['Bunt Groundout', 'Sac Bunt', 'Bunt Pop Out']
DOMAIN_KNOWLEDGE['event']['field'] = ['Field Error']
DOMAIN_KNOWLEDGE['event']['error'] = ['Field Error']
DOMAIN_KNOWLEDGE['stand']['batters'] = ['L', 'R']
DOMAIN_KNOWLEDGE['stand']['lefty'] = ['L']
DOMAIN_KNOWLEDGE['stand']['left-handed'] = ['L']
DOMAIN_KNOWLEDGE['stand']['righty'] = ['R']
DOMAIN_KNOWLEDGE['stand']['right-handed'] = ['R']
DOMAIN_KNOWLEDGE['inning']['inning'] = list(X['inning'].unique())
DOMAIN_KNOWLEDGE['inning']['first'] = ['1']
DOMAIN_KNOWLEDGE['inning']['second'] = ['2']
DOMAIN_KNOWLEDGE['inning']['third'] = ['3']
DOMAIN_KNOWLEDGE['inning']['fourth'] = ['4']
DOMAIN_KNOWLEDGE['inning']['fifth'] = ['5']
DOMAIN_KNOWLEDGE['inning']['sixth'] = ['6']
DOMAIN_KNOWLEDGE['inning']['seventh'] = ['7']
DOMAIN_KNOWLEDGE['inning']['eighth'] = ['8']
DOMAIN_KNOWLEDGE['inning']['ninth'] = ['9']
DOMAIN_KNOWLEDGE['inning_side']['top'] = ['top']
DOMAIN_KNOWLEDGE['inning_side']['bottom'] = ['bottom']
DOMAIN_KNOWLEDGE['o']['outs'] = ['0', '1', '2']
DOMAIN_KNOWLEDGE['is_on1b']['runner'] = ['!NA']
DOMAIN_KNOWLEDGE['is_on2b']['runner'] = ['!NA']
DOMAIN_KNOWLEDGE['is_on3b']['runner'] = ['!NA']
DOMAIN_KNOWLEDGE['px']['x'] = ['is.numeric']
DOMAIN_KNOWLEDGE['px']['location'] = ['is.numeric']
DOMAIN_KNOWLEDGE['px']['left/right'] = ['is.numeric']
DOMAIN_KNOWLEDGE['px']['horizontal'] = ['is.numeric']
DOMAIN_KNOWLEDGE['pz']['z'] = ['is.numeric']
DOMAIN_KNOWLEDGE['pz']['location'] = ['is.numeric']
DOMAIN_KNOWLEDGE['pz']['pitch'] = ['is.numeric']
DOMAIN_KNOWLEDGE['pz']['height'] = ['is.numeric']
DOMAIN_KNOWLEDGE['start_speed']['start'] = ['is.numeric']
DOMAIN_KNOWLEDGE['start_speed']['pitch'] = ['is.numeric']
DOMAIN_KNOWLEDGE['start_speed']['ball'] = ['is.numeric']
DOMAIN_KNOWLEDGE['start_speed']['speed'] = ['is.numeric']
DOMAIN_KNOWLEDGE['start_speed']['velocity'] = ['is.numeric']
DOMAIN_KNOWLEDGE['end_speed']['end'] = ['is.numeric']
DOMAIN_KNOWLEDGE['end_speed']['pitch'] = ['is.numeric']
DOMAIN_KNOWLEDGE['end_speed']['ball'] = ['is.numeric']
DOMAIN_KNOWLEDGE['end_speed']['speed'] = ['is.numeric']
DOMAIN_KNOWLEDGE['end_speed']['velocity'] = ['is.numeric']
DOMAIN_KNOWLEDGE['pfx_x']['x'] = ['is.numeric']
DOMAIN_KNOWLEDGE['pfx_x']['movement'] = ['is.numeric']
DOMAIN_KNOWLEDGE['pfx_x']['left/right'] = ['is.numeric']
DOMAIN_KNOWLEDGE['pfx_x']['horizontal'] = ['is.numeric']
DOMAIN_KNOWLEDGE['pfx_x']['z'] = ['is.numeric']
DOMAIN_KNOWLEDGE['pfx_z']['movement'] = ['is.numeric']
DOMAIN_KNOWLEDGE['pfx_z']['vertical'] = ['is.numeric']
DOMAIN_KNOWLEDGE['ax']['x'] = ['is.numeric']
DOMAIN_KNOWLEDGE['ax']['acceleration'] = ['is.numeric']
DOMAIN_KNOWLEDGE['ax']['left/right'] = ['is.numeric']
DOMAIN_KNOWLEDGE['ax']['horizontal'] = ['is.numeric']
DOMAIN_KNOWLEDGE['az']['z'] = ['is.numeric']
DOMAIN_KNOWLEDGE['az']['acceleration'] = ['is.numeric']
DOMAIN_KNOWLEDGE['az']['vertical'] = ['is.numeric']
DOMAIN_KNOWLEDGE['break_angle']['break'] = ['is.numeric']
DOMAIN_KNOWLEDGE['break_angle']['angle'] = ['is.numeric']
DOMAIN_KNOWLEDGE['break_length']['break'] = ['is.numeric']
DOMAIN_KNOWLEDGE['break_length']['length'] = ['is.numeric']
DOMAIN_KNOWLEDGE['type_confidence']['type'] = ['is.numeric']
DOMAIN_KNOWLEDGE['type_confidence']['confidence'] = ['is.numeric']
DOMAIN_KNOWLEDGE['o']['outs'] = ['is.numeric']

name_ids = ['batter_name', 'pitcher_name']
IDENTIFIERS = defaultdict(set)
for name_id in name_ids:
    IDENTIFIERS[name_id] = set(atbat[name_id])
for name_id in name_ids:
    IDENTIFIERS[name_id] = {x for x in IDENTIFIERS[name_id] if x==x}
DOMAIN_KNOWLEDGE[name_ids[0]]['batter name'] = list(atbat[name_ids[0]].unique())
DOMAIN_KNOWLEDGE[name_ids[1]]['pitcher name'] = list(atbat[name_ids[1]].unique())
DOMAIN_KNOWLEDGE[name_ids[0]]['batter'] = list(atbat[name_ids[0]].unique())
DOMAIN_KNOWLEDGE[name_ids[1]]['pitcher'] = list(atbat[name_ids[1]].unique())

CORE = { # CORE initialized to hold uninformative modifiers for each observation in the dataset
    'pitch': None,
    'pitches': None,
    'atbat': None,
    'atbats': None
}

FEATURE_DIST = {'pitch type': 'pitch_type', 'inning': 'inning', 'month': 'month', 'team': 'team',
               'batter': name_ids[0], 'pitcher': name_ids[1], 'batter name': name_ids[0], 'pitcher name': name_ids[1]}

labels = [list(DOMAIN_KNOWLEDGE[key].keys()) for key in DOMAIN_KNOWLEDGE]
labels = list(itertools.chain.from_iterable(labels))
labels.extend([str(key) for key in CORE])

def train(features):
    model = defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model

NWORDS = train(labels)
alphabet = 'abcdefghijklmnopqrstuvwxyz'

def edits1(word):
    s = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes    = [a + b[1:] for a, b in s if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in s if len(b)>1]
    replaces   = [a + c + b[1:] for a, b in s for c in alphabet if b]
    inserts    = [a + c + b     for a, b in s for c in alphabet]
    return set(deletes + transposes + replaces + inserts)

def known_edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)

def known(words):
    return set(w for w in words if w in NWORDS)

def correct(word):
    candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
    return max(candidates, key=NWORDS.get)

class Recommendation:
    
    '''Recommendation Object that Holds Query Metadata'''
    
    def __init__(self, qry = None):
        self.data = {}
        if qry is not None:
            self.data[qry] = RecommendationItem()
        
class RecommendationItem:
    
    '''Recommendation Item for Automated Query Suggestion'''
    
    def __init__(self, feature = None, index = []):
        self.index_hash = {}
        if feature is not None:
            self.index_hash[feature] = index

added = [] # list of features added to dataframe during analysis
COOCCURENCE_HASH = defaultdict(lambda : defaultdict(float)) # cooccurence hash over features
MOST_RECENT_QUERY = set() # holds set of features in most recent query
EXP_DECAY = defaultdict(int) # hash of each feature on exponentially decaying weight as distance since last fetched increases
ITEMSETS = [] # frequent itemsets
RECOMMENDATION = Recommendation() # global RECOMMENDATION object
RECOMMENDATIONS = defaultdict(list)

CONF_THRESHOLD = 0.9 # confidence that a token must match an entry in DOMAIN_KNOWLEDGE before fetching that feature directly
MODULE_PARSING_THRESHOLD = 0.95 # confidence that token matches module in query apriori
NAME_THRESHOLD = 0.9 # confidence that a token must match an entry in IDENTIFIERS before being labeled an player id
RELEVANCE_FEEDBACK_THRESHOLD = 0.5 # threshold normalized on [0, 1] to label if a feature is relevant after relevance feedback
ASSOC_MIN_SUPPORT = 2 # frequent itemsets support threshold
ASSOC_MIN_CONFIDENCE = 0.5 # frequent itemsets confidence threshold
DECAY = 2e-4 # exponential decay cost

clookup = defaultdict(set)

def update_module_parsers(update_list, module):
    for key in update_list:
        tokens = key.split()
        for token in tokens:
            #module_parsers.add(token)
            module_hash[token].append(module)

def update_modules_reversed(update_list, module):
    for e in update_list:
        modules_reversed[module].add(e)

###############################################################################################################################
########################################################## MODULES ############################################################
class StrikeRate(Module):
    
    def set_module(self):
        return False
    
    def get_lexicon(self):
        result = set()
        result.add('strike rate')
        return result
    
    '''if self.is_filter return an index set else return a table'''
    def execute(self, iset, name, is_grouped):
        group = module_tables[name]
        if is_grouped:
            group = group.apply(lambda g: g).reset_index(drop=True)
        group = group.iloc[iset,:]
        calc = pd.DataFrame([0.0])
        if group.shape[0] > 0:
            calc = group[(group['type'] == 'S') | (group['type'] == 'X')].shape[0] / group.shape[0]
            calc = pd.DataFrame([calc])
        return calc

module_keywords['strike rate'] = StrikeRate
update_module_parsers(['strike rate'], StrikeRate)
update_modules_reversed(['strike rate'], StrikeRate)

class OBP(Module):
    
    def set_module(self):
        return False
    
    def get_lexicon(self):
        result = set()
        result.add('OBP')
        result.add('on-base percentage')
        return result
    
    '''if self.is_filter return an index set else return a table'''
    def execute(self, iset, name, is_grouped):
        group = module_tables[name]
        group = group.iloc[iset,:]
        calc = group[((group['event'] == 'Single') | (group['event'] == 'Double') | (group['event'] == 'Triple') | 
                      (group['event'] == 'Home Run') | (group['event'] == 'Walk') | 
                      (group['event'] == 'Field Error'))].shape[0] / group.shape[0]
        calc = pd.DataFrame([calc])
        return calc
    
module_keywords['OBP'] = OBP
module_keywords['on-base percentage'] = OBP
update_module_parsers(['OBP', 'on-base percentage'], OBP)
update_modules_reversed(['OBP', 'on-base percentage'], OBP)

class LeadRunnerOnThird(Module):
    
    def set_module(self):
        return True
    
    def get_lexicon(self):
        result = set()
        result.add('lead-runner-on-third')
        result.add('lead runner on third')
        result.add('lead runner on third base')
        return result
    
    '''if self.is_filter return an index set else return a table'''
    def execute(self, iset, name, is_grouped):
        group = module_tables[name]
        group = group.iloc[iset,:]
        rset = []
        for index, row in group.iterrows():
            if not math.isnan(row['on_3b']):
                rset.append(index)
        return rset
                           
module_keywords['lead-runner-on-third'] = LeadRunnerOnThird
module_keywords['lead runner on third'] = LeadRunnerOnThird
module_keywords['lead runner on third base'] = LeadRunnerOnThird
update_module_parsers(['lead-runner-on-third', 'lead runner on third', 
                       'lead runner on third base'], LeadRunnerOnThird)
update_modules_reversed(['lead-runner-on-third', 'lead runner on third', 
                         'lead runner on third base'], LeadRunnerOnThird)

class LeadRunnerOnSecond(Module):
    
    def set_module(self):
        return True
    
    def get_lexicon(self):
        result = set()
        result.add('lead-runner-on-second')
        return result
    
    '''if self.is_filter return an index set else return a table'''
    def execute(self, iset, name, is_grouped):
        group = module_tables[name]
        group = group.iloc[iset,:]
        rset = []
        for index, row in group.iterrows():
            if not math.isnan(row['on_2b']) and math.isnan(row['on_3b']):
                rset.append(index)
        return rset
                           
module_keywords['lead-runner-on-second'] = LeadRunnerOnSecond
update_module_parsers(['lead-runner-on-second'], LeadRunnerOnSecond)
update_modules_reversed(['lead-runner-on-second'], LeadRunnerOnSecond)

class LeadRunnerOnFirst(Module):
    
    def set_module(self):
        return True
    
    def get_lexicon(self):
        result = set()
        result.add('lead-runner-on-first')
        return result
    
    '''if self.is_filter return an index set else return a table'''
    def execute(self, iset, name, is_grouped):
        group = module_tables[name]
        group = group.iloc[iset,:]
        rset = []
        for index, row in group.iterrows():
            if not math.isnan(row['on_1b']) and math.isnan(row['on_2b']) and math.isnan(row['on_3b']):
                rset.append(index)
        return rset
                           
module_keywords['lead-runner-on-first'] = LeadRunnerOnFirst
update_module_parsers(['lead-runner-on-first'], LeadRunnerOnFirst)
update_modules_reversed(['lead-runner-on-first'], LeadRunnerOnFirst)