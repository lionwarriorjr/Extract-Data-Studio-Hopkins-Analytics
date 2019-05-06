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
from oracle.modules.module import Module
from oracle.modules.strikerate import StrikeRate
from oracle.modules.obp import OBP
from oracle.modules.leadrunnerthird import LeadRunnerOnThird
from oracle.modules.leadrunnersecond import LeadRunnerOnSecond
from oracle.modules.leadrunnerfirst import LeadRunnerOnFirst
from oracle.modules.steal import Steal
from oracle.modules.bunt import Bunt
from oracle.modules.swing import Swing
from oracle.modules.fpt import FPT
from oracle.modules.fps import FPS
from oracle.modules.power_sequence import PowerSequence
from oracle.modules.zones import ClassifyZones
from oracle.modules.flashcard import Flashcard
from oracle.modules.pitchers_like_me import PitchersLikeMe
from oracle.recommendations.recommendation import Recommendation

class Config:

    def __init__(self, desc_filename):

        self.X = pd.read_csv(desc_filename, encoding='latin1', low_memory=False)  # store initialized table at start
        self.init_default_preprocessing()
        # for simplicity, only consider one table for now (call set_datasets to include more tables)
        self.relations = {}
        self.base = 'X'
        self.domains = {'X': self.X}
        self.domains.update(self.relations)
        self.tables = {}

        self.module_keywords = {}
        self.module_hash = defaultdict(list)
        self.modules_reversed = defaultdict(set)
        self.keywords = self.set_keywords()
        self.numericFilters = self.set_numeric_filters()
        self.FILTER, self.GENITIVE = 'filter', '->'
        self.filters = set()
        self.stoplist = self.set_stoplist()

        self.filtered = self.initialize()  # store current filtered table
        self.time_series = self.filtered
        self.joinedTables = set()  # current tables being joined in the query

        self.qry = ''  # store current user query as config.qry
        self.entityFilters = []
        self.conjunctive = {}
        self.applied_filter_modules = []

        self.module_tables = {'filtered': self.filtered,
                              'time_series': self.time_series}  # store table identifiers
        self.subs = self.set_subs()

        self.DOMAIN_KNOWLEDGE = defaultdict(lambda: defaultdict(list))  # Oracle's domain knowledge
        self.init_default_domain_knowledge()
        self.name_ids, self.IDENTIFIERS = [], defaultdict(set)
        self.init_default_set_identifiers()

        self.CORE = {}
        self.init_default_uninformative_descriptors()
        self.FEATURE_DIST = {}
        self.init_default_feature_dist()

        self.added = []  # list of features added to dataframe during analysis
        self.COOCCURENCE_HASH = defaultdict(lambda: defaultdict(float))  # cooccurence hash over features
        self.MOST_RECENT_QUERY = set()  # holds set of features in most recent query
        self.EXP_DECAY = defaultdict(
            int)  # hash of each feature on exponentially decaying weight as distance since last fetched increases
        self.ITEMSETS = []  # frequent itemsets
        self.RECOMMENDATION = Recommendation()  # global RECOMMENDATION object
        self.RECOMMENDATIONS = defaultdict(list)
        self.clookup = defaultdict(set)

        self.CONF_THRESHOLD = 0.9  # confidence that a token must match an entry in DOMAIN_KNOWLEDGE before fetching that feature directly
        self.MODULE_PARSING_THRESHOLD = 0.9  # confidence that token matches module in query apriori
        self.NAME_THRESHOLD = 0.9  # confidence that a token must match an entry in IDENTIFIERS before being labeled an player id
        self.RELEVANCE_FEEDBACK_THRESHOLD = 0.5  # threshold normalized on [0, 1] to label if a feature is relevant after relevance feedback
        self.ASSOC_MIN_SUPPORT = 2  # frequent itemsets support threshold
        self.ASSOC_MIN_CONFIDENCE = 0.5  # frequent itemsets confidence threshold
        self.DECAY = 2e-4  # exponential decay cost

        self.init_default_module_update()
        self.init_special_modules()

    # INTERFACE with CONFIG
    def set_to_str(self, column, table):
        table[column] = table[column].astype(str)

    # INTERFACE with CONFIG
    def set_to_date(self, column, table):
        table[column] = pd.to_datetime(table[column])

    def init_default_preprocessing(self):
        # domain specific feature coercing
        self.set_to_str('inning', self.X)
        self.set_to_str('outs_when_up', self.X)
        self.set_to_date('Date', self.X)
        months = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                  7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        self.X['month'] = [months[t.month] for t in self.X.Date]

    # INTERFACE with CONFIG     
    def set_datasets(self, desc_filename_hash):
        # desc_filename_hash is a dictionary where key is name of relation
        # and value is file path to the relation
        self.relations = {}
        for name in desc_filename_hash:
            self.relations[name] = pd.read_csv(desc_filename_hash[name])

    def set_keywords(self):
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
        return keywords

    def set_numeric_filters(self):
        return ['above', 'below', 'over', 'under', 'between']

    def set_stoplist(self):
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
        return stoplist

    def is_actor(self, tag):
        '''Return part of speech tags that map to noun phrases in the query'''
        return tag in ['NN', 'NNP', 'NNS', 'PRP', 'CD']

    def is_desc(self, tag):
        '''Return part of speech tags that map to modifiers in the query'''
        return tag in ['CD', 'JJ', 'JJR', 'JJS', 'RBR', 'RBS', 'VBG']

    def is_prep(self, tag):
        '''Return part of speech tags that map to prepositions in the query'''
        return tag in ['IN', 'WRB']

    def is_verb(self, tag):
        '''Return part of speech tags that map to verb phrases in the query'''
        return re.match(r'VB.*', tag[1]) is not None and tag[0] not in self.stoplist['V']

    def is_genitive(self, tag):
        '''Return part of speech tags that map to possessive modifiers in the query'''
        return tag[0] == '->' or tag[1] in ['POS', 'PRP$']

    def is_adv(self, tag):
        '''Return part of speech tags that map to adverbs in the query'''
        return tag == 'RB'

    def is_gerund(self, tag):
        '''Return part of speech tags that map to gerunds in the query'''
        return tag == 'VBG'

    def is_conj(self, tag):
        '''Return part of speech tags that map to conjunctions in the query'''
        return tag in ['and', 'or', 'not']

    def initialize(self):
        # initialize filters set to contain all keywords that are not conjunctions
        for f_key in self.keywords:
            if self.keywords[f_key] != '->' and not self.is_conj(f_key):
                self.filters.add(self.keywords[f_key])
        filtered = self.domains[self.base].copy()
        return filtered

    def set_subs(self):
        # Initialize preset substitution list
        subs = defaultdict(str)
        subs['(time)->(percentage)'] = '(*PCT*)'
        subs['(time)->(percent)'] = '(*PCT*)'
        return subs

    # INTERFACE with CONFIG
    def add_to_tables(self, feature, table):
        self.tables[feature] = table

    # INTERFACE with CONFIG
    def add_to_domain_knowledge(self, column, name, values):
        # add to domain-specific background information to the system's domain knowledge
        # account for common modifiers that reference features in the dataset
        if not isinstance(values, list):
            values = [values]
        self.DOMAIN_KNOWLEDGE[column][name].extend(values)

    def init_default_domain_knowledge(self):
        # working example of domain knowledge for PitchFX database
        self.DOMAIN_KNOWLEDGE['month']['month'] = list(self.X['month'].unique())
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['astros'] = ['houmlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['rangers'] = ['texmlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['angels'] = ['anamlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['diamondbacks'] = ['arimlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['white sox'] = ['chamlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['reds'] = ['cinmlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['rockies'] = ['colmlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['tigers'] = ['detmlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['brewers'] = ['milmlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['yankees'] = ['nyamlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['athletics'] = ['oakmlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['phillies'] = ['phimlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['pirates'] = ['pitmlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['mariners'] = ['seamlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['rays'] = ['tbamlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['blue jays'] = ['tormlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['nationals'] = ['wasmlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['cubs'] = ['chnmlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['dodgers'] = ['lanmlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['marlins'] = ['miamlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['padres'] = ['sdnmlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['orioles'] = ['balmlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['royals'] = ['kcamlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['twins'] = ['minmlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['mets'] = ['nynmlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['cardinals'] = ['slnmlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['giants'] = ['sfnmlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['texas'] = ['texmlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['los angeles'] = ['anamlb', 'lanmlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['arizona'] = ['arimlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['chicago'] = ['chamlb', 'chnmlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['cincinatti'] = ['cinmlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['colorado'] = ['colmlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['detroit'] = ['detmlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['milwaukee'] = ['milmlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['new york'] = ['nyamlb', 'nynmlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['oakland'] = ['oakmlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['philadelphia'] = ['phimlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['pittsburgh'] = ['pitmlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['seattle'] = ['seamlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['tampa bay'] = ['tbamlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['toronto'] = ['tormlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['washington'] = ['wasmlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['miami'] = ['miamlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['san diego'] = ['sdnmlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['baltimore'] = ['balmlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['kansas city'] = ['kcamlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['minnesota'] = ['minmlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['st louis'] = ['stlmlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['san francisco'] = ['sfnmlb']
        self.DOMAIN_KNOWLEDGE['team_id_pitcher']['pitchers'] = list(set(reduce(lambda x, y: x + y,
                                                                               self.DOMAIN_KNOWLEDGE[
                                                                                   'team_id_pitcher'].values(), [])))                              
        self.DOMAIN_KNOWLEDGE['team_id_batter']['batters'] = self.DOMAIN_KNOWLEDGE['team_id_pitcher']['pitchers']
        self.DOMAIN_KNOWLEDGE['type']['strike'].append('S')
        self.DOMAIN_KNOWLEDGE['type']['ball'].append('B')
        self.DOMAIN_KNOWLEDGE['type']['in-play'].append('X')
        self.DOMAIN_KNOWLEDGE['pitch_type']['pitch type'] = list(self.X['pitch_type'].unique())
        self.DOMAIN_KNOWLEDGE['pitch_type']['fastball'] = ['FA', 'FB', 'FC', 'FS']
        self.DOMAIN_KNOWLEDGE['pitch_type']['seamed'] = ['FB']
        self.DOMAIN_KNOWLEDGE['pitch_type']['cutter'] = ['FC']
        self.DOMAIN_KNOWLEDGE['pitch_type']['sinker'] = ['SI']
        self.DOMAIN_KNOWLEDGE['pitch_type']['split'] = ['SF']
        self.DOMAIN_KNOWLEDGE['pitch_type']['fingered'] = ['SF']
        self.DOMAIN_KNOWLEDGE['pitch_type']['slider'] = ['SL']
        self.DOMAIN_KNOWLEDGE['pitch_type']['changeup'] = ['CH']
        self.DOMAIN_KNOWLEDGE['pitch_type']['curveball'] = ['CB', 'CU']
        self.DOMAIN_KNOWLEDGE['pitch_type']['knuckleball'] = ['KC', 'KN']
        self.DOMAIN_KNOWLEDGE['pitch_type']['knucklers'] = ['KC', 'KN']
        self.DOMAIN_KNOWLEDGE['pitch_type']['eephus'] = ['EP']
        self.DOMAIN_KNOWLEDGE['event']['forceout'] = ['force_out']
        self.DOMAIN_KNOWLEDGE['event']['groundout'] = ['Groundout']
        self.DOMAIN_KNOWLEDGE['event']['groundball'] = ['Groundout']
        self.DOMAIN_KNOWLEDGE['event']['strikeout'] = ['strikeout']
        self.DOMAIN_KNOWLEDGE['event']['homerun'] = ['home-run']
        self.DOMAIN_KNOWLEDGE['event']['walk'] = ['walk']
        self.DOMAIN_KNOWLEDGE['event']['single'] = ['single']
        self.DOMAIN_KNOWLEDGE['event']['double'] = ['double']
        self.DOMAIN_KNOWLEDGE['event']['triple'] = ['triple']
        self.DOMAIN_KNOWLEDGE['event']['lineout'] = ['Lineout']
        self.DOMAIN_KNOWLEDGE['event']['fieldout'] = ['field_out']
        self.DOMAIN_KNOWLEDGE['event']['fielderror'] = ['field_error']
        self.DOMAIN_KNOWLEDGE['event']['doubleplay'] = ['double_play', 'sac_bunt_double_play', 'sac_fly_double_play',
                                                        'strikeout_double_play']
        self.DOMAIN_KNOWLEDGE['event']['pickoff'] = ['pickoff_1b', 'pickoff_2b', 'pickoff_3b', 
                                                      'pickoff_caught_stealing_3b', 'pickoff_caught_stealing_2b']
        self.DOMAIN_KNOWLEDGE['event']['caught'] = ['caught_stealing_2b', 'caught_stealing_3b', 'caught_stealing_home', 
                                                    'pickoff_caught_stealing_3b', 'pickoff_caught_stealing_2b']
        self.DOMAIN_KNOWLEDGE['event']['bunt'] = ['Bunt Groundout', 'sac_bunt', 'Bunt Pop Out']
        self.DOMAIN_KNOWLEDGE['event']['field'] = ['Field Error']
        self.DOMAIN_KNOWLEDGE['event']['error'] = ['Field Error']
        self.DOMAIN_KNOWLEDGE['stand']['batters'] = ['L', 'R']
        self.DOMAIN_KNOWLEDGE['stand']['lefty'] = ['L']
        self.DOMAIN_KNOWLEDGE['stand']['left-handed'] = ['L']
        self.DOMAIN_KNOWLEDGE['stand']['righty'] = ['R']
        self.DOMAIN_KNOWLEDGE['stand']['right-handed'] = ['R']
        self.DOMAIN_KNOWLEDGE['p_throws']['pitchers'] = ['L', 'R']
        self.DOMAIN_KNOWLEDGE['p_throws']['lefty'] = ['L']
        self.DOMAIN_KNOWLEDGE['p_throws']['left-handed'] = ['L']
        self.DOMAIN_KNOWLEDGE['p_throws']['righty'] = ['R']
        self.DOMAIN_KNOWLEDGE['p_throws']['right-handed'] = ['R']
        self.DOMAIN_KNOWLEDGE['inning']['inning'] = list(self.X['inning'].unique())
        self.DOMAIN_KNOWLEDGE['inning']['first'] = ['1']
        self.DOMAIN_KNOWLEDGE['inning']['second'] = ['2']
        self.DOMAIN_KNOWLEDGE['inning']['third'] = ['3']
        self.DOMAIN_KNOWLEDGE['inning']['fourth'] = ['4']
        self.DOMAIN_KNOWLEDGE['inning']['fifth'] = ['5']
        self.DOMAIN_KNOWLEDGE['inning']['sixth'] = ['6']
        self.DOMAIN_KNOWLEDGE['inning']['seventh'] = ['7']
        self.DOMAIN_KNOWLEDGE['inning']['eighth'] = ['8']
        self.DOMAIN_KNOWLEDGE['inning']['ninth'] = ['9']
        self.DOMAIN_KNOWLEDGE['inning_side']['top'] = ['top']
        self.DOMAIN_KNOWLEDGE['inning_side']['bottom'] = ['bottom']
        self.DOMAIN_KNOWLEDGE['px']['x'] = ['is.numeric']
        self.DOMAIN_KNOWLEDGE['px']['location'] = ['is.numeric']
        self.DOMAIN_KNOWLEDGE['px']['left/right'] = ['is.numeric']
        self.DOMAIN_KNOWLEDGE['px']['horizontal'] = ['is.numeric']
        self.DOMAIN_KNOWLEDGE['pz']['z'] = ['is.numeric']
        self.DOMAIN_KNOWLEDGE['pz']['location'] = ['is.numeric']
        self.DOMAIN_KNOWLEDGE['pz']['pitch'] = ['is.numeric']
        self.DOMAIN_KNOWLEDGE['pz']['height'] = ['is.numeric']
        self.DOMAIN_KNOWLEDGE['release_speed']['start'] = ['is.numeric']
        self.DOMAIN_KNOWLEDGE['release_speed']['pitch'] = ['is.numeric']
        self.DOMAIN_KNOWLEDGE['release_speed']['ball'] = ['is.numeric']
        self.DOMAIN_KNOWLEDGE['release_speed']['speed'] = ['is.numeric']
        self.DOMAIN_KNOWLEDGE['release_speed']['velocity'] = ['is.numeric']
        self.DOMAIN_KNOWLEDGE['release_speed']['release'] = ['is.numeric']
        #self.DOMAIN_KNOWLEDGE['effective_speed']['end'] = ['is.numeric']
        #self.DOMAIN_KNOWLEDGE['effective_speed']['effective'] = ['is.numeric']
        #self.DOMAIN_KNOWLEDGE['effective_speed']['speed'] = ['is.numeric']
        #self.DOMAIN_KNOWLEDGE['launch_speed']['launch'] = ['is.numeric']
        #self.DOMAIN_KNOWLEDGE['launch_speed']['speed'] = ['is.numeric']
        #self.DOMAIN_KNOWLEDGE['launch_speed']['launch'] = ['is.numeric']
        #self.DOMAIN_KNOWLEDGE['launch_speed']['angle'] = ['is.numeric']
        #self.DOMAIN_KNOWLEDGE['launch_speed_angle']['launch'] = ['is.numeric']
        #self.DOMAIN_KNOWLEDGE['launch_speed_angle']['speed'] = ['is.numeric']
        #self.DOMAIN_KNOWLEDGE['launch_speed_angle']['angle'] = ['is.numeric']
        self.DOMAIN_KNOWLEDGE['pfx_x']['x'] = ['is.numeric']
        self.DOMAIN_KNOWLEDGE['pfx_x']['movement'] = ['is.numeric']
        self.DOMAIN_KNOWLEDGE['pfx_x']['left/right'] = ['is.numeric']
        self.DOMAIN_KNOWLEDGE['pfx_x']['horizontal'] = ['is.numeric']
        self.DOMAIN_KNOWLEDGE['pfx_x']['z'] = ['is.numeric']
        self.DOMAIN_KNOWLEDGE['pfx_z']['movement'] = ['is.numeric']
        self.DOMAIN_KNOWLEDGE['pfx_z']['vertical'] = ['is.numeric']
        self.DOMAIN_KNOWLEDGE['ax']['x'] = ['is.numeric']
        self.DOMAIN_KNOWLEDGE['ax']['acceleration'] = ['is.numeric']
        self.DOMAIN_KNOWLEDGE['ax']['left/right'] = ['is.numeric']
        self.DOMAIN_KNOWLEDGE['ax']['horizontal'] = ['is.numeric']
        self.DOMAIN_KNOWLEDGE['az']['z'] = ['is.numeric']
        self.DOMAIN_KNOWLEDGE['az']['acceleration'] = ['is.numeric']
        self.DOMAIN_KNOWLEDGE['az']['vertical'] = ['is.numeric']
        self.DOMAIN_KNOWLEDGE['break_angle']['break'] = ['is.numeric']
        self.DOMAIN_KNOWLEDGE['break_angle']['angle'] = ['is.numeric']
        self.DOMAIN_KNOWLEDGE['break_length']['break'] = ['is.numeric']
        self.DOMAIN_KNOWLEDGE['break_length']['length'] = ['is.numeric']
        self.DOMAIN_KNOWLEDGE['woba_value']['woba'] = ['is.numeric']
        self.DOMAIN_KNOWLEDGE['woba_value']['value'] = ['is.numeric']
        self.DOMAIN_KNOWLEDGE['babip_value']['babip'] = ['is.numeric']
        self.DOMAIN_KNOWLEDGE['babip_value']['value'] = ['is.numeric']
        self.DOMAIN_KNOWLEDGE['outs_when_up']['outs'] = ['0', '1', '2']
        self.DOMAIN_KNOWLEDGE['outs_when_up']['outs'] = ['is.numeric']
        self.DOMAIN_KNOWLEDGE['strikes']['strikes'] = ['0', '1', '2']
        self.DOMAIN_KNOWLEDGE['strikes']['strikes'] = ['is.numeric']
        self.DOMAIN_KNOWLEDGE['balls']['balls'] = ['0', '1', '2', '3']
        self.DOMAIN_KNOWLEDGE['balls']['balls'] = ['is.numeric']

    # INTERFACE with CONFIG
    def set_identifiers(self, name_ids, id_values):
        # name_ids is list of fields matching to name identifiers
        # id_values is list of lists where ith element is list of
        # ways of referring to the ith value in name_ids
        self.name_ids = name_ids
        self.IDENTIFIERS = defaultdict(set)
        for name_id in name_ids:
            self.IDENTIFIERS[name_id] = set(self.X[name_id])
        for i in range(len(name_ids)):
            name_id = name_ids[i]
            self.IDENTIFIERS[name_id] = {x for x in self.IDENTIFIERS[name_id] if x == x}
            for ref in id_values[i]:
                self.DOMAIN_KNOWLEDGE[name_id][ref] = list(self.X[name_id].unique())

    def init_default_set_identifiers(self):
        self.set_identifiers(['batter_name', 'pitcher_name'],
                             [['batter name', 'batter'], ['pitcher name', 'pitcher']])

    # INTERFACE with CONFIG
    def set_uninformative_descriptors(self, CORE):
        self.CORE = CORE

    def init_default_uninformative_descriptors(self):
        self.set_uninformative_descriptors({'pitch': None,
                                            'pitches': None,
                                            'atbat': None,
                                            'atbats': None})

    # INTERFACE with CONFIG
    def set_feature_dist(self, FEATURE_DIST):
        self.FEATURE_DIST = FEATURE_DIST

    def init_default_feature_dist(self):
        FEATURE_DIST = {'pitch type': 'pitch_type',
                        'inning': 'inning', 'month': 'month', 'team': 'team',
                        'batter': self.name_ids[0], 'pitcher': self.name_ids[1],
                        'batter name': self.name_ids[0], 'pitcher name': self.name_ids[1]}
        self.set_feature_dist(FEATURE_DIST)

    # INTERFACE with CONFIG
    def update_module_parsers(self, update_list, module):
        for key in update_list:
            tokens = key.split()
            for token in tokens:
                # module_parsers.add(token)
                self.module_hash[token].append(module)

    # INTERFACE with CONFIG
    def update_modules_reversed(self, update_list, module):
        for e in update_list:
            self.modules_reversed[module].add(e)

    # INTERFACE with CONFIG
    def set_module_keywords(self, key, module):
        self.module_keywords[key] = module

    def init_default_module_update(self):
        self.module_keywords['strike rate'] = StrikeRate
        self.update_module_parsers(['strike rate'], StrikeRate)
        self.update_modules_reversed(['strike rate'], StrikeRate)
        self.module_keywords['OBP'] = OBP
        self.module_keywords['on-base percentage'] = OBP
        self.update_module_parsers(['OBP', 'on-base percentage'], OBP)
        self.update_modules_reversed(['OBP', 'on-base percentage'], OBP)
        self.module_keywords['lead-runner-on-third'] = LeadRunnerOnThird
        self.module_keywords['lead runner on third'] = LeadRunnerOnThird
        self.module_keywords['lead runner on third base'] = LeadRunnerOnThird
        self.update_module_parsers(['lead-runner-on-third', 'lead runner on third',
                                    'lead runner on third base'], LeadRunnerOnThird)
        self.update_modules_reversed(['lead-runner-on-third', 'lead runner on third',
                                      'lead runner on third base'], LeadRunnerOnThird)
        self.module_keywords['lead-runner-on-second'] = LeadRunnerOnSecond
        self.update_module_parsers(['lead-runner-on-second'], LeadRunnerOnSecond)
        self.update_modules_reversed(['lead-runner-on-second'], LeadRunnerOnSecond)
        self.module_keywords['lead-runner-on-first'] = LeadRunnerOnFirst
        self.update_module_parsers(['lead-runner-on-first'], LeadRunnerOnFirst)
        self.update_modules_reversed(['lead-runner-on-first'], LeadRunnerOnFirst)
        self.module_keywords['stealing'] = Steal
        self.update_module_parsers(['stealing'], Steal)
        self.update_modules_reversed(['stealing'], Steal)
        self.module_keywords['bunt ratio'] = Bunt
        self.update_module_parsers(['bunt ratio'], Bunt)
        self.update_modules_reversed(['bunt ratio'], Bunt)
        self.module_keywords['FPT'] = FPT
        self.module_keywords['first pitch take ratio'] = FPT
        self.update_module_parsers(['FPT','first pitch take ratio'], FPT)
        self.update_modules_reversed(['FPT','first pitch take ratio'], FPT)
        self.module_keywords['FPS'] = FPS
        self.module_keywords['first pitch swing ratio'] = FPS
        self.update_module_parsers(['FPS','first pitch swing ratio'], FPS)
        self.update_modules_reversed(['FPS','first pitch swing ratio'], FPS)
        self.module_keywords['flashcard'] = Flashcard
        self.update_module_parsers(['flashcard'], Flashcard)
        self.update_modules_reversed(['flashcard'], Flashcard)
        self.module_keywords['pitchers-like'] = PitchersLikeMe
        self.module_keywords['similar-pitchers'] = PitchersLikeMe
        self.update_module_parsers(['pitchers-like','similar-pitchers'], PitchersLikeMe)
        self.update_modules_reversed(['pitchers-like','similar-pitchers'], PitchersLikeMe)
        
    def init_special_modules(self):
        #return [Flashcard]
        return [Flashcard, PitchersLikeMe]