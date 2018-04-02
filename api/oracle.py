import os
import random
import numpy as np
from scipy import stats
import pandas as pd
import nltk
from collections import defaultdict, deque
import re
import itertools
import jellyfish
import plotly
import plotly.plotly as py
import plotly.figure_factory as ff
from plotly.graph_objs import *
from pymining import itemmining, assocrules
from functools import reduce
from modules.module import Module
from modules.strikerate import StrikeRate
from modules.obp import OBP
from modules.leadrunnerthird import LeadRunnerOnThird
from modules.leadrunnersecond import LeadRunnerOnSecond
from modules.leadrunnerfirst import LeadRunnerOnFirst
from recommendations.recommendation import Recommendation, RecommendationItem
from dknowledge import Config

class Oracle:
    
    def __init__(self, config):
        self.config = config
        
    def is_number(self, s):
        '''Return if a string can be parsed as a number'''
        try:
            float(s)
            return True
        except ValueError:
            return False
    
    def user_input(self, args):
    
        '''Prompt user to clarify ambiguous tokens in the query'''
      
        print('What do you mean by ' + "\'" + args + "\'")
    
        feature = input('Relevant feature: ') # prompt for feature
        value = input('Associated value: ') # prompt for value
    
        return (feature, [value])
    
    def identify_val_via_user(self, relevant, value):
    
        '''Given a feature determined to be relevant, prompt user for its associated value if ambiguous'''
    
        print('How does ' + "\'" + value + \
          "\'" + 'relate to the field ' + "\'" + relevant + "\'")
        val = input('Relevant value: ') # prompt for value
    
        return [val]
    
    def substitute_conjunction(self, c):
    
        '''Return base Pandas substitution for conjunction'''
    
        result = {'OR': '|', 'AND': '&', 'NOT': '!'}
        return result[c]
    
    def is_expr(self, entity):
        '''Holds suite of pre built functions (averages, etc.)'''
        return entity in config.module_keywords
    
    def module_filter(self, module):
        module = module()
        iset = range(len(config.filtered.index))
        rset = module.execute(iset, config.module_tables['filtered'], False)
        config.filtered = config.filtered.iloc[rset,:]
    
    def print_module_delegation(self, keyword, module):
        print("delegating computation of '" + keyword + "' to module " + str(module))
    
    def match_expr(self, entity, b_list, title, e_filters):
    
        '''driver for filtering on pre-built functions'''

        plot, status = None, True
        result, names, plot_l = '', [], []
        module = None
        config.module_tables = {'filtered': config.filtered, 'time_series': config.filtered}

        if b_list is not None: # if b_list is not empty, perform a plot over distribution of feature
            dist_plot = True
            result = {}
            for name, group in config.filtered: # iterate over each group
                calc = None
                if entity in config.module_keywords:
                    module = config.module_keywords[entity]() if module is None else module
                    calc = module.execute(config.filtered.indices[name], 
                                          config.module_tables['filtered'], True)
                if calc is not None:
                    if (entity not in e_filters or (calc.shape == (1,1) and
                        (re.search(r'(?i)^over$', e_filters[entity][0]) is not None 
                         and calc.iloc[0,0] > float(e_filters[entity][1])) or
                        (re.search(r'(?i)^under$', e_filters[entity][0]) is not None 
                         and calc.iloc[0,0] < float(e_filters[entity][1])))):
                        names.append(name) # add to x labels list
                        result[name] = calc
                        if dist_plot:
                            calc = calc.iloc[0,0]
                            plot_l.append(calc) # add to y values list
                        else:
                            dist_plot = False

            if plot_l and dist_plot: # if calculated results are returned, generate the plot
                plot = pd.DataFrame({'x': names, 'y': plot_l})
                #plot = self.automate_plot_by_(names, plot_l, entity, title, 'bar')
                status = False # set time_series plot status to False

        else: # otherwise plot time series by default

            if entity in config.module_keywords:

                module = config.module_keywords[entity]()
                iset = range(len(config.filtered.index))
                result = module.execute(iset, config.module_tables['filtered'], False)
                config.time_series = config.filtered.groupby('Date') # group filtered dataset by date

                if result.shape == (1,1):
                    for name, group in config.time_series:
                        module = config.module_keywords[entity]() if module is None else module
                        calc = module.execute(config.time_series.indices[name], 
                                              config.module_tables['time_series'], False)
                        calc = calc.iloc[0,0]
                        if calc is not None:
                            names.append(name)
                            plot_l.append(calc)

                if plot_l:
                    # plot time series chart
                    plot = pd.DataFrame({'x': names, 'y': plot_l})
                    #plot = self.automate_plot_by_(names, plot_l, entity, title, 'scatter')

        return result, plot, status
    
    def sequential_entity(self, e_list, b_feat, actors):
    
        '''Perform sequence of filter reductions on the entity list'''

        response = str(' '.join([re.findall(r'^\(?(.+[^)])\)?', actor[0])[0] for actor in actors])).title()
        criteria, added, expr = self.generate_criteria(e_list) # generate entity criteria (feature, [values])
        result = None

        eHash = {}
        for eFilter in config.entityFilters:
            eFilterIndex = config.qry.find(eFilter[0] + ' ' + eFilter[1])
            minDiff = None
            for e in expr:
                eExprIndex = config.qry.find(e)
                if minDiff is None or (eFilterIndex - eExprIndex) < minDiff:
                    minDiff = eFilterIndex - eExprIndex
                    eHash[eFilter] = e
        
        eHash = {v: k for k, v in eHash.items()}

        for i in range(len(criteria)): # sequentially filter database on entity criteria

            exec_str = '(' + self.perform_filter(config.FILTER, criteria[i], b_feat) + ')' # perform filter 

            if not b_feat:
                exec_str = "config.filtered = config.filtered[({0})]".format(exec_str)
            else:
                exec_str = "config.filtered = config.filtered.apply(lambda g: ({0}))".format(exec_str)    
            print('\n' + exec_str)
            exec(exec_str)

            if b_feat:
                config.filtered = config.filtered.groupby(b_feat)

            feat = criteria[i][0]
            config.MOST_RECENT_QUERY.add(feat) # cache this query as most recent
            config.ITEMSETS[len(config.ITEMSETS)-1].add(feat) # add to frequent itemsets cache
            config.EXP_DECAY[feat] = 1 # set distance of feature since last fetched to 1

            for j in range(i, len(criteria)):
                config.COOCCURENCE_HASH[feat][criteria[j][0]] += 1 # update cooccurrence hash

        for e in expr:
            result, plot, status = self.match_expr(e, b_feat, response, eHash) # sequentially execute all user defined actions

        for feat in added: # restore original columns
            del config.filtered[feat]
            del config.DOMAIN_KNOWLEDGE[feat]

        config.ITEMSETS[len(config.ITEMSETS)-1] = tuple(config.ITEMSETS[len(config.ITEMSETS)-1])
        
        sample_size = None
        if result is not None:
            if status: # output textual response
                if result.shape == (1,1):
                    result.columns = ['Output']
                    sample_size = pd.DataFrame([config.filtered.shape[0]])
            else:
                appended = pd.DataFrame()
                names = []
                for name in result:
                    if result[name].shape[1] == 1:
                        result[name].columns = ['Output']
                    appended = appended.append(result[name])
                    names.append(name)
                appended['name'] = names
                appended.columns = ['Output','name']
                result = appended
                sample_size = config.filtered.size().to_frame()
            sample_size.columns = ['Sample Sizes']
        else:
            print('\n The output of your query resulted in a sample size of 0. \
                   Consider refining your query in the case that it was too specific.')
            return None

        rterms = self.generate_features_rf_()

        if rterms:
            print('\n' + '(Relevance Feedback) Investigate Features More Like This: ' + str(rterms) + '\n')
            
        return result, sample_size, plot, rterms
        
    def generate_criteria(self, e_list):
    
        '''Parse each entity token in the query into list of (feature, [value])'''

        entities = []

        for i in range(len(e_list)): # iterate over entity list
            if e_list[i] == config.GENITIVE: 
                continue
            entities.append(re.findall(r'^\(?(.+[^)])\)?', e_list[i])[0])

        prev_feat = None
        criteria, added, expr = [], [], []
        rec_item = config.RECOMMENDATION.data[config.qry] # hash current query into RECOMMENDATIONS hash
        conj = None

        for entity in entities:

            if self.is_expr(entity): # check if token is a pre-built expression
                expr.append(entity)
                continue
            elif config.is_conj(entity): # check if token is a conjunction and tag it
                conj = entity
                continue

            curr_feat = self.match_(entity) # extract relevant feature
            start = config.qry.find(entity)
            end = start + len(entity)
            rec_item.index_hash[curr_feat] = (start, end) 
            config.RECOMMENDATION.data[config.qry] = rec_item

            if curr_feat is None:
                return self.user_input(entity) # prompt user to clarify input
            else:
                # extract associated value to relevant feature
                val = self.match_arg_to_feature_value(curr_feat, entity) 

                while val is None:
                    # prompt user if argument specification is ambiguous
                    val = self.identify_val_via_user(curr_feat, val) 

                if criteria: # handle conjunctions in query
                    if conj == 'OR': # union logic
                        if criteria[len(criteria)-1][1] == curr_feat:
                            criteria[len(criteria)-1][1][0] += '+' + '+'.join(val)
                        else: 
                            pass
                    elif conj == 'NOT': # negation logic
                        criteria.extend([(curr_feat, ['!' + v_key]) for v_key in val])
                    else:
                        criteria.append((curr_feat, ['+'.join(val)]))
                else: # intersection logic
                    criteria.append((curr_feat, ['+'.join(val)]))
                conj = None

        features = [feat for feat in config.RECOMMENDATION.data[config.qry].index_hash]
        features.sort() # sort features to specify common key
        config.RECOMMENDATIONS[str(features)].append(config.RECOMMENDATION) # hash features in RECOMMENDATIONS

        return criteria, added, expr

    def sequential_filter(self, f_list):

        '''Execute sequential filters over dataset on supplied filtering criteria'''

        config.MOST_RECENT_QUERY = set() # reinitialize MOST_RECENT_QUERY
        config.ITEMSETS.append(set()) # reinitialize ITEMSETS

        for feat in config.EXP_DECAY:
            config.EXP_DECAY[feat] += 1 # increment all features distance from being last fetched by 1

        config.filtered = config.X.copy() # reinitialize filtered
        config.time_series = config.filtered
        config.module_tables = {'filtered': config.filtered, 'time_series': config.time_series}
        filters = self.feature_assoc_filters_helper(f_list) # generate filtering criteria as [[(feature, [values])]]
        grouped, b_list = None, []

        for f_key in filters:
            exec_str = ''
            for i in range(len(f_key)): # iterate over each filtering criterion
                f_, c_, conj_ = f_key[i][0], f_key[i][1], f_key[i][2] # extract filtering operation, criteria, join flag
                if c_ is None: continue
                feat = c_[0]
                config.MOST_RECENT_QUERY.add(feat)
                config.ITEMSETS[len(config.ITEMSETS)-1].add(feat) # update ITEMSETS
                config.EXP_DECAY[feat] = 1 # set distance since being fetched to 1
                for j in range(i, len(f_key)):
                    config.COOCCURENCE_HASH[feat][f_key[j][1][0]] += 1 # update cooccurence hash
                if not c_[1]:
                    b_list.append(feat) # add to b_list if filtering criteria is over a feature's entire distribution
                    continue
                if conj_ is not None:
                    conj_ = self.substitute_conjunction(conj_)
                    exec_str += " {0} ".format(conj_)
                if c_[1] in config.module_keywords.values():
                    self.print_module_delegation(feat, str(c_[1]))
                    self.module_filter(c_[1])
                else:
                    exec_str += self.perform_filter(f_, c_, False)
            if exec_str:
                exec_str = '(' + exec_str + ')'
                exec_str = "config.filtered = config.filtered[({0})]".format(exec_str)
                print('\n' + exec_str)
                exec(exec_str)

        if b_list:
            grouped, b_feat = self.by_helper_([b_list, []])
        else:
            for module in config.clookup:
                if module().set_module():
                    keyword = list(module().get_lexicon())[0]
                    self.print_module_delegation(keyword, str(module))
                    self.module_filter(module)

        return (config.filtered, None) if grouped is None else (grouped, b_feat)

    def check_filters(self, fname):

        max_conf, max_filt = 0, None

        for entry in config.module_keywords:
            conf_f = jellyfish.jaro_distance(fname, entry)
            if conf_f > max_conf:
                max_conf, max_filt = conf_f, config.module_keywords[entry]

        if max_conf > config.CONF_THRESHOLD:
            return max_filt
        else:
            return None

    def feature_assoc_filters_helper(self, filters):

        '''Return list of (relevant feature, [associated values]) tuples to filter automatically'''

        f_ = [] # initialize result list
        featureHash = {}
        config.RECOMMENDATION = Recommendation(config.qry) # initialize RECOMMENDATION to hash on current query
        rec_item = RecommendationItem()
        config.entityFilters, prev_numeric, numeric_token = [], [], None

        for f_item in filters:

            tokens = re.findall(r'\(.+?\)|AND|OR|NOT', f_item) # tokenize filters
            tokens[0] = re.findall(r'%(.+)%', tokens[0])[0]
            tokens[1] = tokens[1][1:]
            tokens = [re.findall(r'\((.+?)\)', token)[0] if '(' in token else token for token in tokens]
            prev_feat, relevant, val, cat, joined = None, None, None, '', []
            i, inc, c_flag = 1, False, False
            negated = None

            cset = config.clookup.values()
            cset = [k for ckey in cset for k in ckey]

            while i < len(tokens): # iterate over each filter in the filtered list

                token = tokens[i]
                token = token.split()
                count = 0

                for t in token:
                    for ckey in cset:
                        jaro = jellyfish.jaro_distance(t, ckey)
                        if jaro > config.CONF_THRESHOLD:
                            count += 1

                if count == len(token):
                    f_.append([(tokens[0], None, None)])
                    i += 1
                    continue

                is_filter = False

                if tokens[i].startswith('NOT'): 
                    cat += tokens[i]
                    negated = re.findall(r'NOT|.+', tokens[i])[1]
                    relevant = self.match_(negated)
                    if relevant is None:
                        relevant, val = self.user_input(negated)

                if negated is None:

                    if tokens[0] in config.numericFilters:
                        if not prev_numeric:
                            config.entityFilters.append((tokens[0], tokens[i]))
                            break
                        else: 
                            relevant = prev_numeric.pop()
                            val = [tokens[i]] # set argument value to match token if a numeric feature

                    if relevant is None:
                        relevant = self.match_(tokens[i]) # extract relevant feature for tokens[i]
                        if relevant is None:
                            filter_to_apply = self.check_filters(tokens[i])
                            if filter_to_apply is None:
                                relevant, val = self.user_input(tokens[i]) # prompt user if ambiguous
                            else:
                                is_filter = True
                                f_.append([(tokens[0], (tokens[i], filter_to_apply), None)])
                        if not is_filter:
                            start = config.qry.find(tokens[i]) # start index stored before cached in RECOMMENDATION
                            end = start + len(tokens[i]) # end index stored before cached in RECOMMENDATION
                            rec_item.index_hash[relevant] = (start, end) 

                            # set current recommendation item as the value to RECOMMENDATION hashed on the current query
                            config.RECOMMENDATION.data[config.qry] = rec_item

                    if val is None and not is_filter:

                        cat += tokens[i]
                        joined.append(tokens[i])

                        while (i+1) < len(tokens) and config.is_conj(tokens[i+1].lower()): # check if next token is conjunction
                            # otherwise append the next feature value to the running result
                            cat += tokens[i+1] + tokens[i+2]
                            joined.append(tokens[i+2])
                            i += 3

                        for core_entity in config.CORE:
                            if jellyfish.jaro_distance(core_entity, cat) > config.NAME_THRESHOLD:
                                c_flag = True
                                break

                if not c_flag:

                    if not is_filter:

                        if val is None:

                            # call subroutine to generate filter criteria for current (feature, [arguments]) pair
                            result = self.generate_filter_criteria(cat, relevant)
                            inserted = False

                            if not result[1] or result[1][0] != 'is.numeric':
                                for joinedEntry in joined:
                                    if (joinedEntry in config.conjunctive and 
                                        config.conjunctive[joinedEntry][1] in featureHash):
                                        stored = config.conjunctive[joinedEntry]
                                        if not inserted:
                                            f_[featureHash[stored[1]]].append((tokens[0], result, stored[0]))
                                            inserted = True                                
                                        featureHash[joinedEntry] = featureHash[stored[1]]

                                if not inserted: 
                                    f_.append([(tokens[0], result, None)]) # append to result list
                                    for joinedEntry in joined:    
                                        featureHash[joinedEntry] = len(f_)-1

                            else:
                                numeric_token = cat
                                prev_numeric.append(result[0])

                        elif numeric_token is not None:
                            if val[0] in config.conjunctive and config.conjunctive[val[0]][1] in featureHash:
                                stored = config.conjunctive[val[0]]
                                f_[featureHash[stored[1]]].append((tokens[0], (relevant, val), stored[0]))
                                featureHash[val[0]] = featureHash[stored[1]]
                            elif numeric_token in config.conjunctive and config.conjunctive[numeric_token][1] in featureHash:
                                stored = config.conjunctive[numeric_token]
                                f_[featureHash[stored[1]]].append((tokens[0], (relevant, val), stored[0]))
                                featureHash[numeric_token] = featureHash[stored[1]]
                            else:
                                f_.append([(tokens[0], (relevant, val), None)]) # append to result list
                                featureHash[val[0]] = len(f_)-1

                            numeric_token = None

                if not inc: 
                    i += 1

                inc, c_flag, val, cat, joined = False, False, None, '', []

        return f_


    def perform_filter(self, f_key, c_key, b_list):

        '''Perform relevant filter'''

        result = {
            'filter': lambda D: self.filter_helper_(c_key, b_list),
            'by': lambda D: self.by_helper_(c_key),
            'over': lambda D: self.over_helper_(c_key),
            'under': lambda D: self.under_helper_(c_key),
            'between': lambda D: self.between_helper_(c_key),
            'except': lambda D: self.except_helper_(c_key),
            'near': lambda D: self.near_helper_(c_key),
            'until': lambda D: self.until_helper_(c_key),
            'to': lambda D: self.to_helper_(c_key),
            'after': lambda D: self.after_helper_(c_key),
            'before': lambda D: self.before_helper_(c_key),
            'against': lambda D: self.compare_helper_(c_key)
        }[f_key](config.filtered)

        return result;


    def generate_filter_criteria(self, args, hint=None):

        '''Return relevant (feature, [values]) tuple that matches argument'''

        c_keys = ['AND', 'OR', 'NOT']
        pat = '((?:' + '|'.join(c_keys) + '))'
        c_list = re.split(pat, args) # split on conjunctions
        c_list = [t for t in c_list if t]

        if hint is None:
            return self.user_input(args) # if ambiguous, prompt user to specify
        else:
            relevant = hint

        print('\nfeature association: most relevant feature for arg (', args, ') is ' + relevant)

        if relevant in config.name_ids and args in config.IDENTIFIERS[relevant]:
            return (relevant, [args])

        if 'is.numeric' in list(config.DOMAIN_KNOWLEDGE[relevant].values())[0]:
            if not self.is_number(args):
                test_split = re.split(pat, args)
                criteria = self.generate_filter_helper_(relevant, test_split)
                return (relevant, criteria)
            else:
                return (relevant, ['is.numeric'])

        featureDist = [config.FEATURE_DIST[feat] for feat in config.FEATURE_DIST 
                       if jellyfish.jaro_distance(args, feat) > config.CONF_THRESHOLD]
        if featureDist:
            return (featureDist[0], [])

        criteria = self.generate_filter_helper_(relevant, c_list)

        return (relevant, criteria)

    def generate_filter_helper_(self, relevant, c_list):

        criteria = []
        conj, unionIndex = None, 0

        for term in c_list:

            if config.is_conj(term.lower()):
                conj = term
            else:
                lookup = re.match('(.+)', term).group() # extract token
                val = self.match_arg_to_feature_value(relevant, lookup) # look up associated filter criteria on feature
                print('generate_criteria: ' + str(val))

                while val is None:
                    val = self.identify_val_via_user(relevant, val) # prompt user if associated value is ambiguous

                if conj == 'NOT': # handle negation logic
                    negated = ['!' + v_key for v_key in val]
                    criteria.extend(negated)
                    unionIndex = len(criteria) - len(negated)
                elif criteria:
                    if conj == 'OR' or len(val) > 1: # handle union logic
                        for i in range(unionIndex, len(criteria)):
                            criteria[i] += '+' + '+'.join(val)
                        unionIndex = len(criteria)-1
                    else:
                        criteria.append('+'.join(val))
                        unionIndex = len(criteria)-1
                else: # handle intersection logic
                    criteria.append('+'.join(val))
                    unionIndex = len(criteria)-1    

                conj = None

        return criteria

    def filter_helper_(self, c_key, b_list):

        '''Perform filter on categorical feature'''

        feature = c_key[0] # extract feature
        args = c_key[1] # extract relevant values of feature to filter on
        exec_str = ''
        print('(' + str(feature) + ', ' + str(args) + '): filter_helper_')

        if b_list: # handle automated filtering over distribution of a feature

            for index, f_tok in enumerate(args): # iterate over each argument            

                f_ = re.split(r'(?:\+|!)', f_tok) # split by filters handled with conjunctive logic

                if exec_str:
                    exec_str += ' | '
                if '!' in f_tok: # handle negation
                    exec_str += "(g[g[\'" + feature + "\'] != \'" + f_[1] + "\'])"
                else:
                    union = [item for item in f_]
                    exec_str += "(g[g[\'" + feature + "\'].isin(" + str(union) + ")])"

        else:

            for index, f_tok in enumerate(args): # iterate over each argument

                f_ = re.split(r'(?:\+|!)', f_tok) # split on conjunctions            

                if exec_str:
                    exec_str += ' | '
                if '!' in f_tok: # negation logic
                    exec_str += "(config.filtered[\'" + feature + "\'] != \'" + f_[1] + "\')"
                    continue     
                exec_str += "(config.filtered[\'" + feature + "\'] == \'" + f_[0] + "\')"            

                for i in range(1, len(f_)):
                    exec_str += " | (config.filtered[\'" + feature + "\'] == \'" + f_[i] + "\')" # handle union logic

        return exec_str

    def by_helper_(self, c_key):

        '''Filter database over distribution of a feature'''

        features = c_key[0] # extract features
        config.filtered = config.filtered.groupby(features) # group by features

        return config.filtered, features

    def over_helper_(self, c_key):

        '''Filter database on values over a threshold for a feature'''

        feature = c_key[0] # extract feature
        args = c_key[1] # extract value to filter on
        exec_str = "(config.filtered[\'" + feature + "\'] > " + args[0] + ")"

        return exec_str

    def under_helper_(self, c_key):

        '''Filter database on values under a threshold for a feature'''

        feature = c_key[0] # extract feature
        args = c_key[1] # extract value to filter on
        exec_str = "(config.filtered[\'" + feature + "\'] < " + args[0] + ")"

        return exec_str

    def between_helper_(self, c_key):

        '''Filter database on values between two thresholds for a feature'''

        feature = c_key[0] # extract feature
        args = c_key[1] # extract value to filter on
        left, right = args[0], args[1]
        exec_str = "(config.filtered[\'" + feature + "\'] > " + left + ")" \
                    "& (config.filtered[\'" + feature + "\'] < " + right + ")"

        return exec_str

    def except_helper_(self, c_key):

        '''Filter database on values of a categorical feature except for those specified in the argument'''

        feature = c_key[0] # extract feature
        args = c_key[1] # extract values to filter on
        unique_vals = list(set(list(config.filtered[feature].unique())) - set([args]))
        unique_vals = [x for x in unique_vals if x == x]
        unique_vals = '+'.join(unique_vals)

        return filter_helper_([feature, unique_vals], None)

    def near_helper_(self, c_key):

        '''Filter database on values within +/- 0.5 std of the argument on a numerical feature'''

        feature = c_key[0] # extract relevant feature
        args = c_key[1] # extract values to filter on
        left = str(float(args[0]) - 0.5 * config.filtered[feature].std()) # left bound -0.5 std
        right = str(float(args[1]) + 0.5 * config.filtered[feature].std()) # right bound +0.5 std

        if left > right:
            left, right = right, left

        return between_helper_([feature, [left, right]])

    def automate_plot_by_(self, x, y, entity, title, chart_type):

        '''Return automated visualization relevant to queried features'''

        font=dict(family='Courier New, monospace', size=15, color='#7f7f7f') # set font

        xaxis=dict( # set x-axis plot attributes
            titlefont=dict(
                family='Courier New, monospace',
                size=12,
                color='#7f7f7f'
            )
        )

        yaxis=dict( # set y-axis plot attributes
            title = entity,
            titlefont=dict(
                family='Courier New, monospace',
                size=12,
                color='#7f7f7f'
            )
        )

        if chart_type == 'bar':
            data = [Bar(x=x, y=y)] # set data for bar chart chart
        else:
            data = [plotly.graph_objs.Scatter(x=x, y=y)] # set data for defaulted time series scatter
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label='1m',
                         step='month',
                         stepmode='backward'),
                    dict(count=6,
                         label='6m',
                         step='month',
                         stepmode='backward'),
                    dict(step='all')
                ])
            )

            xaxis['rangeselector'] = rangeselector # add range selector
            xaxis['rangeslider'] = dict()

        layout = plotly.graph_objs.Layout( # adjust plot layout
            title=title,
            font=font,
            xaxis=xaxis,
            yaxis=yaxis
        )

        fig = plotly.graph_objs.Figure(data=data, layout=layout) # store automated plot

        return py.iplot(fig, filename='extract_bar') if chart_type == 'bar' else py.iplot(fig, filename='extract_scatter')        

    def match_(self, args):

        '''Return relevant feature to filter on'''

        featureDist = [config.FEATURE_DIST[feat] for feat in config.FEATURE_DIST 
                       if jellyfish.jaro_distance(args, feat) > config.CONF_THRESHOLD]
        if featureDist:
            return featureDist[0]

        found_batter, found_pitcher = False, False
        for identifier in config.IDENTIFIERS:
            if args in config.IDENTIFIERS[identifier]:
                if identifier == config.name_ids[0]:
                    found_batter = True
                else:
                    found_pitcher = True

        if not (found_batter and found_pitcher):
            if found_batter:
                return (config.name_ids[0])
            elif found_pitcher:
                return (config.name_ids[1])
        else:
            return ((config.name_ids[0], [args]) 
                    if len(config.X[config.X[config.name_ids[0]] == args]) > len(config.X[config.X[config.name_ids[1]] == args]) 
                    else (config.name_ids[1]))

        max_conf, max_feat = 0, ''

        for entry in config.DOMAIN_KNOWLEDGE: # iterate over terms in the system's domain knowledge

            if args in config.DOMAIN_KNOWLEDGE[entry]: # if term matches exactly return it
                return entry

            tokens = args.split() # otherwise split tokens and accumulate evidence of belonging to each feature
            conf_f = 0 # confidence
            skipped = 0 # tokens not considered in parsing of the feature

            for token in tokens: # iterate over each token

                if not self.is_number(token):

                    curr = 0
                    skipped += 1

                    for desc in list(config.DOMAIN_KNOWLEDGE[entry].keys()): # iterate over each term in domain knowledge

                        # compute string similarity of query vs term in DOMAIN_KNOWLEDGE
                        curr = max(jellyfish.jaro_distance(desc, token), curr) # string similarity by jaro_distance

                    conf_f += curr # accumulate confidence score for feature

            conf_f = conf_f/skipped if skipped > 0 else 0 # scaled confidence level of feature match

            if conf_f > max_conf: # update max confidence level and associated feature
                max_conf, max_feat = conf_f, entry

        if max_conf > config.CONF_THRESHOLD:
            return max_feat # return result
        else:
            return None

    def match_arg_to_feature_value(self, feature, args):

        '''Return list of relevant arguments that match feature value'''

        print('(' + feature + ', ' + args + '): ' + 'match_arg_to_feature_value')

        if feature in config.name_ids:
            return [args]

        unique_vals = list(set(list(itertools.chain.from_iterable(config.DOMAIN_KNOWLEDGE[feature].values()))))
        vals = {key:0 for key in unique_vals} # initialize hash to accumulate evidence for each value of feature
        tokens = args.split() # tokenize argument
        max_val, max_arg = 0, []

        for token in tokens: # iterate over each token

            if self.is_number(token):
                return [token]

            for lookup in config.DOMAIN_KNOWLEDGE[feature]: # lookup relevant modifiers in domain knowledge

                jaro = jellyfish.jaro_distance(token, lookup) # compute string similarity of modifier vs query token

                # jaro normalized as a confidence between [0, 1]
                if jaro > config.CONF_THRESHOLD: # check if confidence is greater than preset threshold

                    # iterate over list of feature values that match the current modifier in domain knowledge
                    for i in range(len(config.DOMAIN_KNOWLEDGE[feature][lookup])):

                        vals[config.DOMAIN_KNOWLEDGE[feature][lookup][i]] += jaro # accumulate evidence for lookup in hash

                        if vals[config.DOMAIN_KNOWLEDGE[feature][lookup][i]] > max_val:

                            max_val = vals[config.DOMAIN_KNOWLEDGE[feature][lookup][i]] # update max confidence
                            max_arg = [config.DOMAIN_KNOWLEDGE[feature][lookup][i]] # update associated max argument

                        elif vals[config.DOMAIN_KNOWLEDGE[feature][lookup][i]] == max_val:

                            # accomodate for series of feature values that match the current modifier equally 
                            max_arg.append(config.DOMAIN_KNOWLEDGE[feature][lookup][i])

        return max_arg if max_arg else None
    
    def generate_features_rf_(self):
    
        '''Generate Bag-of-words of Relevant Features on Most Recent Query

        Current implementation supports suggestion of relevant features by relevance feedback (Rochio algorithm)
        Current implementation also supports suggestion of relevant features by frequent itemsets
        Defaulted to implementation of relevance feedback, with results cached for future reference

        Oracle caches features fetched over time and recalculates their weights in the cooccurence hash by
        time since last hit using an exponential decay
        '''

        qry_vector = defaultdict(float)
        rterms, nrterms = [], []

        # represents (term, features) matrix where weight for each (term, feature) is dependent on cooccurence strength
        cooccurence_hash = config.COOCCURENCE_HASH.copy()

        for term, steps in config.EXP_DECAY.items():
            cooccurence_hash[term][term] *= config.DECAY * (np.e ** (-config.DECAY * steps)) # update weights by exp decay

        for feat, term_wgts in cooccurence_hash.items(): # iterate over feature, weight pairs in cooccurence hash

            if feat in config.MOST_RECENT_QUERY: # construct query vector on terms in most recent query
                qry_vector[feat] += 1
                rterms.append(term_wgts)
            else:
                nrterms.append(term_wgts) # construct list of nonrelevant terms for Rochio

        reform = self.rochio_algo(qry_vector, rterms, nrterms, 1, 0.75, 0.15) # compute reformulated query vector by Rochio
        rterms = [] # reinitialize rterms to hold suggested terms to investigate

        for term in cooccurence_hash: # iterate over each term (key) in cooccurence hash
            cos_sim = self.cosine_sim(reform, cooccurence_hash[term]) # compute similarity of each vector in cooccurence_hash
            rterms.append((term, cos_sim)) # append to rterms

        rterms.sort(key=lambda x: x[1]) # sort rterms before inserting into cache

        # only include features deemed relevant over a preset threshold
        rterms = [term[0] for term in rterms if term[1] > config.RELEVANCE_FEEDBACK_THRESHOLD]

        return rterms

    def generate_queries_itemsets(self):

        '''Return association rules from frequent itemsets analysis of relevant features to investigate'''

        relim_input = itemmining.get_relim_input(config.ITEMSETS)
        item_sets = itemmining.relim(relim_input, min_support=config.ASSOC_MIN_SUPPORT) # generate frequent itemsets
        rules = assocrules.mine_assoc_rules(item_sets, min_support=config.ASSOC_MIN_SUPPORT, 
                                            min_confidence=config.ASSOC_MIN_CONFIDENCE) # generate association rules
        rules.sort(key=lambda x: -1 * x[2] * x[3]) # sort rules before inserting into cache

        return rules
    
    def rochio_algo(self, qry_vector, rel_terms, nonrel_terms, a, B, y):
    
        '''Relevance Feedback by Rochio Algorithm for Automated Query Suggestion

        Extension of original algorithm by caching results for future reference
        '''

        rels, nonrels = defaultdict(float), defaultdict(float)
        dr, dnr = len(rel_terms), len(nonrel_terms)

        # iterate over relevant feature set
        for rel_term in rel_terms: 

            if dr <= 0: break

            # iterate over each (feature, weight) tuple in rels
            for rel_key in rel_term: 
                # reweight each feature weight in relevant set
                rels[rel_key] += (B / dr) * rel_term[rel_key] 

        for nonrel_term in nonrel_terms: # iterate over nonrelevant feature set

            if dnr <= 0: break

            # iterate over each (feature, weight) tuple in nonrels
            for nonrel_key in nonrel_term: 
                # reweight each feature weight in nonrelevant set
                nonrels[nonrel_key] += (y / dnr) * nonrel_term[nonrel_key]  

        for term in qry_vector:
            # reweight initial query vector by alpha
            qry_vector[term] *= a 

        # initialize reformulated query vector
        reform = defaultdict(float) 

        for text in [qry_vector, rels]:
            for term in text:
                reform[term] += text[term]

        for nonrel_key in nonrels:

            if nonrel_key in reform:

                # offset reformulated query to drift from the nonrelevant set
                reform[nonrel_key] -= nonrels[nonrel_key] 

                if reform[nonrel_key] < 0:
                    # reset features with weights < 0 to 0 in reform
                    reform.pop(nonrel_key, None) 

        # return reformulated query vector
        return reform

    def cosine_sim(self, vec1, vec2, vec1_norm = 0.0, vec2_norm = 0.0):

        '''Return cosine similarity between two vectors'''

        if not vec1_norm:
            vec1_norm = sum(v * v for v in vec1.values())
        if not vec2_norm:
            vec2_norm = sum(v * v for v in vec2.values())

        # save some time of iterating over the shorter vec
        if len(vec1) > len(vec2):
            vec1, vec2 = vec2, vec1

        # calculate the inner product
        inner_product = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in vec1.keys())

        return inner_product / np.sqrt(vec1_norm * vec2_norm)

    def filter_reduction(self, f):
        reduction = {
            'by': '=>*(%by%)',
            'on': '=>*(%by%)',
            'when': '=>*(%when%)',
            'where': '=>*(%where%)'
        }

        if f in reduction:
            return reduction[f]
    
    def parse_modules(self, qry):
    
        stopwords = set(nltk.corpus.stopwords.words('english'))
        tokens = nltk.word_tokenize(qry)
        candidates = defaultdict(set)
        clookup = defaultdict(set)
        result = set()

        for token in tokens:

            mkeys = config.module_hash.keys()

            for mkey in mkeys:

                jaro = jellyfish.jaro_distance(token, mkey)

                if jaro > config.MODULE_PARSING_THRESHOLD:

                    matches = config.module_hash[mkey]

                    for match in matches:
                        candidates[match].add(mkey)
                        clookup[match].add(token)

        for module in candidates:

            mset = candidates[module]
            keys = config.modules_reversed[module]

            for key in keys:
                combined = set()
                keyset = key.split()
                combined |= set(keyset)
                combined &= mset

                if len(combined)/len(keyset) > config.CONF_THRESHOLD:
                    result.add(module)

        clookup  = {k: v for k, v in clookup.items() if k in result}

        return clookup
    
    def parse_query(self):
        tokens = nltk.word_tokenize(config.qry) # tokenize user query
        pos_tag = nltk.pos_tag(tokens) # apply part of speech tagger
        items = defaultdict(deque)
        for i, term in enumerate(pos_tag):
            items[term[0]].append(i)
        return pos_tag, items
    
    def extract_entities(self, pos_tag):
    
        actors, ind, verbs = [], [], []
        prev = False

        for i, tag in enumerate(pos_tag): # enumerate over each element of the tagged list

            tag = list(tag)

            if config.is_prep(tag[1]) or tag[0] in config.keywords: # check if token is a preposition or a keyword

                actors.append(tag)
                prev = False    

            elif config.is_genitive(tag) or config.is_verb(tag) or config.is_actor(tag[1]):

                if config.is_genitive(tag) or config.is_verb(tag): # check if tagged element is a possessive modifier or a verb phrase

                    if config.is_genitive(tag): 
                        tag[0] = config.GENITIVE # reset query token to be the genitive placeholder '->'
                    elif config.is_verb(tag):
                        verbs.append(i) # add to verbs list

                    actors.append(tag) # add to actors list
                    prev = False

                else:

                    if not prev:
                        actors.append(tag)
                        ind.append(i)
                    else:
                        actors[len(actors)-1][0] += ' ' + tag[0] # concatenate noun phrases with adjacent NNP

                    prev = True

            else:
                prev = False

        return actors, ind, verbs
    
    def group_actor_entities(self, pos_tag, actors, items, ind, verbs):
    
        # initialize actor index and verb index
        a_ind, v_ind = 0, 0 
        prev = False

        for i in range(len(ind)):

            if a_ind < len(actors) and v_ind < len(verbs) and config.is_verb(actors[a_ind]):

                index = verbs[v_ind] + 1

                while index < len(pos_tag) and config.is_adv(pos_tag[index][1]):
                    # group adverbs and verb phrases as a single entity
                    actors[a_ind][0] += ' ' + pos_tag[index][0] 
                    index += 1

                v_ind += 1

            while a_ind < len(actors) and not config.is_actor(actors[a_ind][1]):
                a_ind += 1

            index = items[actors[a_ind][0].split()[0]].popleft() - 1

            while ((a_ind < len(actors) and index >= 0) 
                   and config.is_desc(pos_tag[index][1])):

                # concatenate noun phrases with adjacent modifiers
                actors[a_ind][0] = pos_tag[index][0] + ' ' + actors[a_ind][0]
                index -= 1

            a_ind += 1

        negated = False
        for i, tag in enumerate(actors):
            if re.search(r'(?i)^NOT$', actors[i][0].lower()) is not None:
                negated = True
            if config.is_actor(tag[1]):
                if negated:
                    tag[0] = '(NOT{0})'.format(tag[0])
                else:
                    tag[0] = '({0})'.format(tag[0]) # wrap noun phrases in parenthesis for tagging
                negated = False

        # remove gerund and noun phrase modifers adjacent to the concatenated sets contructed above
        actors[:] = [actors[i] for i in range(len(actors)) if 
                     (not ((i+1) < len(actors) and 
                           (config.is_gerund(actors[i][1]) and 
                            config.is_actor(actors[i+1][1]))) and
                      re.search(r'(?i)^NOT$', actors[i][0].lower()) is None)]

        return actors
    
    def update_conjunctive_lookup(self, actors):
        config.conjunctive = {}
        latest_actor, conj = None, None
        for i in range(len(actors)):
            if config.is_actor(actors[i][1]):
                current = re.findall(r'\((.+?)\)', actors[i][0])[0]
                if current not in config.CORE and current not in config.FEATURE_DIST:
                    if conj is not None:
                        previous = re.findall(r'\((.+?)\)', latest_actor)[0]
                        if previous in config.conjunctive:
                            previous = config.conjunctive[previous][1]
                        config.conjunctive[current] = (conj, previous)
                        conj = None
                    latest_actor = actors[i][0]
            elif config.is_conj(actors[i][0].lower()) and latest_actor is not None:
                # lookup relevant keyword
                conj = config.keywords[actors[i][0].lower()]
                
    def handle_query(self, actors):
    
        prev_p, prev_a, prev_f, start_f, flag = False, False, False, False, -1 # set preposition, actor, filter flags
        immediate_f, latest_feat_for_arg, conj, conj_filter = False, None, None, None
        open_v = False # set flag to check if currently parsing a verb phrase
        extag = '' # parsed result
        pos_index = 0 # holds current position to edit result at in the finite state transduction
        for i in range(len(actors)):
            if not config.is_prep(actors[i][1]):
                immediate_f = False
            elif immediate_f: continue
            if start_f:
                if config.is_prep(actors[i][1]): continue
                else:
                    extag += actors[i][0].lower() + ')'
                    start_f = False
                    continue
            elif not extag and config.is_prep(actors[i][1]):
                reduction = self.filter_reduction(actors[i][0])
                if reduction is not None:
                    extag += reduction + '('
                    start_f = True
                continue
            if flag == 0:
                if re.search(r'(?i)^of$', actors[i][0].lower()) is None:
                    extag += '=>*(%filter%)(' + actors[i][0]
                    flag, prev_f = 1, True
                continue
            if config.is_verb(actors[i]): # check if token is a verb phrase
                op = config.keywords[actors[i][0]] # lookup relevant keyword
                if not op: # check if the token matches a preset token in keywords
                    op = '=>*(%' + actors[i][0].lower() + '%)' # substitute op with user specified action
                pos_index = len(extag)
                extag += op + '(' 
                open_v = True # set current parsing of verb phrase to true
            elif config.is_conj(actors[i][0].lower()): # check if token is a conjunction
                conj = config.keywords[actors[i][0].lower()] # lookup relevant keyword
                pos_index = len(extag)
                continue
            elif config.is_genitive(actors[i]): # check if token is a genitive phrase
                pos_index = len(extag)
                extag += actors[i][0]
                if prev_f: 
                    continue
            # check if preposition non-keyword preposition followers an actor
            elif config.is_actor(actors[i][1]) and prev_p:
                extag = extag[:pos_index] + actors[i][0] + extag[pos_index:] # switch order of actor and preposition
                prev_p = False # set current parsing of preposition to false
            elif config.is_prep(actors[i][1]) or actors[i][0].lower() in config.keywords: # check if token is a verb phrase
                immediate_f = True
                if conj is not None:
                    conj_filter = config.keywords[actors[i][0].lower()]
                    if not conj_filter:
                        conj_filter = '=>*(%' + actors[i][0].lower() + '%)'
                    continue
                if open_v: 
                    extag += ')' # close open verb tag
                    open_v = False
                # substitute with keyword representation encoded in domain knowledge
                op = config.keywords[actors[i][0].lower()]
                if not op:
                    op = '=>*(%' + actors[i][0].lower() + '%)' # substitute with user-specified token if not in keywords
                if op in config.filters:
                    if flag == 1:
                        # concatenate keyword representation to extag
                        extag = extag[:pos_index] + op + '(' + extag[pos_index:]
                        pos_index += len(op) + 1
                    else:
                        pos_index = len(extag)
                        extag += op + '('
                    prev_p, prev_f = False, True # not checking a prepositition but are checking a filter
                    continue
                extag = extag[:pos_index] + op + extag[pos_index:] # update result
                prev_p = True # set parsing of preposition to true
            else:
                if flag == 1: # flag marks prepositional clauses succeeding a verb phrase that it modifies
                    extag = extag[:pos_index] + actors[i][0] + ')' + extag[pos_index:]   
                pos_index = len(extag)
                if flag == 1:
                    flag = -1
                    if prev_f:
                        extag += ')'
                        prev_f = False
                    elif (i+1) < len(actors) and config.is_actor(actors[i+1][1]):
                        extag += '=>*(%filter%)(' # update result to accommodate parsed actions that require a parameter
                        prev_f = True # set parsing of filter to true
                    continue
                if conj is not None:
                    extracted = re.findall(r'\((.+?)\)', actors[i][0])[0]
                    extracted = self.match_(extracted)
                    if (extracted is not None and latest_feat_for_arg is not None 
                        and extracted == latest_feat_for_arg):
                        extag += conj
                    else:
                        if conj_filter is None or conj_filter == config.GENITIVE:
                            conj_filter = '=>*(%filter%)('
                            if flag != -1 or prev_f:
                                conj_filter = ')' + conj_filter
                        else: 
                            conj_filter = ')' + conj_filter + '('
                        extag += conj_filter + actors[i][0]
                        prev_f, conj_filter = True, None
                        continue
                extag += actors[i][0] # update result
                if prev_f and (i+1) < len(actors) and re.search(r'(?i)^of$', actors[i+1][0].lower()) is not None:
                    flag = 0
                    extag += ')'
                    pos_index = len(extag)
                    continue
            if (i+1) < len(actors) and config.is_conj(actors[i+1][0]): # ignore conjunctions that were handled above
                latest_feat_for_arg = re.findall(r'\((.+?)\)', actors[i][0])[0]
                latest_feat_for_arg = self.match_(latest_feat_for_arg)
                continue
            conj, conj_filter = None, None
            if prev_f: # check if modifiying a filtering substitution
                pos_index = len(extag)
                if ((i+1) < len(actors) and 
                    (config.is_genitive(actors[i+1]) or config.is_actor(actors[i+1][1]))):
                    if config.is_actor(actors[i+1][1]):
                        extag += ')=>*(%filter%)(' # accommodate action that requires a parameter
                    continue
                extag += ')'
            prev_f = False
        if open_v or prev_f:
            extag += ')'
        for s in config.subs: # substitute tokens in the parsed result that match tokens in subs
            extag = extag.replace(s, config.subs[s])

        return extag
    
    def generate_parse_lists(self, extag):

        f_list, e_list = [], [] # initialize filters, entities lists
        f_keys = [re.findall(r'\(%(.+?)%\)', f_key)[0] for f_key in config.filters] # extract only content
        f_keys = '(?:' + '|'.join(f_keys) + ')'
        # pattern match against any filter in domain knowledge
        pat = r'=>\*\(' + '%' + f_keys + '%' + r'\)\(\(.+?\)\)\)?' 
        pat_e = r'(' + pat + ')'
        f_list.append(re.findall(pat, extag))

        e_list = re.split(pat_e, extag)
        f_list = list(itertools.chain.from_iterable(f_list))
        e_list = [expr for expr in e_list if expr and re.search(pat_e, expr) is None]
        e_list = [ent for expr in e_list for ent in re.split(r'(\(.+?\)\)?)', expr) if ent] # holds non-filter entities
        pos_s, pos_e = -1, -1

        for i in range(len(e_list)):

            if e_list[i] == '=>*': # mark actions
                pos_s = i    
            elif re.search(r'\(\(.+\)\)', e_list[i]):
                pos_e = i    
            if pos_s > 0 and pos_e > 0:
                e_list[pos_s:(pos_e+1)] = [''.join(e_list[pos_s:(pos_e+1)])]
                pos_s, pos_e = -1, -1

        print('filtered: ', end='\t'); print(f_list)
        print('entity: ', end='\t'); print(e_list)

        return f_list, e_list
    
    def run(self, qry):
        
        '''Oracle's main driver

        Instantiate an instance of this class and call this method on
        a query to generate the Oracle's textual results and visualizations
        @return: result (data frame), sample sizes (data frame),
                 plot (Plotly object), rterms (relevance feedback suggestions) if parsing was successful
        @return: None otherwise (sample size of 0 on query)
        '''
        
        config.qry = qry
        config.clookup = self.parse_modules(config.qry)
        print('\nparsed modules: ' + str(config.clookup.keys()))
        # preprocessing
        pos_tag, items = self.parse_query()
        actors, ind, verbs = self.extract_entities(pos_tag)
        actors = self.group_actor_entities(pos_tag, actors, items, ind, verbs)
        # update info on conjunctions for query
        self.update_conjunctive_lookup(actors)
        # return filter logic in Oracle syntax
        extag = self.handle_query(actors)
        # generate parsed filter and entity lists
        f_list, e_list = self.generate_parse_lists(extag)
        # apply sequential filter
        config.filtered, b_feat = self.sequential_filter(f_list)
        # apply sequential entity filter
        # store results of query processing and generated visualization
        result, sample_size, plot, rterms = self.sequential_entity(e_list, b_feat, actors)
        return result, sample_size, plot, rterms