import numpy as np
import scipy
from scipy import stats
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from sklearn.base import BaseEstimator, ClusterMixin
from oracle.modules.module import Module
import collections
import pickle

class PitchersLikeMe(Module):

    def set_module(self):
        return True
    
    def get_lexicon(self):
        result = set()
        result.add('pitchers-like')
        result.add('similar-pitchers')
        return result
    
    '''if self.is_filter return an index set else return a table'''
    def execute(self, iset, data, params={}, is_grouped=False):
        if is_grouped or 'pitcher' not in params: return []
        pitcher = params['pitcher']
        data = data.iloc[iset,:]
        rset = []
        if data.shape[0] > 0:
            keep = ['pitcher', 'pitch_type', 'release_spin_rate', 'release_pos_x', 'release_pos_y', 'release_pos_z',
                    'release_extension', 'px', 'pz', 'effective_speed', 'p_throws']
            kmeans_pitch_types = ['CH', 'CU', 'FC', 'FB', 'KC', 'SI', 'SL']
            keys = {'release_spin_rate': np.mean, 'release_pos_x': np.mean, 'release_pos_y': np.mean, 'release_pos_z': np.mean,
                    'release_extension': np.mean, 'px': np.mean, 'pz': np.mean, 
                    'effective_speed': np.mean, 'p_throws': np.mean, 'ncount': np.sum}
            pitch_threshold = 50
            pitchers = data[['pitcher', 'pitcher_name']].drop_duplicates()
            df = data[keep]
            df = df.fillna(df.mean())
            if df.shape[0] > 0:
                throws = df['p_throws']
                throws = [1 if throws[i] == 'R' else 0 for i in range(len(throws))]
                df['p_throws'] = throws
                df['ncount'] = 1
                X = df.groupby(['pitch_type', 'pitcher'], as_index=False).agg(keys)
                X = X[X['ncount'] > pitch_threshold]
                del X['ncount']
                pitch_types = X.pitch_type.unique()
                pitch_types = list(set(pitch_types) & set(kmeans_pitch_types))
                scaler = pickle.load(open("pitcher_clusters_scaler.p","rb"))
                kmeans = pickle.load(open("pitcher_clusters_kmeans.p","rb"))
                X.iloc[:,2:] = scaler.fit_transform(X.iloc[:,2:])
                clusters = pd.DataFrame()
                for pitch_type in pitch_types:
                    group = X[X['pitch_type'] == pitch_type]
                    pitch = group.pitcher
                    group = group.drop(['pitch_type','pitcher'], 1)
                    labels = kmeans[pitch_type].predict(group)
                    result = pd.DataFrame()
                    result['pitcher'], result['cluster'], result['pitch_type'] = pitch, labels, pitch_type
                    result = pd.merge(result, pitchers, how='inner')
                    result = result.drop_duplicates()
                    clusters = clusters.append(result)
                current = clusters.loc[clusters['pitcher_name'] == pitcher]
                current = current[['cluster', 'pitch_type']]
                matched = pd.merge(clusters, current, how='inner')
                matched = matched[['pitcher', 'pitch_type']]
                matched = pd.merge(data, matched, how='inner')
                rset = matched.index.tolist()
        return rset