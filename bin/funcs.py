import math
import numpy as np
import operator
import random 

DIST_COLUMNS = ['long', 'lat']


def calculate_cross(X):
    res = X.copy()
    res["fictive_key"] = 1
    cross = res.reset_index().merge(res.reset_index(), on="fictive_key")
    cross["dist"] = (
        (
            cross[[ c+"_x" for c in DIST_COLUMNS]].rename(columns={ c+"_x": c for c in DIST_COLUMNS}) - 
            cross[[ c+"_y" for c in DIST_COLUMNS]].rename(columns={ c+"_y": c for c in DIST_COLUMNS})
        )**2).sum(axis=1).apply(np.sqrt)
    return cross
    
def distance_to_clusters(cross):
    clusters_dist = cross.groupby(["index_x", "cluster_id_y"]).dist.apply(
        lambda x : x.sort(inplace=False)[:int(ceil(x.shape[0]/10))].mean()
    )
    d = pd.DataFrame(clusters_dist).reset_index().pivot("index_x", "cluster_id_y")
    d.columns = d.columns.get_level_values(1)
    return d

def punishment_factor(pct_neig_same_cluster):
    return 9999*(1-pct_neig_same_cluster)/np.exp(13*pct_neig_same_cluster)+1
    

def calc_cluster_weighted_avg_std(X, pct_neig_same_cluster):
    X = X.copy()
    X['pct_neig_same_cluster'] = pct_neig_same_cluster
    
    mean = X.score.mean()
    tot = sum( punishment_factor(X.pct_neig_same_cluster) * (X.score - mean)**2 )
    return math.sqrt(tot/(len(X)-1))

def weighted_avg_std(X, num_clusters):
    pct_neig_same_cluster_x = pct_neig_same_cluster(X)
    
    weighted_std = X.groupby("cluster_id").apply(
        lambda rows: calc_cluster_weighted_avg_std(rows, pct_neig_same_cluster_x)
    ).mean()
    
    sizes = X.groupby("cluster_id").size()
    
    if sum(sizes >= 1/num_clusters/2) < num_clusters:
        return 99999999 # a lot.
    else:
        return weighted_std

def pct_neig_same_cluster(X):
    K = int(max(10, len(X)*0.01))
    cross = calculate_cross(X)
    pct_same_cluster_neighbors = cross.groupby("index_x").apply(
        lambda neighbors : sum(neighbors.sort("dist")[1:K+1].cluster_id_y == neighbors.cluster_id_x.iloc[0])/K
    )
    pct_same_cluster_neighbors.name = "pct_same_cluster"
    return pct_same_cluster_neighbors



