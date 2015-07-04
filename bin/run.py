#! /usr/bin/env python
import numpy as np
import pandas as pd
import itertools
import operator
import random 
import pickle

import sklearn.cluster
import sklearn.neighbors

import deap

from funcs import *

from deap import algorithms, base, creator, tools
#from scoop import futures
import multiprocessing


LAT_LONG_RESULTS_FILENAME = "data/results_latlong.pickle"
RESULTS_LATLONG = pickle.load(open(LAT_LONG_RESULTS_FILENAME, "rb"))

NUM_CLUSTERS = 8
X = RESULTS_LATLONG[-1]["X"]

MU = 10
LAMBDA = 0

def main():
	# Register everything
	creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
	creator.create("Individual", list, fitness=creator.FitnessMin)

	toolbox = base.Toolbox()
	
	pool = multiprocessing.Pool()
	toolbox.register("map", pool.map)
	#toolbox.register("map", futures.map)

	toolbox.register("rand_bit", random.randint, 0, NUM_CLUSTERS)
	toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.rand_bit, n=X.shape[0])
	toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=MU)

	toolbox.register("evaluate", evalAssignment, X=X)
	toolbox.register("mate", mateAssignments, X=X, num_clusters=NUM_CLUSTERS)
	toolbox.register("mutate", mutateAssignment, indpb=0.05, num_clusters=NUM_CLUSTERS)
	toolbox.register("select", tools.selTournament, tournsize=3)



	# Run the algorithm
	print("generating population...")
	pop = toolbox.population()
	    
	hof = tools.HallOfFame(1)

	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("avg", np.mean)
	stats.register("min", np.min)
	stats.register("max", np.max)
	    
	print("Starting the algorithm")
	pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.1, ngen=10, stats=stats, halloffame=hof, verbose=True)

	pickle.dump({ "pop": pop, "logbook": logbook, "hof": hof }, open('results.pickle', 'wb'))



# DEAP functions

def evalAssignment(assignment, X):
    # needs X
    x = X.copy()
    x['cluster_id'] = list(assignment)
    return (weighted_avg_std(x), )

def calculate_connections(rows):
    rows = rows.copy()
    rows['pct_neig_same_cluster'] = pct_neig_same_cluster(rows)
    rows = rows.join(
        rows.groupby("cluster_id").score.mean(),
        on="cluster_id",
        rsuffix="_cluster"
    )
    rows['weighted_score'] = punishment_factor(rows.pct_neig_same_cluster) * (rows.score - rows.score_cluster)**2
    
    cross = calculate_cross(rows)
    cross = cross[cross.index_x < cross.index_y]
    res = cross[["index_x", "index_y"]]
    res["diff"] = abs(cross["weighted_score_x"]-cross["weighted_score_y"])
    return res


def generateConnectionList(ind, X):
    x = X.copy()
    x['cluster_id'] = list(ind)
    conns = x.groupby("cluster_id").apply(
        lambda rows: 
            calculate_connections(rows)
    )
    return conns.sort("diff").reset_index()[conns.columns.get_level_values(0)]

    
    
def mateAssignments(ind1, ind2, X, num_clusters):
    # needs: X, num_clusters
    # generate ordered list of 'connections'
    conns1 = generateConnectionList(ind1, X).iterrows()
    conns2 = generateConnectionList(ind2, X).iterrows()

    # iterate both alternately, adding connections and forming the new assignment
    new_ind = pd.Series(index=X.index)
    
    all_conns = itertools.cycle([conns1, conns2])
    
    while new_ind.isnull().any() or len(new_ind[new_ind.notnull()].unique()) > num_clusters:
        conns = next(all_conns)
        conn = next(conns, None)
        while(conn is not None):
            conn = conn[1]
            index_x = conn["index_x"]
            index_y = conn["index_y"]
            

            if not np.isnan(new_ind[index_x]) and new_ind[index_x] == new_ind[index_y]:
                # they are already in the same cluster. skip
                conn = next(conns, None)
            else: 
                conn = None
                if np.isnan(new_ind[index_x]) and np.isnan(new_ind[index_y]):
                    # both are not assigned yet. assigning both to a new cluster
                    new_cluster_id = new_ind.max() + 1 
                    if np.isnan(new_cluster_id): new_cluster_id = 0
                    new_ind[[index_x, index_y]] = new_cluster_id

                elif not np.isnan(new_ind[index_x]) and not np.isnan(new_ind[index_y]):
                    # in two different clusters. combine them to one
                    new_ind[new_ind==new_ind[index_y]] = new_ind[index_x]

                elif not np.isnan(new_ind[index_x]): 
                    new_ind[index_y] = new_ind[index_x]
                else:
                    new_ind[index_x] = new_ind[index_y]
    
    return creator.Individual(new_ind), creator.Individual(new_ind)
    

def mutateAssignment(ind, indpb, num_clusters):
    # needs mutpb, num_clusters
    
    for i in range(len(ind)):
        if random.random() < indpb:
            ind[i] = random.randint(0,num_clusters)
    
    return ind,
    
if __name__ == "__main__":
	main()    

