#! /usr/bin/env python
import numpy as np
import pandas as pd
import itertools
import operator
import random 
import pickle
import logging 


import sklearn.cluster
import sklearn.neighbors

import deap

from funcs import *

from deap import algorithms, base, creator, tools
#from scoop import futures
import multiprocessing


logging.basicConfig(filename='run.log', filemode='w', level=logging.DEBUG, format='%(asctime)s\t%(levelname)s\t%(message)s')



LAT_LONG_RESULTS_FILENAME = "data/results_latlong.pickle"
RESULTS_LATLONG = pickle.load(open(LAT_LONG_RESULTS_FILENAME, "rb"))

NUM_CLUSTERS = 8
X = RESULTS_LATLONG[-1]["X"]

X = X[:500]

# init with KMeans
km = sklearn.cluster.KMeans(n_clusters=NUM_CLUSTERS)
kmeans_ind = km.fit_predict(X)


MU = 100
LAMBDA = 50

def main():
	# Register everything
	creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
	creator.create("Individual", list, fitness=creator.FitnessMin)

	toolbox = base.Toolbox()
	
	pool = multiprocessing.Pool()
	toolbox.register("map", pool.map)
	
	toolbox.register("rand_bit", random.randint, 0, NUM_CLUSTERS)
	toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.rand_bit, n=X.shape[0])
	toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=MU)

	toolbox.register("evaluate", evalAssignment, X=X, num_clusters=NUM_CLUSTERS)
	toolbox.register("mate", mateAssignments, X=X, num_clusters=NUM_CLUSTERS)
	toolbox.register("mutate", mutateAssignment, indpb=0.05, num_clusters=NUM_CLUSTERS)
	toolbox.register("select", tools.selTournament, tournsize=3)


	# Run the algorithm
	print("generating population...")
	#pop = toolbox.population()
	pop = [ creator.Individual(kmeans_ind) ] * (MU-1)
	pop = [ creator.Individual(kmeans_ind) ] + [ toolbox.mutate(toolbox.clone(ind))[0] for ind in pop ]

	hof = tools.HallOfFame(10)

	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("avg", np.mean)
	stats.register("min", np.min)
	stats.register("max", np.max)
	stats.register("list", list)
	stats.register("num_uniq", lambda pop: len(set([ str(x) for x in pop])) )
		

	print("Starting the algorithm")
	#pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.75, mutpb=0.1, ngen=5, stats=stats, halloffame=hof, verbose=True)
	# pop, logbook = algorithms.eaMuPlusLambda(
	# 	pop, toolbox,
	# 	mu=MU, lambda_=LAMBDA, 
	# 	cxpb=0.8, mutpb=0.2, 
	# 	ngen=5, stats=stats, halloffame=hof, verbose=True
	# 	)
	population = pop
	mu=MU
	lambda_=LAMBDA
	cxpb = 0.9
	mutpb = 0.1
	ngen=50
	halloffame = hof
	verbose = True

	logbook = tools.Logbook()
	logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

	# Evaluate the individuals with an invalid fitness
	invalid_ind = [ind for ind in population if not ind.fitness.valid]
	fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
	for ind, fit in zip(invalid_ind, fitnesses):
		ind.fitness.values = fit

	if halloffame is not None:
		halloffame.update(population)

	record = stats.compile(population) if stats is not None else {}
	logbook.record(gen=0, nevals=len(invalid_ind), **record)
	if verbose:
		print(logbook.stream)

	# Begin the generational process
	for gen in range(1, ngen + 1):
		# Vary the population
		# print("new gen. generate offspring...")
		offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

		# Evaluate the individuals with an invalid fitness
		invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
		# print("evaluate {} invalid individuals from {} offsprings".format(len(invalid_ind), len(offspring)))
		fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
		# print("settings fitness values...")
		for ind, fit in zip(invalid_ind, fitnesses):
			ind.fitness.values = fit

		# Update the hall of fame with the generated individuals
		# print ("updating hof...")
		if halloffame is not None:
			halloffame.update(offspring)

		# Select the next generation population
		# print ("select {} individuals from pop+offspring as the new population".format(mu))
		population[:] = toolbox.select(population + offspring, mu)

		# Update the statistics with the new population
		# print("record statistics...")
		record = stats.compile(population) if stats is not None else {}
		logbook.record(gen=gen, nevals=len(invalid_ind), **record)
		if verbose:
			print(logbook.stream)

	
	pickle.dump({ "pop": pop, "logbook": logbook, "hof": hof }, open('results.pickle', 'wb'))

	

# DEAP functions

def varOr(population, toolbox, lambda_, cxpb, mutpb):
	"""Part of an evolutionary algorithm applying only the variation part
	(crossover, mutation **or** reproduction). The modified individuals have
	their fitness invalidated. The individuals are cloned so returned
	population is independent of the input population.
	:param population: A list of individuals to vary.
	:param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
									operators.
	:param lambda\_: The number of children to produce
	:param cxpb: The probability of mating two individuals.
	:param mutpb: The probability of mutating an individual.
	:returns: The final population
	:returns: A class:`~deap.tools.Logbook` with the statistics of the
						evolution
	The variation goes as follow. On each of the *lambda_* iteration, it
	selects one of the three operations; crossover, mutation or reproduction.
	In the case of a crossover, two individuals are selected at random from
	the parental population :math:`P_\mathrm{p}`, those individuals are cloned
	using the :meth:`toolbox.clone` method and then mated using the
	:meth:`toolbox.mate` method. Only the first child is appended to the
	offspring population :math:`P_\mathrm{o}`, the second child is discarded.
	In the case of a mutation, one individual is selected at random from
	:math:`P_\mathrm{p}`, it is cloned and then mutated using using the
	:meth:`toolbox.mutate` method. The resulting mutant is appended to
	:math:`P_\mathrm{o}`. In the case of a reproduction, one individual is
	selected at random from :math:`P_\mathrm{p}`, cloned and appended to
	:math:`P_\mathrm{o}`.
	This variation is named *Or* beceause an offspring will never result from
	both operations crossover and mutation. The sum of both probabilities
	shall be in :math:`[0, 1]`, the reproduction probability is
	1 - *cxpb* - *mutpb*.
	"""
	assert (cxpb + mutpb) <= 1.0, ("The sum of the crossover and mutation "
																 "probabilities must be smaller or equal to 1.0.")

	offspring = []
	for i in range(lambda_):
		# print("varOr: {}".format(i))
		op_choice = random.random()
		if op_choice < cxpb:            # Apply crossover
			ind1, ind2 = map(toolbox.clone, random.sample(population, 2))
			# print("crossover: {} , {}".format(ind1,ind2))
			ind1, ind2 = toolbox.mate(ind1, ind2)
			del ind1.fitness.values
			offspring.append(ind1)
		elif op_choice < cxpb + mutpb:  # Apply mutation
			# print ("mutation")
			ind = toolbox.clone(random.choice(population))
			ind, = toolbox.mutate(ind)
			del ind.fitness.values
			offspring.append(ind)
		else:                           # Apply reproduction
			# print("reproduction")
			offspring.append(random.choice(population))

	return offspring



def evalAssignment(assignment, X, num_clusters):
	logging.debug("evaluating %s", assignment)
	# needs X
	x = X.copy()
	x['cluster_id'] = list(assignment)
	return (weighted_avg_std(x, num_clusters), )

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

	
def next_cluster_id(ind):
	new_cluster_id = ind.max() + 1
	if np.isnan(new_cluster_id): 
		new_cluster_id = 0
	return new_cluster_id
	
def mateAssignments(ind1, ind2, X, num_clusters):
	logging.debug("mating %s and %s", ind1, ind2)
	# needs: X, num_clusters
	# generate ordered list of 'connections'
	conns1 = generateConnectionList(ind1, X).iterrows()
	conns2 = generateConnectionList(ind2, X).iterrows()

	# iterate both alternately, adding connections and forming the new assignment
	new_ind = pd.Series(index=X.index)
	
	all_conns = itertools.cycle([conns1, conns2])
	exhausted_ctr = 0
	
	while exhausted_ctr<2 and (new_ind.isnull().any() or len(new_ind[new_ind.notnull()].unique()) > num_clusters):
		conns = next(all_conns)
		conn = next(conns, None)
		if conn is None:
			exhausted_ctr += 1
			
			while(conn is not None):
				conn = conn[1]
				index_x = conn["index_x"]
				index_y = conn["index_y"]

				if not np.isnan(new_ind[index_x]) and new_ind[index_x] == new_ind[index_y]:
					# they are already in the same cluster. skip
					conn = next(conns, None)
					if conn is None:
						exhausted_ctr += 1
					else: 
						conn = None
						if np.isnan(new_ind[index_x]) and np.isnan(new_ind[index_y]):
							# both are not assigned yet. assigning both to a new cluster
							new_cluster_id = next_cluster_id(new_ind)
							new_ind[[index_x, index_y]] = int(new_cluster_id)

						elif not np.isnan(new_ind[index_x]) and not np.isnan(new_ind[index_y]):
							# in two different clusters. combine them to one
							new_ind[new_ind==new_ind[index_y]] = new_ind[index_x]

						elif not np.isnan(new_ind[index_x]): 
							new_ind[index_y] = new_ind[index_x]
						else:
							new_ind[index_x] = new_ind[index_y]
									
	# fill nans with unused cluster_ids
	leftovers = list(range(next_cluster_id(new_ind), num_clusters))[:sum(new_ind.isnull())]
	new_ind[new_ind.isnull()][:len(leftovers)] = leftovers
	new_ind[new_ind.isnull()] = np.random.randint(num_clusters, size=sum(new_ind.isnull()))

	return creator.Individual(new_ind), creator.Individual(new_ind)


def mutateAssignment(ind, indpb, num_clusters):
	logging.debug("mutating %s", ind)
	# needs mutpb, num_clusters
	
	for i in range(len(ind)):
		if random.random() < indpb:
			ind[i] = random.randint(0,num_clusters)
	
	return ind,
	
if __name__ == "__main__":
	main()    

