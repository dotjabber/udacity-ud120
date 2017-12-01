#!/usr/bin/python

import sys
import pickle

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test
from tester import dump_classifier_and_data
from utils import cleanAttributes
from utils import createRatioAttribute

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi" (ok, i moved the poi attribute for a particular reason ;)
feature_attributes = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                      'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                      'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock',
                      'director_fees', 'to_messages', 'from_poi_to_this_person',
                      'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

## Throw out points that are not valid
data_dict.pop("TOTAL", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)

# Clean the data and add some attributes based on ratio
cleanAttributes(data_dict)
feature_attributes.append(createRatioAttribute(data_dict, 'from_poi_to_this_person', 'from_messages', 'from_poi_ratio'))
feature_attributes.append(createRatioAttribute(data_dict, 'from_this_person_to_poi', 'to_messages', 'to_poi_ratio'))

feature_attributes.append(createRatioAttribute(data_dict, 'salary', 'total_payments', 'salary_total_ratio'))
feature_attributes.append(createRatioAttribute(data_dict, 'bonus', 'total_payments', 'bonus_total_ratio'))
feature_attributes.append(createRatioAttribute(data_dict, 'long_term_incentive', 'total_payments', 'long_term_ratio'))
feature_attributes.append(createRatioAttribute(data_dict, 'deferred_income', 'total_payments', 'deferred_total_ratio'))
feature_attributes.append(
    createRatioAttribute(data_dict, 'deferral_payments', 'total_payments', 'deferral_total_ratio'))
feature_attributes.append(createRatioAttribute(data_dict, 'loan_advances', 'total_payments', 'loan_advances_ratio'))
feature_attributes.append(createRatioAttribute(data_dict, 'other', 'total_payments', 'other_ratio'))
feature_attributes.append(createRatioAttribute(data_dict, 'expenses', 'total_payments', 'expenses_ratio'))
feature_attributes.append(createRatioAttribute(data_dict, 'director_fees', 'total_payments', 'director_fees_ratio'))

feature_attributes.append(
    createRatioAttribute(data_dict, 'restricted_stock_deferred', 'total_stock_value', 'restricted_deferred_ratio'))
feature_attributes.append(
    createRatioAttribute(data_dict, 'exercised_stock_options', 'total_stock_value', 'exercised_options_ratio'))
feature_attributes.append(createRatioAttribute(data_dict, 'restricted_stock', 'total_stock_value', 'restricted_ratio'))


def build(individual):
    my_dataset = data_dict

    feature_list = ['poi']
    for i in range(0, len(individual)):
        if individual[i] == 1:
            feature_list.append(feature_attributes[i])

    data = featureFormat(my_dataset, feature_list, sort_keys=True)
    labels, features = targetFeatureSplit(data)

    from sklearn import preprocessing
    mms = preprocessing.MinMaxScaler()
    features = mms.fit_transform(features)

    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    from sklearn.naive_bayes import GaussianNB

    clf = GaussianNB()
    pca = PCA()

    pipe = Pipeline(steps=[('pca', pca), ('gaussian', clf)])

    return pipe, labels, features, feature_list

def eval(individual):
    clf, labels, features, feature_list = build(individual)
    true_negatives, false_negatives, true_positives, false_positives, \
    total_predictions, accuracy, precision, recall, f1, f2 = test(clf, labels, features)

    return f1,

import random

random.seed(64)

from deap import base
from deap import creator
from deap import tools

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(feature_attributes))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", eval)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

pop = toolbox.population(n=20)
print("Start of evolution")

# Evaluate the entire population
fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

# Extracting all the fitnesses of
fits = [ind.fitness.values[0] for ind in pop]

for g in range(1, 30):
    print("-- Generation %i --" % g)

    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):

        # cross two individuals with probability CXPB
        if random.random() < 0.5:
            toolbox.mate(child1, child2)

            # fitness values of the children
            # must be recalculated later
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:

        # mutate an individual with probability MUTPB
        if random.random() < 0.2:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(invalid_ind))

    # The population is entirely replaced by the offspring
    pop[:] = offspring

    # Gather all the fitnesses in one list and print the stats
    fits = [ind.fitness.values[0] for ind in pop]

    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x * x for x in fits)
    std = abs(sum2 / length - mean ** 2) ** 0.5

    print("  Min %s" % min(fits))
    print("  Max %s" % max(fits))
    print("  Avg %s" % mean)
    print("  Std %s" % std)

best_ind = tools.selBest(pop, 1)[0]
print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

clf, labels, features, feature_list = build(best_ind)
dump_classifier_and_data(clf, data_dict, feature_list)
