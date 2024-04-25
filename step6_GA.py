#IMPORTAR LIBRERÍAS
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from random import random, sample, choice, uniform
from math import floor
from tqdm import tqdm
from numpy import array, dot, mean
from numpy.linalg import pinv
from sys import exit

#REGRESIÓN LINEAL MÚLTIPLE
def multiple_linear_regression(sel_desc):
    """
    Perform MLR model, calculating R2, coefficients and interception.
    """
    #Cambiar el nombre del archivo de data_training
    inputs = (data_training[list(sel_desc)].astype(float)).values.tolist()
    #outputs = data_training["MeanValue"].values.tolist()
    outputs = data_training["log(EC3)"].values.tolist()
    #outputs = data_training["LnValue"].values.tolist()
    model = LinearRegression().fit(inputs,outputs)
    n = len(outputs)
    r_sq = model.score(inputs,outputs)
    ad_r_sq = 1-((1-r_sq)*(n-1)/(n-individual_size-1))
    intercept = model.intercept_
    coefficients = model.coef_
    cv_error = -cross_val_score(model, inputs, outputs, scoring='neg_mean_squared_error').mean()
    return {"R2": r_sq, "Intercept": intercept, "Coefficients": coefficients, "Selection": sel_desc, "Cross_Val": cv_error, "Predictibilidad": ad_r_sq}

#FUNCIONES DEL ALGORITMO GENÉTICO
def create_individual(individual_size):
    """
    Create an individual: The function chooses randomly "individual_size" descriptors from the descriptor list.
    """
    return [choice(descriptors) for _ in range(individual_size)]

def create_population(individual_size, population_size):
    """
    Create an initial population: The function creates "population_size" individuals from create_individual function
    """
    return [create_individual(individual_size) for i in range(population_size)]

def evaluate_population(population):
    """
    Evaluate a population of individuals and return the best among them.
    1/ Apply MLR algorithm to individual selection in population.
    2/ Sort the results from best to worst -> Select the "selection_size" best results.
    4/ Add the best results to best_individuals list.
    """
    fitness_list = [multiple_linear_regression(individual)
                    for individual in tqdm(population)]
    R_list = sorted(fitness_list, key=lambda i: i["Cross_Val"])
    best_individuals = R_list[: selection_size]
    return best_individuals

def crossover(parent_1, parent_2):
    """
    Return offspring given two parents.
    Unlike real scenarios, genes in the chromosomes aren't necessarily linked.
    1/ loci_1: Random selection of the 60% of the individual_size
    2/ loci_2: The other numbers within individual size
    3/ chromosome_1 and Chro = Change numbers by coefficient in this position
    4/ child: new individual with a combination of chromosomes form their parents
    """
    child = {}
    loci = [i for i in range(0, individual_size)]
    loci_1 = [choice(loci) for _ in range(floor(0.6*(individual_size)))]
    chromosome_1 = [[i, parent_1["Selection"][i]] for i in loci_1]
    chromosome_1_values = [value for _, value in chromosome_1]
    available_values = [val for val in parent_2["Selection"] if val not in chromosome_1_values]
    chromosome_2 = []
    for i in loci:
      if i not in loci_1:
        if available_values:
            value = choice(available_values)
            available_values.remove(value)
        else:
            value = choice(parent_2["Selection"])
        chromosome_2.append([i, value])        
    child.update({key: value for (key, value) in chromosome_1})
    child.update({key: value for (key, value) in chromosome_2})
    return [child[i] for i in loci]

def mutate(individual):
    """
    Mutate an individual.
    The gene transform decides whether we'll add or deduct a random value.
    1/ Random selection of the position for the mutated coefficient
    2/ Selection of the descriptors which does not exist in the individual
    3/ Mutation for a new descriptor
    """
    loci = [i for i in range(0, individual_size)]
    no_of_genes_mutated = floor(probability_of_gene_mutating*individual_size)
    loci_to_mutate = sample(loci, no_of_genes_mutated)
    for count, locus in enumerate(loci_to_mutate):
        desc_to_mutation = [descriptor for descriptor in descriptors if descriptor not in individual]
        mutation = sample(desc_to_mutation, no_of_genes_mutated)
        individual[locus] = mutation[count]
    return individual

def get_new_generation(selected_individuals):
    """
    Given selected individuals, create a new population by mating them.
    Here we also apply variation operations like mutation and crossover.
    1/ Select 2 individuals from "selected individuals" (which are the best ones from the last step) to create a pair "population_size" times
    2/ For each pair, the crossover function creates a new individual who is a mix of the coefficients from the original individuals.
    3/ Selection of a specific amount of the new individuals to perform a mutation
    4/ Mutate a specific % of this individual (Usually 1 of the coefficient will be randozed again)
    5/ New population will be the crossover offspring and a % of mutated offspring
    """
    parent_pairs = [sample(selected_individuals, 2)
                    for i in range(population_size)]
    offspring = [crossover(pair[0], pair[1]) for pair in parent_pairs]
    offspring_indices = [i for i in range(population_size)]
    offspring_to_mutate = sample(
        offspring_indices,
        floor(probability_of_individual_mutating*population_size)
    )
    mutated_offspring = [[i, mutate(offspring[i])]
                         for i in offspring_to_mutate]
    for child in mutated_offspring:
        offspring[child[0]] = child[1]
    return offspring


#IMPORT DATA
print("Importing data...")
data_training = pd.read_excel("alergenos_training_set_1_OUT.xlsx")
data_test = pd.read_excel("Alergenos_test_set_1.xlsx")
data_training.index = data_training["COMP"]
data_test.index = data_test["COMP"]

descriptors = ("BCUT.4", "BCUT.2", "petitjeanShapeIndex.1", "CPSA.21", "WHIM.3", "MaxEStateIndex", "Weta2.unity", "WNSA-1", "Wlambda2.unity", "types.GeometricalRadius",
 "WD.unity", "ALogP", "ALogp2", "ATSC3c", "ATSC1e", "ATSC3i", "AATSC3m", "AATSC2i", "MATS2m", "MATS1s", "MATS3s", "GATS1c", "GATS2c", "GATS3e", "VE3_Dzp", "VR3_Dzi",
  "SM1_Dzs", "VR1_Dzs", "SpMin5_Bhs", "AVP-1", "gmin", "BIC3", "TDB3m", "TDB3i", "WPSA-326", "RNCS33", "GRAV-444", "RDF35u", "RDF15s", "E1m", "Dv", "De", "E2p", "E2i",
   "E1s", "Ks"
)

#DATA NORMALIZATION
print("Normalizing data...")
mean_dict = {}
std_dict = {}
for desc in descriptors:
    desc_mean = np.mean(data_training[desc])
    desc_std = np.std(data_training[desc])
    z_score = [(x-desc_mean)/desc_std for x in data_training[desc]]
    mean_dict[desc] = desc_mean
    std_dict[desc] = desc_std
    data_training[desc] = z_score

mean_zscore = pd.DataFrame(mean_dict, index=[0])
std_zscore = pd.DataFrame(std_dict, index=[0])

#GA-MLR
top10 = []

individual_size = 9 #Número de descriptores
population_size = 10000 #Número de combinaciones
selection_percentage = 0.20 #Porcentaje de combinaciones para recombinar
selection_size = floor(selection_percentage*population_size) #Combinaciones que pasan a la siguiente fase
max_generations = 15 #Máximo de iteraciones
probability_of_individual_mutating = 0.20 #Probabilidad de combinaciones susceptibles a mutación
probability_of_gene_mutating = 3/9 #Descriptores mutados dentro de cada selección
initial_population = create_population(individual_size, 50000) Nueva población
current_population = initial_population
generation_count = 0
print("Starting Genetic Algorithm")

#Nos da las 10 mejores (criterio a verificar en la definición de la función)
while generation_count <= max_generations:
    print('Generation: ', generation_count, " OF ", max_generations)
    best_individuals = evaluate_population(current_population)
    current_population = get_new_generation(best_individuals)
    best10 = best_individuals[: 10]
    for each in best10:
        top10.append([each, {"Generation": generation_count}])
    top10 = sorted(top10, key=lambda x: x[0]['R2'], reverse=True)
    top10 = top10[0:10]
    generation_count += 1
else:
    print("___FINAL___")
    print("MEJORES 10 RESULTADOS: ")
    for c, top in enumerate(top10):
        print("Top ", c+1)
        print(top)



