import pandas as pd
from deap import creator, base, tools
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import random



# diavazoume to arxeio me ta kathara dedomena
df = pd.read_csv('data_sitting.csv')

# arxikopoiisi genetikou
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
otherclasses=['sittingdown', 'standing', 'standingup', 'walking']

# pairnoume to synolo
def create_individual():
    return [df[col].sample().values[0] for col in df.columns]


def evaluate(individual):
    c=0.1
    s2=0
    for cc in otherclasses:
        s2=s2+cosine(individual, trg[cc])
    s1= cosine(individual, trg['sitting'])
    
    return (s1+c*(1-0.25*s2))/(1+c)
#    return euclidean(individual,trg['sitting']) 

#target sto sitting

trg={}


trg['sitting']= [ 0.6112520814799609,
   0.4048549417065527,
   0.7470855771205881,
  0.6046642427894137,
   0.6173349565419396,
   0.7820436469113999,
  0.5185653987714872,
   0.4232498937708882,
  0.5108366979175695,
   0.7138902095107355,
  0.29457471109044747,
  0.4557490959051028]
  
trg['sittingdown']= [ 0.5670803969005361,
   0.5745345534869088,
   0.3552190288936883,
   0.5481784521277702,
  0.5973426519206603,
   0.6305378626706898,
   0.47587952065539707,
  0.507469159811245,
   0.5167665930367081,
   0.39716476225274455,
   0.501579781986077,
  0.7260267737942099]
 
 
trg['standing']= [0.5029862366888191,
   0.5957097649182623,
   0.4085219896136874,
   0.6376596656751479,
   0.719348288786277,
   0.6432218888706867,
   0.5314379158483169,
   0.5226106017914491,
   0.5432018535259941,
   0.43868641143999443,
   0.5912388124614663,
   0.5510371748504502]

trg['standingup']= [ 0.47282936957342614,
   0.5339810454789352,
   0.3921790153033061,
   0.6040267093086397,
   0.6439368835377118,
   0.6817737618024219,
   0.5219605303797933,
   0.4401966077589575,
   0.5892168428337742,
   0.3834863205579063,
   0.5243219318491341,
   0.5520842407875222]

trg['walking']= [ 0.5095713825330305,
   0.5939543509111143,
   0.3953562998036087,
   0.5782693594860466,
   0.6518107999914347,
   0.5984338934762028,
   0.5080996370749912,
   0.5460656978617889,
   0.47075291533880576,
   0.39209437951953496,
   0.5700846193489693,
   0.4080129238040945]


toolbox.register("attribute", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=12)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


toolbox.register("evaluate", evaluate)  # evaluation
toolbox.register("mate", tools.cxTwoPoint)  # diastavrosi
toolbox.register("mutate", tools.mutFlipBit, indpb=0.2)  # metallaxi
toolbox.register("select", tools.selTournament, tournsize=3)  #epilogi
    

# pairnoume population 100 tyxaia digmata 
population_size = 20
#arithmos geneon
generations = 1000
cx_prob = 0.1
mut_prob = 0.01

# arxikos plithismos
population = toolbox.population(n=population_size)
best2=population[0]
k=0
apo=[]
# ektelesi genetikou
for generation in range(generations):
    # ypologismo katalilotitas
    fitnesses = [toolbox.evaluate(individual) for individual in population]

    # katalilotita gia kathe atomo
    for individual, fitness in zip(population, fitnesses):
        individual.fitness.values = (fitness,)

    # goneis gia diastavrosi
    selected = toolbox.select(population, len(population))
    offspring = [toolbox.clone(individual) for individual in selected]

    # diastavrosi
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < cx_prob:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # metalaxi
    for mutant in offspring:
        if random.random() < mut_prob:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # axiologisi
    invalid_individuals = [individual for individual in offspring if not individual.fitness.valid]
    fitnesses = [toolbox.evaluate(individual) for individual in invalid_individuals]

    for individual, fitness in zip(invalid_individuals, fitnesses):
        individual.fitness.values = (fitness,)

    # ananeosi plithismou
    population[:] = offspring
    best1=min(population, key=lambda individual: individual.fitness.values[0])
    apo.append(best1.fitness.values[0])
    if(euclidean(best1,best2)==0):
        k=k+1
    else:
        k=0
    best2=best1
    if(k>10):
        break
    
# apotelesma tou plithismou
best_individual = min(population, key=lambda individual: individual.fitness.values[0])

print("best result:")
print(best_individual)
print("evaluation:")
print(best_individual.fitness.values[0])


maxx=[16,136,-15,250,295,122,122,212,-4,-92,-53,-134]
minx=[-29,41,-171,-426,-462,-560,-87,-3,-184,-244,-132,-186]
print("Teliki thesi afou epanaferoume apo kanonikopoiisi:")
for i in range(12):
    x=best_individual[i]
    print(x*(maxx[i]-minx[i])+minx[i])

print(k,generation)
plt.plot(apo)
plt.show()