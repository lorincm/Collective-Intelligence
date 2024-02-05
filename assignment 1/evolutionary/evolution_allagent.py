from enum import Enum, auto
import pygame as pg
from pygame.math import Vector2
from vi import Agent, Simulation, HeadlessSimulation
from vi.config import Config, dataclass, deserialize
import random
import math
import polars as pl
import seaborn as sns
from multiprocessing import Pool
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class SimulationFitness(object):

    def __init__(self):
        self.DURATION = 20*60
        self.SIMULATION_NR = 10
        self.POPULATION = 100
        self.zone1 =[]
        self.zone2 = []
        # coordinates of zone 1
        self.X1 = 225
        self.Y1 = 375
        # coordinates of zone 2
        self.X2 = 525
        self.Y2 = 375
    
    def split_params(self, params):
        self.params = params.copy()
        self.params = params
        print(self.params)
        self.alpha=int(self.params.tolist()[0])
        self.beta=int(self.params.tolist()[1])
        self.gamma=int(self.params.tolist()[2])

    def simulation(self, i):
        cockroach_config = CockroachConfig()
        cockroach_config.alpha = self.alpha
        cockroach_config.beta= self.beta
        cockroach_config.gamma= self.gamma

        cockroach_agent_partial = partial(Cockroach, config=cockroach_config)
        
        sim=HeadlessSimulation(Config(movement_speed=5, duration=self.DURATION, seed=i, radius=36))
        
        sim.batch_spawn_agents(self.POPULATION, cockroach_agent_partial, images=["images/green.png", "images/red.png"])
        sim.spawn_site('images/bubble-small.png', self.X1, self.Y1)
        sim.spawn_site('images/bubble-full.png', self.X2, self.Y2)    
        return sim
    
    def fitness(self, params):

        self.split_params(params)

        for i in range(self.SIMULATION_NR):

            sim = self.simulation(i)

            #running the simulation and saving the dataframe
            df = ((sim.run().snapshots))

            #get the last n rows of the dataset, where n is the population
            df = df.tail(100)
            df = df.to_pandas()
            zone1_count = 0
            zone2_count = 0

            for i in range(len(df)):
                nth_row = df.loc[i]
                #dont forget to add that if image_index == 1
                if nth_row.loc['image_index'] == 1:
                    X, Y = nth_row.loc['x'], nth_row.loc['y']
                    zone1_dist = math.sqrt((self.X1-X)**2+(self.Y1-Y)**2)
                    zone2_dist = math.sqrt((self.X2-X)**2+(self.Y2-Y)**2)

                    if zone1_dist > zone2_dist:
                        zone2_count += 1
                    else:
                        zone1_count += 1

            self.zone1.append(zone1_count)
            self.zone2.append(zone2_count)
        mean_zone1 =  round(sum(self.zone1) / len(self.zone1), 2)
        mean_zone2 =  round(sum(self.zone2) / len(self.zone2), 2)

        return (mean_zone2/mean_zone1+1)

class CockroachEA(object):

    def __init__(self, simulation_fitness, pop_size, tournament_size, mutation_threshold, crossover="one", bounds_min=None, bounds_max=None):
        self.pop_size = pop_size
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max
        self.tournament_size = tournament_size
        self.mutation_threshold = mutation_threshold
        self.simulation_fitness = simulation_fitness
        self.crossover = crossover
    
    def sorter(self, fitness, genes):
        fitness=np.array(fitness)
        genes=np.array(genes)
        args = fitness.argsort()
        
        fitness = fitness[args]
        genes = genes[args]
        
        return fitness, genes

    def parent_selection(self, x_old, f_old):

        x_parents = x_old #population array of arrays which contain 4 floats as the genes
        f_parents = f_old #fitness values (float)
    
            
        f_parents, x_parents = self.sorter(f_parents, x_parents)
        

        f_parents, x_parents = f_parents[0:50], x_parents[0:50]
        return x_parents, f_parents

    def recombination(self, x_parents, f_parents):
        x_children = []
        if self.crossover == "one":
            for i in range(len(x_parents)-1):
                # see the else statement for explanation for the random number generation
                cross_point = np.random.randint(1,4)                
                #by the chosen point we add up the two genes
                x_sub = x_parents[i][0:cross_point]
                x_sub2 = x_parents[i+1][cross_point:]
                x_sub = np.append(x_sub, x_sub2)                       
                x_children.append(x_sub)
        else:
            for i in range(len(x_parents)-1):
                x_sub = []
                # inner loop as we need to decide on 4 part of the genes
                for t in range(3):
                    x_sub.append(x_parents[i+np.random.randint(0,2)][t])
                x_children.append(x_sub)
        #lets keep the numpy array property, as up until to this poit we worked with normal arrays
        return np.asarray(x_children)

    def mutation(self, x_children, f_old):
       # Select 25% of the children randomly for mutation

        if len(f_old) > 1 and f_old[-1] <= f_old[-2]:
            self.mutation_threshold *= 1.05  # Increase mutation rate by 5%
        elif self.mutation_threshold > 0.01:  # Lower limit for mutation rate
            self.mutation_threshold *= 0.95  # Decrease mutation rate

        for i in range(len(x_children)):
            for t in range(3):
                #we draw a random sample from a uniform distribution between 0 and 1
                #if its below the threshold we mutate a gene randomly according to the corresponding boundaries
                if np.random.uniform(0,1)<self.mutation_threshold:
                    x_children[i][t] = np.random.uniform(low=self.bounds_min[t], high=self.bounds_max[t])
        
        return x_children

    def survivor_selection(self, x_old, x_children, f_old, f_children):
        num_elites = int(0.1 * len(x_old))  # For example, keep 10% of the old population as elites
        
        # Combine the old and new generation
        x = np.concatenate([x_old, x_children])
        f = np.concatenate([f_old, f_children])
    
        # Sort individuals based on fitness
        f, x = self.sorter(f, x)

        # Select the top individuals to form the new generation, including the elites
        x, f = x[0:self.pop_size - num_elites], f[0:self.pop_size - num_elites]
        
        # Add elites
        x = np.concatenate([x, x_old[:num_elites]])
        f = np.concatenate([f, f_old[:num_elites]])
        return x, f
        

    def evaluate(self, x):
        fitnesses = []
        for params in x:
            fitness = self.simulation_fitness.fitness(params)
            fitnesses.append(fitness)
        return fitnesses

    def step(self, x_old, f_old):
        # Parent selection
        x_parents, f_parents = self.parent_selection(x_old, f_old)

        # Recombination
        x_children = self.recombination(x_parents, f_parents)

        # Mutation
        x_children = self.mutation(x_children, f_old)

        # Evaluation
        f_children = self.evaluate(x_children)
        # print(x_old)
        # print(x_children)
        # print(f_old)
        # print(f_children)

        # Survivor selection
        x, f = self.survivor_selection(x_old, x_children, f_old, f_children)

        return x, f

class State(Enum):
    
    JOINING = auto()
    LEAVING = auto()
    STILL = auto()
    WANDERING = auto()

class CockroachConfig(Config):

    alpha: float = 0.5
    beta: float = 0.5
    gamma: float = 0.5

class Cockroach(Agent):

    def __init__(self, config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.PJOIN = 0.8
        # self.PLEAVE = 0.0001
        self.TJOIN = 13
        self.TLEAVE = 34
        self.ticker = 0
        self.state = State.WANDERING
        self.move_save=[]
        self.A=1.70188
        self.B=3.58785
        self.config=config

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def change_position(self):
        neighbours=list(self.in_proximity_accuracy())
        n = len(neighbours)
        self.PJOIN = 0.03 + 0.48 * (1-(math.e**(-self.A*n)))

        if n<=3:
            self.PLEAVE=(math.e)**(-self.B*n)
        else:
            self.PLEAVE=(math.e)**(-self.B*3)

        # print('PJOIN: {}, PLEAVE: {}'.format(self.PJOIN, self.PLEAVE))

        changed = self.there_is_no_escape()

        prng = self.shared.prng_move

        # Always calculate the random angle so a seed could be used.
        deg = prng.uniform(-30, 30)

        # Only update angle if the agent was teleported to a different area of the simulation.
        if changed:
            self.move.rotate_ip(deg)

        if self.state == State.WANDERING and self.in_proximity_accuracy().count() >= 1:
            neighbours=list(self.in_proximity_accuracy())
            #allignment
            sum_v=Vector2()
            sum_x=Vector2()
            sum_difference=Vector2()
            alpha: float = self.config.alpha
            beta: float = self.config.beta #separation
            gamma: float = self.config.gamma
            delta_time: float = 3
            mass: int = 5
            MAX_VELOCITY=float(3)

            for i in neighbours:

                object=i[0]

                v=object.move
                sum_v+=v

                x=object.pos
                sum_x+=x

                sum_difference+=(self.pos-x)
            

            n=len(neighbours)
            avg_v=sum_v/n
            alignment_force=avg_v-self.move

            avg_x=sum_x/n
            cohesion_force=avg_x-self.pos
            # c=cohesion_force-self.move

            separation=sum_difference/n

            total=(alpha*alignment_force)+(beta*separation)+(gamma*cohesion_force)
            ftotal=total/mass

            self.move += ftotal

            if Vector2.length(self.move) > MAX_VELOCITY:
                self.move=Vector2.normalize(self.move) * MAX_VELOCITY

            self.pos=self.pos+(self.move*delta_time)

        if self.state == State.WANDERING:
            #small rectangle and big rectangle bound check
            if ((185 < self.pos[0] < 265) and (335 < self.pos[1] < 415)) or ((440 < self.pos[0] < 610) and (290 < self.pos[1] < 460)):     
                if random.uniform(0, 1) <= self.PJOIN:
                    self.state = State.JOINING
                else:
                    self.pos+=self.move
            else:
                self.pos += self.move
        
        if self.state == State.JOINING:
            if self.on_site() or (self.ticker == self.TJOIN):
            # if self.ticker == self.TJOIN:
                self.state=State.STILL
                self.ticker=0
            else:
                self.ticker +=1
                self.pos+=self.move

        if self.state==State.STILL:
            if len(self.move_save) == 0:
                self.move_save.append(self.move)
            self.move=Vector2()
            self.change_image(1)
            # self.PLEAVE=math.e**(-B*n)
            # print(self.PLEAVE)
            if random.uniform(0, 1) <= self.PLEAVE:
                self.move=self.move_save[0]
                # print(self.move)
                self.ticker=0
                self.state = State.LEAVING
                self.change_image(0)
        
        if self.state==State.STILL:
            for neighbour in neighbours:
                if neighbour[1] <= 8 and neighbour[0].state == State.STILL:
                    self.move=self.move_save[0]
                    self.ticker=0
                    self.state = State.LEAVING
                    self.change_image(0)
        
        if self.state==State.LEAVING:
            if self.ticker == self.TLEAVE:
                self.state=State.WANDERING
                self.ticker=0
            else:
                self.ticker+=1
                self.pos+=self.move
def run():
    # PLEASE DO NOT REMOVE!
    num_generations = 25  # if necessary, please increase the number of generations
    pop_size = 12
    bounds_min = [0.2, 0.05, 0.05]
    bounds_max = [0.7, 0.4, 0.4]

    tournament_size=3
    mutation_threshold=20

    simulation_fitness=SimulationFitness()

    ea = CockroachEA(simulation_fitness=simulation_fitness, pop_size=pop_size, tournament_size=tournament_size, mutation_threshold=mutation_threshold, bounds_min=bounds_min, bounds_max=bounds_max)
    # -------

    # Init the population
    x = np.random.uniform(low=bounds_min, high=bounds_max, size=(pop_size, 3))
    f = ea.evaluate(x)

    populations = []
    populations.append(x)

    f_best = [max(f)]
 

    index=f.index(max(f))
    x_best = [x[index]]

    # Run the EA.
    for i in range(num_generations):
        print("Generation: {}, best fitness: {:.2f}".format(i, max(f)))
        x, f = ea.step(x, f)
        populations.append(x)
        f_best.append(max(f.tolist()))
        index=f.tolist().index(max(f.tolist()))
        x_best.append(x[index])       
        

    print("FINISHED!")
    print(f_best)
    print(x_best)
    return(f_best, x_best)

def plot_evolution(x_best, f_best):
    x_best_params = np.array(x_best)
    tjoin = x_best_params[:, 0]
    tleave = x_best_params[:, 1]
    radius = x_best_params[:, 2]
    
    data = pd.DataFrame({
        'Generation': list(range(len(x_best))),
        'alpha': tjoin,
        'beta': tleave,
        'gamma': radius,
        'Fitness': f_best,
    })
    
    plt.figure(figsize=(10, 6))
    plt.title("Evolution of Flocking Parameters and Fitness over Generations")
    sns.lineplot(data=data, x='Generation', y='alpha', label='alpha')
    sns.lineplot(data=data, x='Generation', y='beta', label='beta')
    sns.lineplot(data=data, x='Generation', y='gamma', label='gamma')
    plt.ylabel("Parameter Values")  # Add label to the left y-axis
    plt.legend()
    
    plt.twinx()
    sns.lineplot(data=data, x='Generation', y='Fitness', color='black', label='Fitness')
    plt.legend(loc='lower right')
    
    plt.savefig('evolution_2.png')  # save the plot to a file


if __name__ == "__main__":
    f_best, x_best=run()
    plot_evolution(x_best, f_best)


'''
[3.095576619273302, 3.0523076923076924, 3.04232995658466, 3.0411650485436894, 3.0309734513274336, 3.0293767368003173, 3.0243068742878845, 3.0232380952380953, 3.022239263803681, 3.020745131244708, 2.976595744680851, 2.8808066258552394, 2.879811930649427, 2.879811930649427, 2.879811930649427, 2.879811930649427]
[array([23.03948534, 46.21393022, 52.066076  ]), array([17.54799104, 46.58590765, 48.48866979]), array([ 6.87950634, 42.59853928, 37.59730712]), array([ 7.15280705, 45.33365442, 55.42091965]), array([12.9530004 , 43.68014356, 49.38255377]), array([37.15963342, 46.96362073, 17.02009257]), array([36.74130469, 26.16051486, 44.22357039]), array([23.7054651 , 26.23804032, 39.56630417]), array([31.08626888, 16.59004177, 32.31001292]), array([32.36531737, 43.89712503, 15.62249202]), array([29.38383218, 57.52838428, 52.95644456]), array([11.14087894, 51.3047105 , 20.89441527]), array([ 6.58785468, 51.191933  , 29.20828696]), array([ 6.58785468, 51.191933  , 29.20828696]), array([ 6.58785468, 51.191933  , 29.20828696]), array([ 6.58785468, 51.191933  , 29.20828696])]

'''