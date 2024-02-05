from enum import Enum, auto
import pygame as pg
from pygame.math import Vector2
from vi import Agent, HeadlessSimulation
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
import differentsizeall as agent_all
import differentsizelimit as agent_limit
import differentsizeorigin as agent_origin
import differentsizestacking as agent_stacking
import differentsizestackinglimit as agent_stackinglimit


class SimulationFitness(object):

    def __init__(self, agent, name):
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
        self.agent = agent
        self.name = name

    def simulation(self, i):
        
        sim=HeadlessSimulation(Config(movement_speed=5, duration=self.DURATION, seed=i, radius=36))
        
        sim.batch_spawn_agents(self.POPULATION, self.agent, images=["images/green.png", "images/red.png"])
        sim.spawn_site('images/bubble-small.png', self.X1, self.Y1)
        sim.spawn_site('images/bubble-full.png', self.X2, self.Y2)    
        return sim

    def fitness(self):


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

        data = pd.DataFrame({'small zone': self.zone1, 'large zone': self.zone2})

        data['experiments'] = range(1, len(data) + 1)
        data = data.melt(id_vars='experiments', var_name='population', value_name='count')

        plt.figure(figsize=(12, 8))
        plt.title("Agent: '{}'".format(self.name))

        # Use lineplot instead of relplot
        sns.lineplot(x='experiments', y='count', hue='population', data=data)

        # Adding the mean lines to the plot
        plt.axhline(mean_zone1, color='b', linestyle='--', label=f'Mean for small zone: {mean_zone1}')
        plt.axhline(mean_zone2, color='orange', linestyle='--', label=f'Mean for large zone: {mean_zone2}')

        # Setting the range for the population
        plt.ylim(0, self.POPULATION)

        # Add the legend and display the plot
        plt.legend()
        plt.savefig('{}.png'.format(self.name))

        return (mean_zone2/mean_zone1+1)

agents= [[agent_all.Cockroach, 'all'], [agent_limit.Cockroach, 'limit'], [agent_origin.Cockroach, 'origin'], [agent_stacking.Cockroach, 'stacking'], [agent_stackinglimit.Cockroach, 'stackinglimit']]

for agent in agents:
    fit=SimulationFitness(agent[0], agent[1])
    print('\n{}'.format(agent[1]))
    print(fit.fitness())

'''
flocking(with-all)
4.779661016949152

limit
3.3358208955223883

origin
3.207017543859649

stacking
3.2606837606837606

stacking&limit
3.2467532467532467

'''

