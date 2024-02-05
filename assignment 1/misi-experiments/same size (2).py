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

A=1.70188
B=3.58785

class State(Enum):
    JOINING = auto()
    LEAVING = auto()
    STILL = auto()
    WANDERING = auto()

class Cockroach(Agent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.PJOIN = 0.8
        # self.PLEAVE = 0.0001
        self.TJOIN = 10
        self.TLEAVE = 30
        self.ticker = 0
        self.state = State.WANDERING
        self.move_save=[]

    def change_position(self):

        neighbours=list(self.in_proximity_accuracy())
        n = len(neighbours)
        self.PJOIN = 0.03 + 0.48 * (1-(math.e**(-A*n)))
        self.PLEAVE=(math.e)**(-B*n)
        # print('PJOIN: {}, PLEAVE: {}'.format(self.PJOIN, self.PLEAVE))

        changed = self.there_is_no_escape()

        prng = self.shared.prng_move

        # Always calculate the random angle so a seed could be used.
        deg = prng.uniform(-30, 30)

        # Only update angle if the agent was teleported to a different area of the simulation.
        if changed:
            self.move.rotate_ip(deg)

        if self.state == State.WANDERING:
            #small rectangle and big rectangle bound check
            if ((135 < self.pos[0] < 315) and (285 < self.pos[1] < 465)) or ((435 < self.pos[0] < 615) and (285 < self.pos[1] < 465)):     
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
        
        if self.state==State.LEAVING:
            if self.ticker == self.TLEAVE:
                self.state=State.WANDERING
                self.ticker=0
            else:
                self.ticker+=1
                self.pos+=self.move

# some constants
SIMULATION_NR = 10
POPULATION = 100
DURATION  = 10*60 # n times 60, which is the default frame settings per second
# coordinates of zone 1
X1 = 225
Y1 = 375
# coordinates of zone 2
X2 = 525
Y2 = 375
#storing the zone member count calculations 
zone1 = []
zone2 = []

#defining the simulation
def simulation():
    sim=HeadlessSimulation(Config(movement_speed=5, duration=DURATION, seed=i))
    sim.batch_spawn_agents(POPULATION, Cockroach, ["images/green.png", 'images/red.png'])
    sim.spawn_site('images/bubble-full.png', X1, Y1)
    sim.spawn_site('images/bubble-full.png', X2, Y2)    
    return sim

#running n experiments
for i in range(SIMULATION_NR):
    #setting up the simulation
    sim = simulation()

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
            zone1_dist = math.sqrt((X1-X)**2+(Y1-Y)**2)
            zone2_dist = math.sqrt((X2-X)**2+(Y2-Y)**2)

            if zone1_dist > zone2_dist:
                zone2_count += 1
            else:
                zone1_count += 1

    zone1.append(zone1_count)
    zone2.append(zone2_count)

data = pd.DataFrame({'zone 1': zone1, 'zone 2': zone2})

data['experiments'] = range(1, len(data) + 1)
data = data.melt(id_vars='experiments', var_name='population', value_name='count')

# Create the relplot
sns.relplot(x='experiments', y='count', hue='population', data=data, kind='line')

#calculating the mean lines
mean_zone1 =  round(sum(zone1) / len(zone1), 2)
mean_zone2 =  round(sum(zone2) / len(zone2), 2)

#adding the mean lines to the plot
plt.axhline(mean_zone1, color='b', linestyle='--', label=f'Mean for zone 1: {mean_zone1}')
plt.axhline(mean_zone2, color='orange', linestyle='--', label=f'Mean for zone 2: {mean_zone2}')

#setting the range for the population
plt.ylim(0, POPULATION)

#add the legend and display the plot
plt.legend()
plt.show()
