#imports
# from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum, auto
import pygame as pg
from pygame.math import Vector2
from vi import Agent, HeadlessSimulation, Simulation, Matrix
from vi.config import Config, dataclass, deserialize
import random
import math
from multiprocessing import Pool
import polars as pl
import time 

'''
Once a predator eats, delay one second before it can eat again
'''

predators_inital=0


class State(Enum):

    PREDATOR = auto()
    PREY = auto()

class Animal(Agent):

    last_update = -1
    max_prey = 0
    max_predators = 0
    predators = 10
    prey = 100
    total_frames = 0

    PREDATORS=10

    ALPHA=1.1
    BETA=0.4
    DELTA=0.1
    GAMMA=0.2
    COOLDOWN_TIME = 10

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.repocount=0
        self.predator_count=0
        self.state=State.PREY
        self.frame_count = 0
        self.check_interval = 60 
        self.cooldown=False
        self.cooldown_count = 0
        if predators_inital != Animal.PREDATORS:
            self.get_predators()

    
    @classmethod
    def update_max_values(cls):
        current_time = int(time.time()) 
        
        # print(current_time)
        if current_time != cls.last_update:

            # print('update')
            cls.max_prey = cls.prey*Animal.ALPHA
            cls.max_predators = cls.predators*cls.prey*Animal.DELTA
            cls.last_update = current_time
            # print(cls.prey)
            # print(cls.max_prey)
    
    def get_predators(self):
        global predators_inital
        # if self.predator_count != 10:
        self.state=State.PREDATOR
        self.change_image(1)
        predators_inital=predators_inital+1

    def change_position(self):
        super().change_position()
        global predators_inital

        if self.cooldown == True:
            self.cooldown_count+=1
            if self.cooldown_count == Animal.COOLDOWN_TIME:
                self.cooldown = False

        self.frame_count += 1

        Animal.total_frames += 1

        Animal.update_max_values()

        if self.state == State.PREY:

            self.neighbours=list(self.in_proximity_accuracy())
            self.predator_neighbours=[neighbour for neighbour in self.neighbours if neighbour[0].state == State.PREDATOR]
            self.predator_distances = [neighbour[1] for neighbour in self.predator_neighbours]
            
            if self.predator_distances:
                self.prey_die()
        
        if self.frame_count >= self.check_interval:

            self.frame_count = 0

            if self.state ==  State.PREDATOR:

                self.neighbours=list(self.in_proximity_accuracy())
                self.prey_neighbours=[neighbour for neighbour in self.neighbours if neighbour[0].state == State.PREY]
                self.prey_distances = [neighbour[1] for neighbour in self.prey_neighbours]

                if not self.prey_neighbours:
                    self.predator_die()
                    
            
            else:
                if not self.predator_distances:
                    self.prey_reproduce()
                
    def predator_die(self):
        if random.uniform(0,1) < Animal.GAMMA:
            self.kill()
            Animal.predators-=1

    def prey_die(self):
        predator=self.predator_neighbours[np.argmin(self.predator_distances)][0]
        if predator.cooldown == False:
            if random.uniform(0,1) < Animal.BETA:
                if Animal.predators<=Animal.max_predators: #delta not probabilty but limit of increase per second
                    predator_copy=predator.reproduce()
                    predator.cooldown=True
                    Animal.predators+=1
                    predator_copy.state=State.PREDATOR
                    predator_copy.change_image(1)
                    predator_copy.frame_count=0
                    prng = self.shared.prng_move
                    deg = prng.uniform(-30, 180)
                    predator_copy.move.rotate_ip(deg)
                self.kill()
                Animal.prey-=1
            predator.cooldown =True
            predator.cooldown_count = 0


    def prey_reproduce(self):
        if Animal.prey<=Animal.max_prey: #this is a limit of increase
            prey_copy=self.reproduce()
            prey_copy.state=State.PREY
            prey_copy.frame_count = 0
            Animal.prey+=1

DURATION=60*360
def visual():

    sim = Simulation(Config(movement_speed=1, seed=1, radius=12, duration=DURATION))
    sim.batch_spawn_agents(110, Animal, ["images/prey.png", "images/predator.png"])
    # sim.run()

    df = ((sim.run().snapshots))
    df = df.to_pandas()

    # create a pivot table with 'frame' as index, 'image_index' as columns and counts as values
    pivot_df = df.groupby(['frame', 'image_index']).size().unstack()

    pivot_df.index = pivot_df.index / 60

    # plot the data
    plt.plot(pivot_df.index, pivot_df[0], label='Prey population')  # 0 is the image_index for prey
    plt.plot(pivot_df.index, pivot_df[1], label='Predator population')  # 1 is the image_index for predator

    # Let's make our plot a little more attractive
    plt.title('Prey vs Predator population over time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Population')
    plt.legend()

    # plt.savefig('Assignment2/repro_direct/predator_prey_repro_360.png')

def graph():

    

    # for i in range(10):

    sim = HeadlessSimulation(Config(movement_speed=1, radius=12, seed=1, duration=DURATION))
    sim.batch_spawn_agents(110, Animal, ["images/prey.png", "images/predator.png"])
    sim.run()

    df = ((sim.run().snapshots))
    df = df.to_pandas()

    # create a pivot table with 'frame' as index, 'image_index' as columns and counts as values
    pivot_df = df.groupby(['frame', 'image_index']).size().unstack()

    pivot_df.index = pivot_df.index / 60

    # plot the data
    plt.plot(pivot_df.index, pivot_df[0], label='Prey population')  # 0 is the image_index for prey
    plt.plot(pivot_df.index, pivot_df[1], label='Predator population')  # 1 is the image_index for predator

    # Let's make our plot a little more attractive
    plt.title('Prey vs Predator population over time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Population')
    plt.legend()

    plt.savefig('Assignment2/predator_prey_first.png')

visual()
# graph()

# class Lotka_Volterra(object):

#     def __init__(self, y_0=[10, 1], parameters=[1.1, 0.4, 0.1, 0.4]):
#         self.y_0 = y_0
#         self.parameters = parameters
#         self.t = np.linspace(0, 50, num=1000)
#         self.solution = None

#     def get_differential(self, variables, t):
#         x, y = variables
#         alpha, beta, delta, gamma = self.parameters

#         dx_dt = alpha*x - beta*x*y
#         dy_dt = delta*x*y - gamma*y

#         return [dx_dt, dy_dt]

#     def solve_differential(self):
#         self.solution = odeint(self.get_differential, self.y_0, self.t)
#         return self.solution

#     def plot_graph(self):
#         if self.solution is None:
#             print("Please solve the differential equations first.")
#             return

#         f, (ax1, ax2) = plt.subplots(2)

#         ax1.plot(self.t, self.solution[:,0], color='b')
#         ax1.set_ylabel('Rabbit population')

#         ax2.plot(self.t, self.solution[:,1], color='r')
#         ax2.set_ylabel('Fox population')
#         ax2.set_xlabel('Time')

#         plt.savefig('Assignment2/Rabbit-Fox.png')

# lotka=Lotka_Volterra()
# lotka.plot_graph()
# @deserialize
# @dataclass
# class PPConfig(Config):
#     alpha: float = 1.1
#     beta: float = 0.4
#     delta: float = 0.1
#     gamma: float = 0.4


# sim.batch_spawn_agents(100, Rabbit, ["images/white.png"])


# class Fox(Agent):

#     config: PPConfig

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.state = State.PREDATOR
#         self.repocount=0

#     def change_position(self):
#         if not self._moving:
#             return

#         changed = self.there_is_no_escape()

#         prng = self.shared.prng_move

#         # Always calculate the random angle so a seed could be used.
#         deg = prng.uniform(-30, 30)

#         # Only update angle if the agent was teleported to a different area of the simulation.
#         if changed:
#             self.move.rotate_ip(deg)

#         self.neighbours=list(self.in_proximity_accuracy())
#         self.prey_neighbours=[neighbour for neighbour in self.neighbours if neighbour[0].state == State.PREY]
#         self.prey_distances = [neighbour[1] for neighbour in self.prey_neighbours]

#         self.pos+=self.move

#         hold_repo=self.repocount
#         for _ in range(self.repocount):
#             if random.uniform(0,1) < self.config.delta:
#                 self.reproduce()
#             hold_repo-=1          
#         self.repocount=hold_repo

#         if not self.prey_neighbours:
#             self.die()
        
        
        
#     def die(self):
#         if random.uniform(0,1) < self.config.gamma:
#             self.kill()

# class Rabbit(Agent):

#     config: PPConfig

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.state = State.PREY

#     def change_position(self):

#         if not self._moving:
#             return

#         changed = self.there_is_no_escape()

#         prng = self.shared.prng_move

#         # Always calculate the random angle so a seed could be used.
#         deg = prng.uniform(-30, 30)

#         # Only update angle if the agent was teleported to a different area of the simulation.
#         if changed:
#             self.move.rotate_ip(deg)
        
#         self.pos+=self.move

#         self.neighbours=list(self.in_proximity_accuracy())
#         self.predator_neighbours=[neighbour for neighbour in self.neighbours if neighbour[0].state == State.PREDATOR]
#         self.predator_distances = [neighbour[1] for neighbour in self.predator_neighbours]
#         if self.predator_distances:
#             self.die()
#         else:
#             self.areproduce()
    
#     def die(self):
#         if random.uniform(0,1) < self.config.beta:
#             predator=self.predator_neighbours[np.argmin(self.predator_distances)][0]
#             predator.repocount+=1
#             self.kill()

#     def areproduce(self):
#         if random.uniform(0,1) < (self.config.alpha-1):
#             self.reproduce()




# lv_model = Lotka_Volterra()
# lv_model.solve_differential()
# lv_model.plot_graph()


