'''
After energy...

Introduce larger cooldown rate, and stop flocking predators.

Have flocking prey, huddle together, if all together cant be killed.

'''

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
MAX_VELOCITY=float(1.0)

class State(Enum):

    PREDATOR = auto()
    PREY = auto()

class Animal(Agent):

    last_update = -1
    max_prey = 0
    max_predators = 0
    predators = 30
    prey = 100
    total_frames = 0

    PREDATORS=30

    ALPHA=1.2
    BETA=0.6
    DELTA=0.005
    # GAMMA=0.25
    COOLDOWN_TIME = 60

    ENERGY_DECAY_RATE = 0.005
    ENERGY_GAIN_FROM_EATING = 0.5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.repocount=0
        self.predator_count=0
        self.state=State.PREY
        self.frame_count = 0
        self.check_interval = 60 
        self.cooldown=False
        self.cooldown_count = 0
        self.energy=1

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
            #print(cls.max_predators)
    
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

        location = self.in_shelter()

        if location == 'none':
            try:
                self.move=Vector2.normalize(self.move) 
            except:
                self.move=self.mo

            if self.state == State.PREY:

                self.neighbours=list(self.in_proximity_accuracy())
                self.predator_neighbours=[neighbour for neighbour in self.neighbours if neighbour[0].state == State.PREDATOR]
                self.predator_distances = [neighbour[1] for neighbour in self.predator_neighbours]
                self.prey_neighbours = [neighbour for neighbour in self.neighbours if neighbour[0].state == State.PREY]
                
                
                if self.predator_distances:
                    self.prey_die()
                
                if self.prey_neighbours:
                    n=len(self.prey_neighbours)
                    if n > 4:
                        n=4
                    self.move=Vector2.normalize(self.move) * 1/n
                    # for neigh in self.prey_neighbours:
                    #     if neigh[1] < 8:
                    #         try:
                    #             self.move += (self.pos - neigh[0].pos).normalize() 
                    #         except:
                    #             continue
                
            else:
                self.energy -= Animal.ENERGY_DECAY_RATE
            
            if self.frame_count >= self.check_interval:

                self.frame_count = 0

                if self.state ==  State.PREDATOR:

                    self.neighbours=list(self.in_proximity_accuracy())
                    self.prey_neighbours = [neighbour for neighbour in self.neighbours if neighbour[0].state == State.PREY]
                    self.prey_distances  = [neighbour[1] for neighbour in self.prey_neighbours]



                    if self.energy <= 0:
                        self.predator_die()
                        
                
                else:
                    if not self.predator_distances:
                        self.prey_reproduce()

        elif location == 'prey':
            if self.state == State.PREY:
                # self.move=Vector2.normalize(self.move) * 0.5
                None
            else:
                self.predator_die()

        else:
            if self.state == State.PREDATOR:
                # self.move=Vector2.normalize(self.move) * 0.5
                self.energy=1
            else:
                self.kill()
                Animal.prey-=1

                    
                
    def predator_die(self):
        # if random.uniform(0,1) < Animal.GAMMA:
        self.kill()
        Animal.predators-=1

    def prey_die(self):
        reduced_probability = len(self.prey_neighbours) 
        if reduced_probability ==0:
            reduced_probability=1
        predator=self.predator_neighbours[np.argmin(self.predator_distances)][0]
        if predator.cooldown == False:
            if random.uniform(0,1) < (Animal.BETA/reduced_probability):
                if (Animal.predators<=Animal.max_predators): #delta not probabilty but limit of increase per second
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
                predator.energy += Animal.ENERGY_GAIN_FROM_EATING
            else:
                prng = self.shared.prng_move
                deg = prng.uniform(150, 190)
                a=len(self.prey_neighbours)
                if a==0:
                    a=1
                elif a > 3:
                    a=3
                predator.move=Vector2.normalize(predator.move) * a
                predator.move.rotate_ip(deg)
                
            # Limit the energy to 1 (or whatever maximum you decide)
        if predator.energy > 1:
            predator.energy = 1


    def prey_reproduce(self):
        if Animal.prey<=Animal.max_prey and (random.uniform(0,1)<0.4): #this is a limit of increase
            prey_copy=self.reproduce()
            prey_copy.state=State.PREY
            prey_copy.frame_count = 0
            Animal.prey+=1
    
    def in_shelter(self):
        if 225<self.pos[0]<525 and 0<self.pos[1]<150:
            return 'predator'
        
        elif 225<self.pos[0]<525 and 600<self.pos[1]<750:
            return 'prey'
        
        else:
            return 'none'

DURATION=60*180

def graph():
    # for i in range(10):

    sim = Simulation(Config(movement_speed=1, radius=12, seed=1, duration=DURATION))
    sim.batch_spawn_agents(130, Animal, ["images/prey.png", "images/predator.png"])
    sim.spawn_site('images/shelter.png', 375, 0)
    sim.spawn_site('images/shelter-prey.png', 375, 750)
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

    plt.savefig('Assignment2/shelter/predator_prey_cooldown_huddling.png')

    # Change this part to create a phase-plane plot
    plt.figure()
    plt.plot(pivot_df[0], pivot_df[1])  # plot predator population against prey population

    plt.title('Phase-plane plot: Prey vs Predator')
    plt.xlabel('Prey Population')
    plt.ylabel('Predator Population')

    plt.savefig('Assignment2/shelter/predator_prey_phase_plane_huddling.png')

graph()